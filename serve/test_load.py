"""serve/test_load.py - GET /v1/load: the load figure clients use to hold back work that is not urgent.

    python -m unittest serve.test_load -v
"""
from __future__ import annotations

import collections
import json
import sys
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import serve.server as server  # noqa: E402
from serve.server import Service, read_load_peak  # noqa: E402
from serve.test_image_alone import engine_without_process, run  # noqa: E402


def service(batch=16, decoding=0, reading=0, peak=660.0, waiting=0):
    s = object.__new__(Service)
    s.engine = SimpleNamespace(batch=batch, waiting=waiting)
    s.status_lock = threading.Lock()
    s.status = {"busy": False}
    now = time.time()
    s.live_reqs = {}
    for i in range(decoding + reading):
        first = now - 10 if i < decoding else None
        st = {"busy": True, "started": now - 12, "first_token": first, "generated": 100 if first else 0}
        rate = [(now - 2, 0), (now, 40)] if first else []   # 20 tok/s each
        s.live_reqs[i] = (st, rate)
    s.rate = []
    s.load_peak, s.load_peak_path, s._load_peak_saved = peak, None, (0.0, 0.0)
    return s


def on(e, peak=660.0):
    """A Service over an engine with the real admission state (test_image_alone's engine without a process)."""
    s = object.__new__(Service)
    s.engine = e
    s.status_lock = threading.Lock()
    s.status = {"busy": False, "queued": 0}
    s.live_reqs, s.rate = {}, []
    s.load_peak, s.load_peak_path, s._load_peak_saved = peak, None, (0.0, 0.0)
    return s


def register(s, tok_s=None):
    """A request as Service._run registers it - before engine.generate, so one that waits is in live_reqs too."""
    now = time.time()
    st = {"busy": True, "started": now - 30, "first_token": (now - 20) if tok_s else None,
          "generated": int(20 * tok_s) if tok_s else 0}
    rate = collections.deque(maxlen=32)
    if tok_s:
        for k in range(32):
            rate.append((now - (31 - k) / tok_s, st["generated"] - 31 + k))
    s.live_reqs[id(st)] = (st, rate)
    return st


class LoadTest(unittest.TestCase):
    def test_idle(self):
        v = service().load_view()
        self.assertEqual((v["load_pct"], v["level"]), (0, "idle"))
        self.assertEqual(v["slots"]["free"], 16)
        self.assertFalse(v["would_wait"])

    def test_mixed(self):
        v = service(decoding=10, reading=2).load_view()
        self.assertEqual(v["slots"], {"total": 16, "busy": 12, "decoding": 10, "reading_prompt": 2, "free": 4,
                                      "waiting": 0})
        self.assertEqual(v["load_pct"], 75)     # slots 12/16 beat the 200/660 tok/s share
        self.assertEqual(v["level"], "busy")

    def test_saturated_with_queue(self):
        # the 3 that wait are registered too (as _run does), so 16 + 3 requests are live
        v = service(decoding=16, reading=3, waiting=3).load_view()
        self.assertEqual((v["load_pct"], v["level"]), (100, "saturated"))
        self.assertEqual((v["slots"]["busy"], v["slots"]["reading_prompt"], v["slots"]["waiting"]), (16, 0, 3))
        self.assertIn("queueing", v["advice"])


class AdmissionStateTest(unittest.TestCase):
    """The states the batched server really has (the gate code of StrataEngine itself)."""

    def setUp(self):
        self.cancels = []

    def tearDown(self):
        for c in self.cancels:
            c.set()

    def take(self, e, plen, **kw):
        c = threading.Event()
        threading.Thread(target=lambda: run(e._take_control(c, plen, **kw)), daemon=True).start()
        self.cancels.append(c)

    def image(self, e, me, plen):
        c = threading.Event()
        threading.Thread(target=lambda: run(e._take_engine_alone(c, me, plen)), daemon=True).start()
        self.cancels.append(c)

    def until(self, cond):
        end = time.time() + 5
        while time.time() < end and not cond():
            time.sleep(0.01)
        self.assertTrue(cond())

    def test_an_image_request_alone_leaves_no_slot_and_the_ones_behind_it_wait(self):
        e = engine_without_process(16)
        s = on(e)
        a, b = object(), object()
        e.image_line += [a, b]
        self.assertTrue(e.ctl.acquire(timeout=1))           # `a` runs alone on the solo path to its DONE
        register(s, tok_s=103.0)
        register(s)                                         # b, behind it
        self.image(e, b, 499)
        for _ in range(2):                                  # two text requests: they stand back
            register(s)
            self.take(e, 2000, new=True)
        self.until(lambda: e.standing == 2)
        v = s.load_view()
        e.ctl.release()
        self.assertEqual(e.waiting, 0)                      # (not counted as waiting by the engine)
        self.assertTrue(v["image_alone"] and v["would_wait"])
        self.assertEqual((v["slots"]["free"], v["slots"]["waiting"], v["slots"]["busy"]), (0, 3, 1))
        self.assertEqual((v["load_pct"], v["level"]), (100, "saturated"))
        self.assertIn("image", v["advice"])

    def test_a_request_waiting_for_the_control_lines_is_counted_once(self):
        e = engine_without_process(16)
        s = on(e)
        self.assertTrue(e.ctl.acquire(timeout=1))           # a long read on the control lines
        register(s)
        for _ in range(3):
            register(s)
            self.take(e, 1500, new=True)
        self.until(lambda: e.waiting == 3)
        v = s.load_view()
        e.ctl.release()
        self.assertEqual(v["slots"], {"total": 16, "busy": 1, "decoding": 0, "reading_prompt": 1, "free": 15,
                                      "waiting": 3})
        self.assertTrue(v["would_wait"])
        self.assertIn("queueing", v["advice"])

    def test_requests_waiting_for_a_slot_are_waiting(self):
        e = engine_without_process(16)
        s = on(e)
        with e.slot_cv:
            e.slot_busy = [True] * 16
            e.admitting = 2
        for _ in range(16):
            register(s, tok_s=37.5)
        register(s)
        register(s)
        v = s.load_view()
        self.assertEqual((v["slots"]["busy"], v["slots"]["decoding"], v["slots"]["reading_prompt"],
                          v["slots"]["waiting"], v["slots"]["free"]), (16, 16, 0, 2, 0))
        self.assertIn("queueing", v["advice"])

    def test_an_image_request_draining_the_slots_admits_nothing_new(self):
        e = engine_without_process(16)
        s = on(e)
        me = object()
        with e.slot_cv:
            for b in range(4):
                e.slot_busy[b] = True
            e.image_line.append(me)
        for _ in range(4):
            register(s, tok_s=40.0)
        register(s)
        self.image(e, me, 220000)
        v = s.load_view()
        self.assertEqual((v["slots"]["busy"], v["slots"]["decoding"], v["slots"]["free"], v["slots"]["waiting"]),
                         (4, 4, 0, 1))
        self.assertEqual(v["level"], "saturated")


class PeakTest(unittest.TestCase):
    """The 100 % mark: the token count over LOAD_SUSTAIN_S with half the slots decoding, never one second's figure."""

    def drive(self, s, seconds, per_s, start=1000.0, burst_at=None):
        """One telemetry sample a second; tokens_out grows per_s a second (at burst_at: a consumer stalled for a
        second and read its backlog the next one)."""
        clock = [start]
        with mock.patch.object(server.time, "time", lambda: clock[0]):
            for k in range(seconds):
                if burst_at is None or k not in (burst_at, burst_at + 1):
                    s.tokens_out += per_s
                elif k == burst_at + 1:
                    s.tokens_out += 2 * per_s
                s._load_sample()
                clock[0] += 1.0
        return clock[0]

    def test_cold_start_one_request_is_not_the_100_mark(self):
        s = on(engine_without_process(16), peak=0.0)
        register(s, tok_s=103.0)
        self.drive(s, 30, 103)
        v = s.load_view()
        self.assertEqual(s.load_peak, 0.0)                  # one request is not the server's capacity
        self.assertIsNone(v["tok_s"]["peak"])
        self.assertEqual((v["load_pct"], v["level"]), (6, "idle"))

    def test_a_burst_does_not_raise_the_peak(self):
        s = on(engine_without_process(16), peak=0.0)
        for _ in range(16):
            register(s, tok_s=37.5)
        self.drive(s, 40, 600, burst_at=20)
        self.assertGreaterEqual(s.load_peak, 590)
        self.assertLessEqual(s.load_peak, 600 * 1.11)       # (a 1 s backlog in a 10 s window: at most +10 %)

    def test_a_rise_inside_the_save_interval_is_written_later(self):
        with tempfile.TemporaryDirectory() as d:
            s = on(engine_without_process(16), peak=0.0)
            s.load_peak_path = str(Path(d) / "p.json")
            s._load_peak_saved = (0.0, 1000.0)              # written at start (main() does this)
            for _ in range(16):
                register(s, tok_s=37.5)
            t = self.drive(s, 15, 600)                      # the peak rises 10-15 s after the last write
            self.assertFalse(Path(s.load_peak_path).exists())
            self.drive(s, 60, 300, start=t)                 # never as high again
            self.assertEqual(json.loads(Path(s.load_peak_path).read_text())["peak_tok_s"], 600.0)
            self.assertEqual(sorted(p.name for p in Path(d).iterdir()), ["p.json"])   # no .tmp left

    def test_peak_file_validation(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "p.json"
            for text, want in (('{"peak_tok_s": 660.0, "at": 0}', 660.0),             # written by hand: taken
                               ('{"peak_tok_s": 700, "config": "k"}', 700.0),
                               ('{"peak_tok_s": 700, "config": "other"}', 0.0),       # another config's
                               ('{"peak_tok_s": NaN}', 0.0), ('{"peak_tok_s": Infinity}', 0.0),
                               ('{"peak_tok_s": -5}', 0.0), ('{"peak_tok_s": "x"}', 0.0), ('[1]', 0.0),
                               ('{"peak_tok_s": 6', 0.0), ('', 0.0)):
                p.write_text(text)
                self.assertEqual(read_load_peak(p, "k"), want, text)
            self.assertEqual(read_load_peak(Path(d) / "missing.json", "k"), 0.0)

    def test_ref_env_must_be_a_finite_number(self):
        s = service(decoding=16)
        for value in ("nan", "inf", "-1", "x"):
            with mock.patch.dict("os.environ", {"STRATA_LOAD_REF_TOK_S": value}):
                self.assertEqual(s.load_view()["tok_s"]["peak"], 660.0, value)
        with mock.patch.dict("os.environ", {"STRATA_LOAD_REF_TOK_S": "320"}):
            self.assertEqual(s.load_view()["tok_s"]["peak"], 320.0)


class RouteTest(unittest.TestCase):
    """GET /v1/load and /load beside POST /load and /v1/load (which load the model)."""

    def setUp(self):
        from serve.frontend import ChatTemplate
        from serve.server import ByteTokenizer, MockEngine, Server, api_key_of, host_names_for, make_handler
        tok = ByteTokenizer()
        self.svc = Service(MockEngine(tok, "Hello."), tok, ChatTemplate(Path(server.__file__).parent / "chat_template.jinja"))
        self.svc.api_key = api_key_of("secret")
        self.svc.host_names = host_names_for("127.0.0.1", self.svc.allowed_hosts, self.svc.trusted_origins)
        self.httpd = Server(("127.0.0.1", 0), make_handler(self.svc))     # (no telemetry: serve() would start it)
        self.httpd.svc = self.svc
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def tearDown(self):
        self.httpd.shutdown()
        self.httpd.server_close()

    def call(self, method, path, key=None, body=None):
        headers = {"Content-Type": "application/json", **({"Authorization": f"Bearer {key}"} if key else {})}
        req = urllib.request.Request(self.base + path, method=method, headers=headers,
                                     data=json.dumps(body).encode() if body is not None else None)
        try:
            r = urllib.request.urlopen(req, timeout=10)
        except urllib.error.HTTPError as err:
            r = err
        with r:
            return r.status, json.loads(r.read() or b"null")

    def test_get_load_needs_the_key_and_post_still_loads_the_model(self):
        self.assertEqual(self.call("GET", "/v1/load")[0], 401)
        code, v = self.call("GET", "/v1/load", "secret")
        self.assertEqual(code, 200)
        self.assertIn("load_pct", v)
        code, v = self.call("GET", "/load/", "secret")      # vLLM's path: its server_load figure is there too
        self.assertEqual((code, v["server_load"]), (200, 0))
        code, v = self.call("POST", "/v1/load", "secret", {})
        self.assertEqual((code, v["status"]), (200, "loaded"))
        self.assertNotIn("load_pct", v)


class AgeOutTest(unittest.TestCase):
    def messages(self):
        img = {"type": "image", "source": "data:image/png;base64,AAAA"}
        return [{"role": "user", "content": [{"type": "text", "text": "look"}, dict(img)]},
                {"role": "assistant", "content": "ok"},
                {"role": "tool", "content": [dict(img)]},
                {"role": "assistant", "content": "seen"},
                {"role": "tool", "content": [dict(img)]}]

    def test_old_tool_images_become_notes_user_and_current_stay(self):
        from serve.frontend import images_of
        s = object.__new__(Service)
        m = self.messages()
        s._age_out_tool_images(m)
        self.assertEqual(len(images_of(m)), 2)              # the user's picture and the current tool picture
        self.assertEqual(m[2]["content"][0]["type"], "text")

    def test_keep(self):
        import os
        from serve.frontend import images_of
        s = object.__new__(Service)
        m = self.messages()
        os.environ["STRATA_IMAGE_HISTORY"] = "keep"
        try:
            s._age_out_tool_images(m)
        finally:
            del os.environ["STRATA_IMAGE_HISTORY"]
        self.assertEqual(len(images_of(m)), 3)


if __name__ == "__main__":
    unittest.main()
