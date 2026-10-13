"""--batch ("parallel"): a request with an image has the engine alone - unless the engine keeps one M-RoPE table per slot.

An engine without per-slot tables keeps one M-RoPE position table (an image's positions) per GPU, rebuilt at every
admission, while each batch row reads it at its own slot's position: an image decoding next to other slots gave them
its positions and took theirs (silently wrong text).  With such an engine the server runs an image request on the solo
path (GENI) with no other slot active for its whole run: from its arrival no new request is admitted, the slots that
decode finish, then it runs; image requests queue one after another; a cancelled or failed one gives the turn back.
An engine that says INFO mrope_slots=1 keeps a table region per slot: its image requests go through the batch like any
other (BGENI beside the text slots).  Also: a dead image encoder fails only requests with images.

The admission logic is tested on an engine object without a process; the whole path over HTTP with the fake engine
of test_parallel (taught GENI, and to log the lines it receives).

    python -m unittest serve.test_image_alone -v
"""
import json
import sys
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from unittest import mock

import serve.server as server
from serve import test_parallel
from serve.server import EngineDied, StrataEngine

# the fake engine of test_parallel, which also reads GENI / BGENI and logs every line it receives (RECV <command>),
# and the end of a GENI (END GENI, written before its DONE: a line the server sent before that came during it)
FAKE = test_parallel.FAKE_BATCH
for old, new in (
        ('lines, stop = queue.Queue(), threading.Event()\n',
         'lines, stop = queue.Queue(), threading.Event()\n'
         'evf = open(args[args.index("--events") + 1], "a") if "--events" in args else None\n'
         'evl = threading.Lock()\n'
         'def note(s):\n'
         '    if evf:\n'
         '        with evl:\n'
         '            evf.write(s + "\\n"); evf.flush()\n'),
        ('        l = l.strip()\n', '        l = l.strip()\n        note("RECV " + l.split(" ", 1)[0])\n'),
        # --mrope-slots: the engine keeps one M-RoPE table region per slot and says so
        ('(" slot_cache=1" if "--slotcache" in args else "")',
         '(" slot_cache=1" if "--slotcache" in args else "") + (" mrope_slots=1" if "--mrope-slots" in args else "")'),
        ('if line.startswith(("GEN ", "BGEN ")):', 'if line.startswith(("GEN ", "BGEN ", "GENI ", "BGENI ")):'),
        ('slot = int(f[1]) if f[0] == "BGEN" else None', 'slot = int(f[1]) if f[0] in ("BGEN", "BGENI") else None'),
        ('max_new = int(f[2] if f[0] == "BGEN" else f[1])', 'max_new = int(f[2] if f[0] in ("BGEN", "BGENI") else f[1])'),
        ('        fin = "stop" if out and toks[out - 1] == 257 else ("cancel" if stop.is_set() else "length")\n',
         '        fin = "stop" if out and toks[out - 1] == 257 else ("cancel" if stop.is_set() else "length")\n'
         '        if f[0] == "GENI":\n'
         '            note("END GENI")\n'),
        # VREFUSE in a GENI's prompt: the first one is refused as the engine refuses a picture while a slot decodes
        ('            continue\n        if log:\n',
         '            continue\n'
         '        if f[0] == "GENI" and b"VREFUSE" in bytes(int(x) & 255 for x in ids) and b"VREFUSE" not in fired:\n'
         '            fired.add(b"VREFUSE")\n'
         '            print("ERR vision: an image request is not taken while batch slot 0 decodes", flush=True)\n'
         '            continue\n'
         '        if log:\n')):
    assert FAKE.count(old) == 1, old
    FAKE = FAKE.replace(old, new)

LONG_REPLY = "ok, " + "la " * 15 + "done."


def run(gen):
    """Consumes a generator of the engine (heartbeats and tokens); -> its return value."""
    try:
        while True:
            next(gen)
    except StopIteration as s:
        return s.value


def engine_without_process(n=2):
    """A StrataEngine with the admission state of a started one and no process: alive() is `e.up`."""
    e = StrataEngine("missing-executable", [], lazy=True)
    e.batch = n
    e.slot_busy = [False] * n
    e.slot_claims, e.slot_releasing, e._orphan_seen = set(), set(), {}
    e.slot_cv = threading.Condition()
    e.waiting, e.admitting, e.wait_lens, e.ctl_epoch = 0, 0, [], 0
    e.ctl = threading.Lock()
    e.image_line = []
    e.gen = 1
    e.up = True
    e.alive = lambda: e.up
    return e


def until(cond, timeout=5.0):
    end = time.time() + timeout
    while time.time() < end:
        if cond():
            return True
        time.sleep(0.01)
    return cond()


class Admission(unittest.TestCase):
    """The gate itself: who may take the control lines while an image request waits for the engine alone."""

    def busy(self, e, b):
        with e.slot_cv:
            e.slot_busy[b] = True
            e.slot_claims.add(b)

    def free(self, e, b):
        with e.slot_cv:
            e.slot_busy[b] = False
            e.slot_claims.discard(b)
            e.slot_cv.notify_all()

    def taker(self, e, plen, got, res, **kw):
        def go():
            try:
                res.append(run(e._take_control(threading.Event(), plen, **kw)))
                got.set()
            except Exception as x:     # noqa: BLE001 - asserted by the test
                res.append(x)
        th = threading.Thread(target=go, daemon=True)
        th.start()
        return th

    def image(self, e, cancel, res):
        def go():
            try:
                for _ in e._generate_alone([1, 2, 3], 8, {}, cancel, "x.sve"):
                    pass
                res.append("ended")
            except Exception as x:     # noqa: BLE001
                res.append(x)
        th = threading.Thread(target=go, daemon=True)
        th.start()
        self.assertTrue(until(lambda: e.image_line))
        return th

    def test_a_read_that_gave_way_goes_on_and_a_new_request_stands_back(self):
        """A long read gave way (BYIELD) to a short request and waits in its slot for the control lines; an image
        request arrives.  The short one - not admitted yet - stands back and is not counted as waiting, so the read
        that gave way is not left waiting for it (it holds the slot the image waits for: that was a deadlock)."""
        e = engine_without_process()
        self.assertTrue(e.ctl.acquire(timeout=1))          # the long read on the control lines
        self.busy(e, 1)                                     # ... and the slot its part read will wait in
        short_got, short = threading.Event(), []
        self.taker(e, 5, short_got, short, new=True)
        self.assertTrue(until(lambda: e.waiting == 1))
        epoch = e.ctl_epoch
        me = object()
        with e.slot_cv:
            e.image_line.append(me)                         # an image request comes
        e.ctl.release()                                     # the long read gives way
        long_got, long = threading.Event(), []
        self.taker(e, 3000, long_got, long, after_epoch=epoch)
        self.assertTrue(long_got.wait(5), "the read that gave way never got the control lines back (deadlock)")
        self.assertFalse(short_got.is_set(), "a new request was admitted while an image request waited")
        self.assertEqual((e.waiting, e.wait_lens), (0, []))
        e.ctl.release()                                     # it read the rest and decoded; its slot is free
        self.free(e, 1)
        time.sleep(0.6)
        self.assertFalse(short_got.is_set())                # still the image's turn
        with e.slot_cv:
            e.image_line.remove(me)                         # the image request is done
            e.slot_cv.notify_all()
        self.assertTrue(short_got.wait(5))
        self.assertEqual(short, [True])
        e.ctl.release()
        self.assertEqual((e.waiting, e.wait_lens), (0, []))

    def test_the_image_waits_for_the_slots_without_the_control_lines(self):
        e = engine_without_process()
        self.busy(e, 0)
        me = object()
        e.image_line.append(me)
        got, res = threading.Event(), []

        def go():
            res.append(run(e._take_engine_alone(threading.Event(), me, 10)))
            got.set()
        threading.Thread(target=go, daemon=True).start()
        time.sleep(0.7)
        self.assertFalse(got.is_set())
        self.assertFalse(e.ctl.locked())                    # a continuing request can still take them
        self.assertEqual(e.waiting, 0)                      # nor is it counted (no solo request STOPped for it)
        self.free(e, 0)
        self.assertTrue(got.wait(5))
        self.assertEqual(res, [True])
        self.assertTrue(e.ctl.locked())                     # it has them, every slot free
        e.ctl.release()

    def test_images_take_turns_in_arrival_order(self):
        e = engine_without_process()
        a, b = object(), object()
        e.image_line += [a, b]
        got_b = threading.Event()
        threading.Thread(target=lambda: (run(e._take_engine_alone(threading.Event(), b, 10)), got_b.set()),
                         daemon=True).start()
        self.assertTrue(run(e._take_engine_alone(threading.Event(), a, 10)))
        time.sleep(0.6)
        self.assertFalse(got_b.is_set())
        e.ctl.release()
        time.sleep(0.6)
        self.assertFalse(got_b.is_set())                    # the lines are free, but `a` is still running
        with e.slot_cv:
            e.image_line.remove(a)
            e.slot_cv.notify_all()
        self.assertTrue(got_b.wait(5))
        e.ctl.release()

    def test_a_cancelled_image_request_gives_the_turn_back(self):
        e = engine_without_process()
        self.busy(e, 0)
        cancel, img = threading.Event(), []
        th = self.image(e, cancel, img)
        text_got, text = threading.Event(), []
        self.taker(e, 5, text_got, text, new=True)
        time.sleep(0.7)
        self.assertFalse(text_got.is_set())
        self.assertEqual(e.waiting, 0)
        cancel.set()
        th.join(5)
        self.assertEqual(img, ["ended"])
        self.assertEqual(e.image_line, [])
        self.assertTrue(text_got.wait(5))
        e.ctl.release()

    def test_engine_death_gives_the_turn_back(self):
        e = engine_without_process()
        self.busy(e, 0)
        img = []
        th = self.image(e, threading.Event(), img)
        text_got, text = threading.Event(), []
        self.taker(e, 5, text_got, text, new=True)
        time.sleep(0.3)
        e.up = False
        th.join(5)
        self.assertEqual(len(img), 1)
        self.assertIsInstance(img[0], EngineDied)
        self.assertEqual(e.image_line, [])
        self.assertTrue(until(lambda: text))                # the waiter ends too (a clean 503), never hangs
        self.assertIsInstance(text[0], EngineDied)

    def test_an_image_that_waits_too_long_gives_up_and_the_text_goes_on(self):
        e = engine_without_process()
        self.busy(e, 0)                                     # a slot that decodes on and on
        img = []
        with mock.patch.object(server, "IMAGE_WAIT_S", 0.8):
            th = self.image(e, threading.Event(), img)
            text_got, text = threading.Event(), []
            self.taker(e, 5, text_got, text, new=True)
            time.sleep(0.3)
            self.assertFalse(text_got.is_set())             # held behind the image
            th.join(5)
        self.assertEqual(len(img), 1)
        self.assertIsInstance(img[0], server.ImageWaitTimeout)
        self.assertIsInstance(img[0], server.GpuBusy)       # (a 503 the client can retry)
        self.assertEqual(e.image_line, [])
        self.assertTrue(text_got.wait(5))                   # the held text request goes on
        self.assertFalse(e.ctl.locked() and not text)
        e.ctl.release()

    def test_a_restart_under_way_keeps_it_waiting_then_it_goes_on(self):
        e = engine_without_process()
        self.busy(e, 0)
        me = object()
        e.image_line.append(me)
        got, res = threading.Event(), []

        def go():
            try:
                res.append(run(e._take_engine_alone(threading.Event(), me, 10)))
            except Exception as x:     # noqa: BLE001
                res.append(x)
            got.set()
        threading.Thread(target=go, daemon=True).start()
        e.starting, e.up = True, False
        time.sleep(0.7)
        self.assertFalse(got.is_set())
        with e.slot_cv:                                     # restart(): a new process with every slot free
            e.slot_busy = [False, False]
            e.slot_claims = set()
            e.up, e.starting = True, False
            e.slot_cv.notify_all()
        self.assertTrue(got.wait(5))
        self.assertEqual(res, [True])
        e.ctl.release()

    def test_a_leaked_busy_flag_does_not_hold_it(self):
        e = engine_without_process()
        e.ORPHAN_SLOT_S = 0.3
        e.slot_busy[1] = True                               # nobody owns it (#1603)
        me = object()
        e.image_line.append(me)
        self.assertTrue(run(e._take_engine_alone(threading.Event(), me, 10)))
        self.assertEqual(e.slot_busy, [False, False])
        e.ctl.release()

    def test_the_handshake_decides_whether_an_image_runs_alone(self):
        e = engine_without_process()
        for info, alone in (({}, True), (None, True), ({"mrope_slots": 0}, True), ({"slot_cache": 1}, True),
                            ({"mrope_slots": 1}, False), ({"mrope_slots": 1, "batch_slots": 4}, False)):
            e.info = info
            self.assertEqual(e.image_alone(), alone, info)

    def test_no_slot_goes_back_to_the_solo_path_while_an_image_waits(self):
        e = engine_without_process()
        e.info = {"slot_cache": 1}
        self.busy(e, 0)
        self.assertTrue(e._may_go_solo(100, 0, None))
        e.image_line.append(object())
        self.assertFalse(e._may_go_solo(100, 0, None))


class DeadVision:
    """An image encoder that died and cannot start again."""

    def __init__(self, d):
        self.dir = Path(d)
        self.rows = self.dir / "img.sve"
        self.rows.write_bytes(b"rows")
        self.starts = 0

    def alive(self):
        return False

    def restart(self):
        self.starts += 1
        raise RuntimeError("no GPU for it")

    def unload(self):
        pass

    def encode(self, source, keep=()):
        return self.rows, 3


class LiveVision(DeadVision):
    def alive(self):
        return True


class OverHttp(unittest.TestCase):
    """The real StrataEngine and Service, the fake engine behind them."""

    start_engine, tearDown = test_parallel.ParallelService.start, test_parallel.ParallelService.tearDown
    chat = test_parallel.ParallelService.chat

    def start(self, vision=LiveVision, slots=2, mrope=False):
        self.start_engine(slots, more=["--mrope-slots"] if mrope else [])
        script = Path(self.tmp.name) / "fake_strata.py"
        script.write_text(FAKE, encoding="utf-8")
        self.events = Path(self.tmp.name) / "events.log"
        # the fake was started with test_parallel's script: start this one (same arguments, plus --events)
        real = server.subprocess.Popen
        self.engine.unload()
        patch = mock.patch.object(server.subprocess, "Popen", lambda cmd, **kw: real(
            [sys.executable, str(script), *cmd[1:], "--events", str(self.events)], **kw))
        patch.start()
        self.addCleanup(patch.stop)
        self.engine.restart()
        self.vdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.vdir.cleanup)
        self.svc.vision = vision(self.vdir.name)

    def image_chat(self, text, max_tokens=64):
        body = {"messages": [{"role": "user", "content": [
                    {"type": "text", "text": text},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}]}],
                "max_tokens": max_tokens, "reasoning_effort": "none"}
        req = urllib.request.Request(self.base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.loads(r.read().decode())

    def go(self, fn, text, out, errors):
        def one():
            try:
                out[text] = fn(text)["choices"][0]["message"]["content"]
            except Exception as e:   # noqa: BLE001 - reported by the test
                errors.append((text, repr(e)))
        th = threading.Thread(target=one)
        th.start()
        return th

    def recv(self):
        return [x.split(" ", 1)[1] if x.startswith("RECV ") else x for x in self.events.read_text().splitlines()]

    def assert_alone(self, ev):
        """Every GENI is followed by its own end before the engine receives any other line (but its own STOP: the
        consumer leaves at the end-of-turn token, before the DONE)."""
        for i, x in enumerate(ev):
            if x == "GENI":
                during = ev[i + 1:ev.index("END GENI", i)]
                self.assertEqual([y for y in during if y != "STOP"], [], ev)

    def test_an_image_waits_for_the_slots_and_runs_alone(self):
        self.start(slots=3)                                 # a slot stays free: the late request could take it
        out, errors = {}, []
        threads = [self.go(self.chat, "first LONGREPLY", out, errors)]
        time.sleep(0.1)
        threads.append(self.go(self.chat, "second LONGREPLY", out, errors))
        self.assertTrue(until(lambda: sum(self.engine.slot_busy) == 2))   # both decode in the slots
        threads.append(self.go(self.image_chat, "the picture LONGREPLY", out, errors))
        self.assertTrue(until(lambda: self.engine.image_line))
        time.sleep(0.1)
        threads.append(self.go(self.chat, "late", out, errors))
        for th in threads:
            th.join(60)
        self.assertEqual(errors, [])
        self.assertEqual(out, {"first LONGREPLY": LONG_REPLY, "second LONGREPLY": LONG_REPLY,
                               "the picture LONGREPLY": LONG_REPLY, "late": "ok, done."})
        ev = self.recv()
        self.assertEqual(ev.count("GENI"), 1, ev)
        self.assert_alone(ev)
        # the late request was sent only after the image's end; nothing was admitted after the image came
        after = ev[ev.index("END GENI") + 1:]
        self.assertTrue([x for x in after if x in ("GEN", "BGEN")], ev)
        self.assertEqual([x for x in ev[:ev.index("GENI")] if x in ("GEN", "BGEN", "GENI", "BGENI")],
                         ["GEN", "BGEN", "BGEN"], ev)       # first solo, then both moved into the slots
        # the fake logged how many slots were active when the GENI came: none
        geni = [x.split() for x in self.log.read_text().splitlines() if x.startswith("GENI")]
        self.assertEqual([g[3] for g in geni], ["0"])
        e = self.engine
        self.assertEqual((e.image_line, e.waiting, e.wait_lens, any(e.slot_busy), e.ctl.locked()),
                         ([], 0, [], False, False))
        self.assertNotIn("mrope_slots", e.info)             # (an engine without per-slot tables)

    def test_with_slot_tables_an_image_shares_the_batch(self):
        """INFO mrope_slots=1: the image request is admitted into a slot (BGENI) while the text slots decode, and a text
        request that comes after it is admitted while the image slot decodes; nothing waits for the engine alone."""
        self.start(slots=4, mrope=True)
        self.assertEqual(self.engine.info.get("mrope_slots"), 1)
        out, errors = {}, []
        threads = [self.go(self.chat, "first LONGREPLY", out, errors)]
        time.sleep(0.1)
        threads.append(self.go(self.chat, "second LONGREPLY", out, errors))
        self.assertTrue(until(lambda: sum(self.engine.slot_busy) == 2))   # both decode in the slots
        threads.append(self.go(self.image_chat, "the picture LONGREPLY", out, errors))
        self.assertTrue(until(lambda: sum(self.engine.slot_busy) == 3))   # the image in a slot beside them
        self.assertEqual(self.engine.image_line, [])
        threads.append(self.go(self.chat, "late LONGREPLY", out, errors))
        for th in threads:
            th.join(60)
        self.assertEqual(errors, [])
        self.assertEqual(out, {"first LONGREPLY": LONG_REPLY, "second LONGREPLY": LONG_REPLY,
                               "the picture LONGREPLY": LONG_REPLY, "late LONGREPLY": LONG_REPLY})
        ev = self.recv()
        self.assertNotIn("GENI", ev)                        # never the solo path alone
        self.assertEqual(ev.count("BGENI"), 1, ev)
        # the fake logged how many slots were active at each admission: the image came beside both text slots, and
        # the late text request beside the image slot
        adm = [x.split() for x in self.log.read_text().splitlines() if x.startswith(("BGEN", "BGENI"))]
        kinds = [a[0] for a in adm]
        i = kinds.index("BGENI")
        self.assertGreaterEqual(int(adm[i][3]), 2, adm)
        self.assertTrue([a for a in adm[i + 1:] if a[0] == "BGEN" and int(a[3]) >= 1], adm)
        e = self.engine
        self.assertTrue(until(lambda: not any(e.slot_busy)))
        self.assertEqual((e.image_line, e.waiting, e.wait_lens, e.ctl.locked()), ([], 0, [], False))

    def test_with_slot_tables_an_image_alone_takes_the_solo_path(self):
        """INFO mrope_slots=1 and nothing else running: the image runs on the solo path (GENI), as a text request
        would, and a text request that comes meanwhile does not wait for it to end."""
        self.start(slots=3, mrope=True)
        out, errors = {}, []
        threads = [self.go(self.image_chat, "the picture LONGREPLY", out, errors)]
        time.sleep(0.15)
        threads.append(self.go(self.chat, "text LONGREPLY", out, errors))
        for th in threads:
            th.join(60)
        self.assertEqual(errors, [])
        self.assertEqual(set(out.values()), {LONG_REPLY})
        ev = self.recv()
        self.assertEqual(ev[:1], ["GENI"], ev)
        # the text request arriving STOPped the image's solo run, which went on in a slot (BGENI with the reply so
        # far), and the text request was admitted beside it (one slot active at its BGEN)
        self.assertIn("BGENI", ev)
        adm = [x.split() for x in self.log.read_text().splitlines() if x.startswith("BGEN ")]
        self.assertTrue(adm and int(adm[0][3]) >= 1, adm)
        self.assertEqual(self.engine.image_line, [])

    def test_a_restarted_engine_says_its_slot_tables_again(self):
        """mrope_slots is the running engine's: a restart into an engine that does not say it (its tables did not fit)
        sends image requests alone again."""
        self.start(slots=2, mrope=True)
        self.assertEqual(self.engine.image_alone(), False)
        exe, args, *rest = self.engine.spawn
        self.engine.spawn = (exe, [a for a in args if a != "--mrope-slots"], *rest)
        self.engine.restart()
        self.assertNotIn("mrope_slots", self.engine.info)
        self.assertEqual(self.engine.image_alone(), True)

    def test_images_queue_one_after_another(self):
        self.start()
        out, errors = {}, []
        threads = [self.go(self.image_chat, "one LONGREPLY", out, errors)]
        self.assertTrue(until(lambda: self.engine.image_line))
        threads.append(self.go(self.image_chat, "two LONGREPLY", out, errors))
        time.sleep(0.05)
        threads.append(self.go(self.chat, "text LONGREPLY", out, errors))
        for th in threads:
            th.join(60)
        self.assertEqual(errors, [])
        self.assertEqual(set(out.values()), {LONG_REPLY})
        self.assertEqual(len(out), 3)
        ev = self.recv()
        self.assertEqual(ev.count("GENI"), 2, ev)
        self.assertNotIn("BGENI", ev)
        self.assert_alone(ev)
        self.assertEqual(self.engine.image_line, [])

    def test_an_engine_refusal_waits_for_the_slots_again(self):
        """The engine refuses the picture (it still had a slot decoding that this server had already freed): the
        request waits for the slots again and is sent once more instead of failing."""
        self.start()
        out = self.image_chat("VREFUSE what is it?")
        self.assertEqual(out["choices"][0]["message"]["content"], "ok, done.")
        ev = self.recv()
        self.assertEqual(ev.count("GENI"), 2, ev)
        e = self.engine
        self.assertEqual((e.image_line, any(e.slot_busy), e.ctl.locked()), ([], False, False))
        self.assertEqual(self.chat("after")["choices"][0]["message"]["content"], "ok, done.")

    def test_a_dead_encoder_fails_only_image_requests(self):
        self.start(vision=DeadVision)
        self.assertEqual(self.chat("hi")["choices"][0]["message"]["content"], "ok, done.")
        with self.assertRaises(urllib.error.HTTPError) as cm:
            self.image_chat("what is it?")
        with cm.exception as e:
            self.assertEqual(e.code, 503)
            self.assertIn("the vision encoder failed to start", json.loads(e.read().decode())["error"]["message"])
        self.assertEqual(self.svc.vision.starts, 1)          # the text request did not try it
        # the model unloaded (idle): a text request loads it; the encoder is tried with it, its failure only said
        self.engine.unload()
        self.assertEqual(self.chat("again")["choices"][0]["message"]["content"], "ok, done.")
        self.assertEqual(self.svc.vision.starts, 2)
        self.assertTrue(self.engine.alive())


if __name__ == "__main__":
    unittest.main()
