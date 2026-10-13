"""serve/test_load.py - GET /v1/load: the load figure clients use to hold back work that is not urgent.

    python -m unittest serve.test_load -v
"""
from __future__ import annotations

import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve.server import Service  # noqa: E402


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


class LoadTest(unittest.TestCase):
    def test_idle(self):
        v = service().load_view()
        self.assertEqual((v["load_pct"], v["level"]), (0, "idle"))
        self.assertEqual(v["slots"]["free"], 16)

    def test_mixed(self):
        v = service(decoding=10, reading=2).load_view()
        self.assertEqual(v["slots"], {"total": 16, "busy": 12, "decoding": 10, "reading_prompt": 2, "free": 4,
                                      "waiting": 0})
        self.assertEqual(v["load_pct"], 75)     # slots 12/16 beat the 200/660 tok/s share
        self.assertEqual(v["level"], "busy")

    def test_saturated_with_queue(self):
        v = service(decoding=16, waiting=3).load_view()
        self.assertEqual((v["load_pct"], v["level"]), (100, "saturated"))
        self.assertIn("queueing", v["advice"])

    def test_peak_rises_and_is_kept(self):
        with tempfile.TemporaryDirectory() as d:
            s = service(decoding=4, peak=10.0)
            s.load_peak_path = str(Path(d) / "p.json")
            s.LOAD_PEAK_SAVE_S = 0
            s._load_sample()
            self.assertGreater(s.load_peak, 10.0)
            self.assertTrue(Path(s.load_peak_path).exists())




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
