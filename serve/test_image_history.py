"""serve/test_image_history.py - pictures of answered turns become a note (Service._age_out_tool_images), on every
request path, and the image request's log line.

    python -m unittest serve.test_image_history -v
"""
from __future__ import annotations

import contextlib
import io
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve.frontend import ChatTemplate, anthropic_to_messages, openai_to_messages  # noqa: E402
from serve.responses import input_messages  # noqa: E402
from serve.server import ByteTokenizer, MockEngine, Service  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PNG = "data:image/png;base64,AAAA"
OA_IMG = {"type": "image_url", "image_url": {"url": PNG}}
AN_IMG = {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "AAAA"}}
CALL = {"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "type": "function",
                                                            "function": {"name": "read_file", "arguments": "{}"}}]}
USE = {"role": "assistant", "content": [{"type": "tool_use", "id": "t1", "name": "Read", "input": {}}]}


def bare(vision=True):
    s = object.__new__(Service)
    s.vision = object() if vision else None
    return s


def where(msgs):
    return [f"{i}:{m['role']}" for i, m in enumerate(msgs) if isinstance(m.get("content"), list)
            and any(isinstance(x, dict) and x.get("type") == "image" for x in m["content"])]


def aged(msgs, env=None, vision=True):
    with mock.patch.dict(os.environ, {"STRATA_IMAGE_HISTORY": env} if env is not None else {}, clear=False):
        if env is None:
            os.environ.pop("STRATA_IMAGE_HISTORY", None)
        n = bare(vision)._age_out_tool_images(msgs)
    return n, where(msgs)


class PathsTest(unittest.TestCase):
    def test_tool_pictures_of_answered_turns_on_every_api(self):
        cases = {
            "openai": openai_to_messages({"messages": [
                {"role": "user", "content": "go"}, CALL, {"role": "tool", "tool_call_id": "c1", "content": [OA_IMG]},
                CALL, {"role": "tool", "tool_call_id": "c1", "content": "text"}]})[0],
            "anthropic": anthropic_to_messages({"messages": [
                {"role": "user", "content": "go"}, USE,
                {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": [AN_IMG]}]}, USE,
                {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "ok"}]}]})[0],
            "responses": input_messages({"input": [
                {"role": "user", "content": "go"},
                {"type": "function_call", "call_id": "a", "name": "shot", "arguments": "{}"},
                {"type": "function_call_output", "call_id": "a", "output": [{"type": "input_image", "image_url": PNG}]},
                {"type": "function_call", "call_id": "b", "name": "ls", "arguments": "{}"},
                {"type": "function_call_output", "call_id": "b", "output": "ok"}]}),
        }
        for name, msgs in cases.items():
            with self.subTest(api=name):
                self.assertEqual(aged(copy_of(msgs))[0], 0)            # off by default: the last release's prompt
                self.assertEqual(aged(copy_of(msgs), "keep")[0], 0)
                self.assertEqual(aged(msgs, "tools"), (1, []))

    def test_the_turn_being_answered_keeps_its_pictures_also_with_a_prefill(self):
        msgs = anthropic_to_messages({"messages": [
            {"role": "user", "content": "go"}, USE,
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": [AN_IMG]}]},
            {"role": "assistant", "content": "The screenshot shows"}]})[0]
        self.assertEqual(aged(msgs, "tools"), (0, ["2:tool"]))

    def test_user_pictures_stay_by_default(self):
        msgs = openai_to_messages({"messages": [
            {"role": "system", "content": "s"}, {"role": "user", "content": "env"},
            {"role": "assistant", "content": "Got it."},
            {"role": "user", "content": [{"type": "text", "text": "build this"}, OA_IMG, OA_IMG, OA_IMG]},
            CALL, {"role": "tool", "tool_call_id": "c1", "content": "r"}, CALL,
            {"role": "tool", "tool_call_id": "c1", "content": "r"}]})[0]
        self.assertEqual(aged(msgs), (0, ["3:user"]))
        self.assertEqual(aged(msgs, "keep"), (0, ["3:user"]))
        self.assertEqual(aged(msgs, "tools"), (0, ["3:user"]))         # a user's pictures are not a tool's

    def test_opt_in_user_pictures_after_n_answers_and_moved_tool_pictures(self):
        def chat(rounds):
            m = [{"role": "system", "content": "s"},
                 {"role": "user", "content": [{"type": "text", "text": "build this"}, OA_IMG]}]
            for _ in range(rounds):
                m += [CALL, {"role": "tool", "tool_call_id": "c1", "content": "r"}]
            return openai_to_messages({"messages": m})[0]
        self.assertEqual(aged(chat(2), "3"), (0, ["1:user"]))       # answered twice: stays
        self.assertEqual(aged(chat(3), "3"), (1, []))               # three times: a note from now on
        moved = openai_to_messages({"messages": [
            {"role": "user", "content": "go"}, CALL, {"role": "tool", "tool_call_id": "c1", "content": "see below"},
            {"role": "user", "content": [OA_IMG]}, CALL, {"role": "tool", "tool_call_id": "c1", "content": "r"}]})[0]
        self.assertEqual(aged(copy_of(moved), "tools"), (0, ["3:user"]))   # "tools": a user message's picture
        self.assertEqual(aged(moved, "50"), (1, []))                # opt-in: the tool's, answered once

    def test_without_an_encoder_nothing_is_aged(self):
        msgs = openai_to_messages({"messages": [
            {"role": "user", "content": "go"}, CALL, {"role": "tool", "tool_call_id": "c1", "content": [OA_IMG]},
            CALL, {"role": "tool", "tool_call_id": "c1", "content": "r"}]})[0]
        self.assertEqual(aged(msgs, "tools", vision=False), (0, ["2:tool"]))


def copy_of(msgs):
    import copy
    return copy.deepcopy(msgs)


class FakeVision:
    def __init__(self, d):
        self.dir = Path(d)
        self.rows = self.dir / "img.sve"
        self.rows.write_bytes(b"rows")

    def encode(self, source):
        return self.rows, 64


class PrepareTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.tok = ByteTokenizer()
        self.svc = Service(MockEngine(self.tok, "ok", max_context=1 << 20), self.tok,
                           ChatTemplate(ROOT / "serve/chat_template.jinja"), vision=FakeVision(self.tmp.name))

    def tearDown(self):
        if self.svc.embeddings.path:
            Path(self.svc.embeddings.path).unlink(missing_ok=True)
        self.tmp.cleanup()

    def prepare(self, raw):
        out = io.StringIO()
        with contextlib.redirect_stdout(out), mock.patch.dict(os.environ, {}, clear=False):
            os.environ["STRATA_IMAGE_HISTORY"] = "tools"
            ids, _, _ = self.svc.prepare(openai_to_messages({"messages": raw})[0], None, {})
        return ids, out.getvalue()

    def test_the_prompt_changes_once_where_the_picture_was_then_stays(self):
        base = [{"role": "system", "content": "S" * 500}, {"role": "user", "content": "go"}, CALL,
                {"role": "tool", "tool_call_id": "c1", "content": [OA_IMG]}]
        turns = [base, base + [CALL, {"role": "tool", "tool_call_id": "c1", "content": "one"}],
                 base + [CALL, {"role": "tool", "tool_call_id": "c1", "content": "one"},
                         CALL, {"role": "tool", "tool_call_id": "c1", "content": "two"}]]
        (a, log_a), (b, log_b), (c, _) = (self.prepare(t) for t in turns)
        pad = self.tok.encode("<|image_pad|>", parse_special=True)[0]
        self.assertEqual((a.count(pad), b.count(pad), c.count(pad)), (64, 0, 0))
        self.assertEqual(c[:len(b) - 5], b[:len(b) - 5])          # turn 3 continues turn 2: no second change
        first = next(i for i, (x, y) in enumerate(zip(a, b)) if x != y)
        self.assertIn("[image omitted", self.tok.decode(b[first:first + 40]))
        self.assertIn(f"{len(a):,} tokens with them", log_a)       # the count with the picture's rows
        self.assertIn("1 older picture(s) left out", log_b)
        self.assertNotIn("S" * 20, log_a + log_b)                    # indices and roles only, no content


if __name__ == "__main__":
    unittest.main()
