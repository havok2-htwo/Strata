#!/usr/bin/env python3
"""STRATA_PIPE_OVERLAP: batch admissions read WHILE the pipelined slot groups decode (a layer split, --batch with
more than one slot group).  Every slot and every admission must equal its solo tokens, greedy.

  1. solo references (GEN) of every prompt: background prompts (~170 tokens), test prompts of ~170, ~1.2K and ~5K
     tokens (the 5K one with a system prompt past --prompt-cache-root: a root checkpoint, two batched segments), and
     the next turn of the first test prompt
  2. the background fills slots 0 .. bg-1 (each admission after the first is read beside decoding slots)
  3. admissions into the last two slots, one after the other, while the background decodes:
       short A; A's next turn into A's slot (slot reuse: copy_from_slot before the read); long B with a BSTOP of a
       background slot and a BYIELD sent during its read; huge C; huge D stopped (STOP) during its read (DONE
       cancel); short E; long F.  Before B: short G (few tokens) into the last slot, then long H into the one below
       it - G ends while H is read, and G's next turn (G2) must continue from G's slot (its RESUME past G's prompt:
       a group start padded up over a parked conversation would have wiped it)
  4. the background is stopped (BSTOP); every request's tokens must equal its solo tokens (a stopped one: a prefix)

  python3 tools/batch_overlap_test.py --exe engine/strata-g3-overlap --config <T16 config> \\
      --extra "--pcie-frac 0 --adapt-every 1000000 --no-prefill-borrow" --env STRATA_PIPE_OVERLAP=1 --dump on.json

--split 13,25,37 overrides the config's layer split; --env K=V (repeatable) adds engine env (the forced arms:
STRATA_PIPE_OVERLAP_FORCE_PADUP=1, STRATA_PIPE_OVERLAP_FORCE_WAIT=1|2); --compare A.json checks this run's tokens
against another run's dump (env 0 against 1); --solo-json reuses solo references from an earlier dump (same
prompts and lengths: the dump records them).  --expect-fault: an arm of STRATA_PIPE_OVERLAP_TEST_FAULT=poll|throw|
stage|stall|cuda (with stall: STRATA_WATCHDOG_S=10 STRATA_PIPE_OVERLAP_JOIN_S=2), passed when the engine answers exactly
one ERR and ends by itself (not by the watchdog, no std::terminate).  The engine log (BATCH_TEST_LOG) is checked too, with
STRATA_TRACE=1 set here: with the env on, an "overlapped read of" line per overlapped read, no "not captured
before an overlapped read", and no "captured the batch window" between a read's start and its stats line.
"""
import argparse, json, os, queue, sys, threading, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from batch_test import Engine, tokenizer  # noqa: E402

QUESTIONS = [
    "Explain how a B-tree keeps itself balanced.", "Compare TCP and QUIC in a few sentences.",
    "What causes the seasons on Earth?", "Describe the water cycle briefly.",
    "How does a hash map handle collisions?", "Summarize the plot of Hamlet.",
    "Why is the sky blue?", "What is a closure in programming?", "How do vaccines train the immune system?",
    "Explain the Monty Hall problem.", "What does a compiler's register allocator do?",
    "Describe how a refrigerator works.", "What is entropy in information theory?",
    "How does public-key cryptography work?", "Explain gradient descent.", "What is a Fourier transform?",
    "How do bees communicate?", "Explain the CAP theorem.", "What is photosynthesis?", "Describe a red-black tree.",
]
PARA = ("The committee reviewed item %d of the long agenda, noting the budget, the schedule and the open risks, and "
        "asked the staff to report back next quarter. ")
SYSPARA = ("Rule %d: answer precisely, cite the relevant section of the handbook, keep the tone neutral and never "
           "invent facts that the handbook does not state. ")


def filler(tok, pattern, n_tokens, start):
    per = max(1, len(tok.encode(pattern % 100)))
    return "".join(pattern % (start + i) for i in range(n_tokens // per + 1))


def chat(tok, q, fill=0, system=0, seed=0):
    text = ""
    if system:
        text += f"<|im_start|>system\n{filler(tok, SYSPARA, system, seed)}<|im_end|>\n"
    body = (filler(tok, PARA, fill, seed * 1000) + "\n\n" if fill else "") + q
    text += f"<|im_start|>user\n{body}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    return tok.encode(text, parse_special=True)


class EngineErr(Exception):
    pass


class Driver:
    def __init__(self, eng):
        self.eng = eng
        self.q = queue.Queue()
        self.slot_req = {}   # slot -> request name (decoding)
        self.toks = {}       # request name -> tokens
        self.fin = {}        # request name -> finish reason
        self.resume = {}     # request name -> the prompt tokens it reused (RESUME)
        self.events = []
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self):
        for line in self.eng.p.stdout:
            self.q.put(line.rstrip("\n"))
        self.q.put(None)

    def get(self, timeout=600):
        try:
            line = self.q.get(timeout=timeout)
        except queue.Empty:
            raise SystemExit(f"no engine output for {timeout} s (see {self.eng.log_path})")
        if line is None:
            raise SystemExit(f"the engine ended (rc {self.eng.p.wait()}; see {self.eng.log_path})")
        return line

    def handle(self, line):
        """BT/BDONE of decoding slots; ERR fails the test.  True if handled."""
        if line.startswith("BT "):
            _, s, y = line.split()
            self.toks[self.slot_req[int(s)]].append(int(y))
        elif line.startswith("BDONE "):
            f = line.split()
            name = self.slot_req.pop(int(f[1]))
            self.fin[name] = f[3]
            self.events.append(f"{name}: BDONE {f[3]} ({len(self.toks[name])} tokens)")
        elif line.startswith("ERR"):
            raise EngineErr(line)
        else:
            return False
        return True

    def solo(self, ids, max_new):
        self.eng.send(f"GEN {max_new} " + ",".join(map(str, ids)))
        got = []
        while True:
            line = self.get()
            if line.startswith("T "):
                got.append(int(line.split()[1]))
            elif line.startswith("DONE"):
                return got
            elif line.startswith("ERR"):
                raise SystemExit("engine: " + line)

    def admit(self, slot, name, ids, max_new, during=()):
        """BGEN into `slot`; `during`: (seconds after the BGEN, line) sent while its prompt is read."""
        self.toks[name] = []
        self.eng.send(f"BGEN {slot} {max_new} " + ",".join(map(str, ids)))
        t0, sched = time.time(), sorted(during)
        while True:
            while sched and time.time() - t0 >= sched[0][0]:
                self.eng.send(sched.pop(0)[1])
            try:
                line = self.q.get(timeout=0.005 if sched else 600)
            except queue.Empty:
                continue
            if line is None:
                raise SystemExit(f"the engine ended during {name} (see {self.eng.log_path})")
            if line.startswith("T "):
                self.toks[name].append(int(line.split()[1]))
            elif line.startswith("DONE"):
                self.fin[name] = line.split()[5]
            elif line.startswith("RESUME "):
                self.resume[name] = int(line.split()[1])
            elif line.startswith("ERR"):
                raise EngineErr(line)
            elif line.startswith("BADM "):
                for _, l in sched:   # whatever was not sent yet goes now (a read shorter than planned)
                    self.eng.send(l)
                if line.split()[2] == "1":
                    self.slot_req[slot] = name
                    self.fin.pop(name, None)
                self.events.append(f"{name}: admitted into slot {slot} ({len(ids)} tokens) in {time.time() - t0:.2f} s"
                                   f"{'' if line.split()[2] == '1' else ' - ended: ' + self.fin.get(name, '?')}")
                return
            else:
                self.handle(line)

    def wait_done(self, name):
        while name in self.slot_req.values():
            self.handle(self.get())

    def stop_all(self):
        for s in list(self.slot_req):
            self.eng.send(f"BSTOP {s}")
        while self.slot_req:
            self.handle(self.get())


def check_log(path, env_on):
    lines = Path(path).read_text(errors="replace").splitlines()
    stats = [l for l in lines if "strata batch: overlapped read of" in l]
    bad = [l for l in lines if "not captured before an overlapped read" in l]
    inside, open_read = [], False
    for l in lines:
        if "strata trace: overlapped read of tokens" in l:
            open_read = True
        elif "strata batch: overlapped read of" in l:
            open_read = False
        elif open_read and "captured the batch window" in l:
            inside.append(l)
    refused = sorted({l for l in lines if "overlapped read off:" in l})
    print(f"log: {len(stats)} overlapped reads, {len(bad)} 'not captured' errors, {len(inside)} captures inside a read, "
          f"{len(refused)} refusal reasons")
    for l in refused:
        print("   ", l)
    for l in stats[:3] + (["    ..."] if len(stats) > 6 else []) + stats[3:][-3:]:
        print("   ", l)
    for l in inside[:5]:
        print("   capture inside a read:", l)
    ok = not bad and not inside
    if env_on:
        ok &= len(stats) > 0
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exe", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--bg", type=int, default=14, help="background slots decoding through the test (<= batch-2)")
    ap.add_argument("--bg-new", type=int, default=600, help="background max_new (they are stopped at the end)")
    ap.add_argument("--max-new", type=int, default=24, help="max_new of the test admissions")
    ap.add_argument("--g-new", type=int, default=6, help="max_new of G (it must end while H is read)")
    ap.add_argument("--split", default="", help="layer split, e.g. 13,25,37 (default: the config's)")
    ap.add_argument("--extra", default="")
    ap.add_argument("--env", action="append", default=[], help="K=V engine env (repeatable)")
    ap.add_argument("--dump", default="")
    ap.add_argument("--compare", default="", help="a dump of another run: its tokens must equal this run's")
    ap.add_argument("--solo-json", default="", help="reuse the solo references of an earlier dump")
    ap.add_argument("--long", type=int, default=1200)
    ap.add_argument("--huge", type=int, default=5000)
    ap.add_argument("--expect-fault", action="store_true",
                    help="a STRATA_PIPE_OVERLAP_TEST_FAULT arm: pass if the engine answers exactly one ERR and ends "
                         "(not by the watchdog) within --fault-exit-s")
    ap.add_argument("--fault-exit-s", type=float, default=60)
    ap.add_argument("--stop-after", type=float, default=0.4, help="seconds after D's BGEN its STOP is sent")
    a = ap.parse_args()
    cfg = json.loads(Path(a.config).read_text())
    if a.split:
        cfg["layer_split"] = a.split
    tok = tokenizer(cfg["tokenizer"])
    assert a.bg <= a.batch - 2
    s14, s15 = a.batch - 2, a.batch - 1

    P = {}
    for i in range(a.bg):
        P[f"bg{i}"] = (chat(tok, QUESTIONS[i % len(QUESTIONS)], fill=150, seed=i), a.bg_new)
    P["A"] = (chat(tok, "Name three prime numbers and say why they are prime.", fill=150, seed=50), a.max_new)
    P["B"] = (chat(tok, "List the risks the committee noted.", fill=a.long, seed=51), a.max_new)
    P["C"] = (chat(tok, "Which rule matters most, and why?", fill=a.long, system=max(a.huge - a.long, 2100), seed=52),
              a.max_new)
    P["D"] = (chat(tok, "Summarize the agenda in one line.", fill=a.huge, seed=53), a.max_new)
    P["E"] = (chat(tok, "What is 17 times 23?", fill=150, seed=54), a.max_new)
    P["F"] = (chat(tok, "Which item came last?", fill=a.long, seed=55), a.max_new)
    P["G"] = (chat(tok, "Name a color.", fill=150, seed=56), a.g_new)
    P["H"] = (chat(tok, "How many items were reviewed?", fill=a.long, seed=57), a.max_new)

    env = {"STRATA_IQ_MT_MIN": "1", "STRATA_TRACE": "1", **(cfg.get("env") or {})}
    for kv in a.env:
        k, _, v = kv.partition("=")
        env[k] = v
    env_on = env.get("STRATA_PIPE_OVERLAP", "0") not in ("", "0")
    eng = Engine(a.exe, cfg, a.batch, env, a.extra.split())
    d = Driver(eng)
    print(f"engine: {a.exe} split {cfg.get('layer_split')} env {' '.join(f'{k}={v}' for k, v in sorted(env.items()))}",
          flush=True)

    solo = {}
    if a.solo_json:
        old = json.loads(Path(a.solo_json).read_text())
        if old.get("prompts") == {k: v[0] for k, v in P.items()} and old.get("max_new") == {k: v[1] for k, v in P.items()}:
            solo = old["solo"]
            print("solo references reused from", a.solo_json)
        else:
            print("solo references in", a.solo_json, "are for other prompts: computed again")
    if not solo:
        t0 = time.time()
        for k, (ids, mn) in P.items():
            solo[k] = d.solo(ids, mn)
        print(f"solo: {len(P)} prompts in {time.time() - t0:.0f} s", flush=True)
    # A's next turn: A, its solo reply, a new user turn - read from A's slot in the batch
    a2 = P["A"][0] + solo["A"] + tok.encode("<|im_end|>\n<|im_start|>user\nAnd three more?<|im_end|>\n"
                                            "<|im_start|>assistant\n<think>\n\n</think>\n\n", parse_special=True)
    P["A2"] = (a2, a.max_new)
    if "A2" not in solo:
        solo["A2"] = d.solo(a2, a.max_new)
    g2 = P["G"][0] + solo["G"] + tok.encode("<|im_end|>\n<|im_start|>user\nAnd another?<|im_end|>\n"
                                            "<|im_start|>assistant\n<think>\n\n</think>\n\n", parse_special=True)
    P["G2"] = (g2, a.max_new)
    if "G2" not in solo:
        solo["G2"] = d.solo(g2, a.max_new)
    print("prompt tokens:", {k: len(v[0]) for k, v in P.items()}, flush=True)

    try:
        return batch_phase(a, d, eng, P, solo, s14, s15, env_on)
    except EngineErr as x:
        if not a.expect_fault:
            raise SystemExit(f"engine: {x}")
        return fault_end(a, d, eng, str(x))


def fault_end(a, d, eng, first):
    t0, errs, rest = time.time(), [first], []
    while True:   # the rest of stdout until the engine ends
        try:
            line = d.q.get(timeout=max(1.0, a.fault_exit_s - (time.time() - t0)))
        except queue.Empty:
            break
        if line is None:
            break
        (errs if line.startswith("ERR") else rest).append(line)
    try:
        rc = eng.p.wait(timeout=max(1.0, a.fault_exit_s - (time.time() - t0)))
    except Exception:
        rc = None
    took = time.time() - t0
    if rc is None:
        eng.p.kill()
        eng.p.wait()
    eng.log.close()
    log = Path(eng.log_path).read_text(errors="replace")
    watchdog = "during a request (" in log and "issue #29" in log
    terminate = "terminate called" in log
    print(f"fault: {len(errs)} ERR line(s): {errs}")
    print(f"fault: the engine ended {'with rc ' + str(rc) if rc is not None else 'NOT (killed)'} {took:.1f} s after "
          f"the first ERR; watchdog fired: {watchdog}; std::terminate: {terminate}")
    for key in ("did not end in", "no prompt chunk finished", "strata batch: overlapped read:", "test fault",
                "strata serve: "):
        for l in log.splitlines():
            if key in l and ("strata serve: prompt" not in l):
                print("   log:", l[:200])
                break
    ok = len(errs) == 1 and rc is not None and not watchdog and not terminate
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 2


def batch_phase(a, d, eng, P, solo, s14, s15, env_on):
    t0 = time.time()
    for i in range(a.bg):
        d.admit(i, f"bg{i}", *P[f"bg{i}"])
    print(f"background: {a.bg} slots admitted in {time.time() - t0:.1f} s", flush=True)
    d.admit(s15, "A", *P["A"])
    d.wait_done("A")
    d.admit(s15, "A2", *P["A2"])   # slot 14 idle: A's slot is not a pad row meanwhile, so its state is reused
    d.wait_done("A2")
    # G ends while H is read: its slot is parked above its group's own width during that read, and G2 must
    # continue from it (a pad row there would leave G2 only the main session's copy of G's prompt)
    d.admit(s15, "G", *P["G"])
    d.admit(s14, "H", *P["H"])
    g_in_read = "G" in d.fin
    d.wait_done("G")
    d.wait_done("H")
    d.admit(s15, "G2", *P["G2"])
    d.wait_done("G2")
    d.admit(s14, "B", *P["B"], during=[(0.05, f"BSTOP {a.bg - 1}"), (0.10, f"BYIELD {s14}")])
    d.admit(s15, "C", *P["C"])
    d.wait_done("B")
    d.admit(s14, "D", *P["D"], during=[(a.stop_after, "STOP")])
    d.wait_done("C")
    d.admit(s15, "E", *P["E"])
    d.admit(s14, "F", *P["F"])
    d.wait_done("E")
    d.wait_done("F")
    active = len(d.slot_req)
    d.stop_all()
    print(f"admissions done in {time.time() - t0:.1f} s ({active} slots still decoding at the end)", flush=True)
    for e in d.events:
        print("   ", e)
    g2_resume = d.resume.get("G2", 0)
    g_slot = len(P["G"][0]) + len(solo["G"]) - 1   # G's prompt and every token a window fed
    print(f"G2: RESUME {g2_resume} (G's prompt {len(P['G'][0])}, its slot {g_slot}); G ended during H's read: "
          f"{g_in_read}")
    ok_resume = g2_resume >= g_slot
    if not g_in_read:
        print("note: G did not end while H was read (--g-new smaller?): the parked-slot case was not exercised")

    ok = ok_resume
    if not ok_resume:
        print("G2: DID NOT continue from G's slot")
    for k in P:
        b, s = d.toks.get(k, []), solo[k]
        fin = d.fin.get(k, "?")
        if k == "D":   # stopped while read: no tokens (or, if the read beat the STOP, a prefix)
            same = b == s[: len(b)]
            verdict = f"{'cancelled' if fin == 'cancel' else 'NOT cancelled (' + fin + ')'}, {len(b)} tokens, " \
                      f"{'a prefix of solo' if same else 'DIFFERS'}"
        elif fin == "cancel":   # stopped (BSTOP): a prefix
            same = len(b) > 0 and b == s[: len(b)]
            verdict = f"stopped at {len(b)} of {len(s)}: {'PREFIX OK' if same else 'DIFFERS'}"
        else:
            same = b == s
            first = next((i for i in range(min(len(b), len(s))) if b[i] != s[i]), None)
            verdict = f"{len(b)} tokens ({fin}), solo {len(s)}: " + ("IDENTICAL" if same else f"DIFFERS at {first}")
        ok &= same
        print(f"{k}: {verdict}")
    if "cancel" != d.fin.get(bg := f"bg{a.bg - 1}") and d.toks.get(bg) is not None:
        print(f"note: {bg} was not ended by the BSTOP sent during B's read ({d.fin.get(bg)})")
    eng.send("QUIT")
    try:
        eng.p.wait(timeout=180)
    except Exception:
        eng.p.kill()
    eng.log.close()
    ok &= check_log(eng.log_path, env_on)
    batch_toks = {k: d.toks.get(k, []) for k in P}
    if a.dump:
        Path(a.dump).write_text(json.dumps({"prompts": {k: v[0] for k, v in P.items() if k not in ("A2", "G2")},
                                            "max_new": {k: v[1] for k, v in P.items() if k not in ("A2", "G2")},
                                            "solo": solo, "batch": batch_toks, "resume": d.resume}))
    if a.compare:
        other = json.loads(Path(a.compare).read_text())["batch"]
        diff = [k for k in P if k in other and other[k] != batch_toks[k] and k[:2] != "bg"]
        # background slots are stopped at a time that depends on the run: compare their common prefix
        diff += [k for k in P if k[:2] == "bg" and k in other and
                 other[k][: min(len(other[k]), len(batch_toks[k]))] != batch_toks[k][: min(len(other[k]), len(batch_toks[k]))]]
        print(f"compare with {a.compare}: " + ("token-identical" if not diff else f"DIFFERS: {diff}"))
        ok &= not diff
        oth = json.loads(Path(a.compare).read_text()).get("resume", {})
        rdiff = [k for k in oth if k in d.resume and oth[k] != d.resume[k]]
        print(f"compare with {a.compare}: RESUME " + ("identical" if not rdiff else
              "DIFFERS: " + ", ".join(f"{k} {oth[k]} -> {d.resume[k]}" for k in rdiff)))
        ok &= not rdiff
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
