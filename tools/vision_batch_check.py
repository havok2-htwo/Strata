#!/usr/bin/env python3
"""--vision with --batch: image and text slots decoding side by side (the per-slot M-RoPE regions, INFO mrope_slots=1).

GPU only: it starts the engine (and the image encoder) on the config's cards.  Every comparison is of greedy TOKENS
(engine mode) or reply texts (server mode), never of snapshot bytes: a slot's indexer block_pos metadata holds its
region's base, so a parked slot's bytes differ from a main-session capture while every token is the same.

ENGINE MODE (the strong one): the engine binary and a strata-<model>.json with a "vision" block, spoken to over its
stdin/stdout protocol (GEN/GENI solo, BGEN/BGENI into a slot), as tools/batch_test.py does.
  process 1 (references): every request alone on the solo path (GEN / GENI), the image's turn 1 included
  process 2 (checks): first the follow-up turn of the image conversation alone in a fresh engine (nothing cached:
           its fresh solo reference), then
    (i)   text slots == their solo tokens while a long image prompt is read beside them (its chunks interleave with
          their windows) and while the image slot decodes (a text request admitted beside it)
    (ii)  the image request in that busy batch == the same image request solo
    (iii) two different images in two slots at once (and a text slot beside them), each == solo
    reset a text request admitted into a slot an image request used == solo (its region back to the identity)
    (iv)  the image conversation's follow-up turn after turn 1 decoded in a slot: into another slot beside a text
          slot (from the live session), then after unrelated requests (from the conversation cache / slot park /
          the SSD tier), then once more on the solo path - each == the fresh solo run of the same history
  --text-ab OLD_EXE (v): text only, byte-identical tokens between OLD_EXE and --exe, each with --vision on and off
          (4 engine runs; every run's solo and batch tokens must be the same in all four, and batch == solo)
  --negative-control: process 2 with STRATA_MROPE_REGION_SKIP=1 (the engine skips the region copy): the image checks
          must FAIL - proof that the comparisons see a missing copy.  Exit 0 only when they do.

  python3 tools/vision_batch_check.py --exe ../Strata/engine/strata-g3-143v1 \\
      --config ../linux/configs/T16-143-prod262r-vision-gpu.json --batch 16 --out vision_batch.json
  python3 tools/vision_batch_check.py --exe ../Strata/engine/strata-g3-143v1 --config <same> --batch 16 \\
      --text-ab ../Strata/engine/strata-g3-143k5
  python3 tools/vision_batch_check.py --exe ../Strata/engine/strata-g3-143v1 --config <same> --batch 16 \\
      --negative-control

Exact comparisons need deterministic decoding: by default STRATA_IQ_MT_MIN=1 and --extra "--pcie-frac 0
--adapt-every 1000000 --no-prefill-borrow" (as tools/batch_test.py and tools/batch_interleave_test.py; without
--no-prefill-borrow the slots decoding during a long read see the experts that read borrowed run on the CPU, which
rounds differently).  The config's own "env" is applied (--no-config-env leaves it out); STRATA_CONV_TIER_DIR points at
a fresh directory per engine run so a disk tier from an earlier run cannot answer a "fresh" reference.

SERVER MODE (--url http://127.0.0.1:8080/v1, a running server with "parallel" and "vision", nothing else using it): the
same scenarios over /chat/completions at temperature 0, comparing reply texts.  A text-only control at the same
concurrency runs first: if it already differs from solo (the production settings are not exact: adaptive tier, PCIe
share, prompt borrowing), the image results are reported as inconclusive rather than failed.
"""
import argparse, base64, io, json, os, queue, shutil, subprocess, sys, tempfile, threading, time, urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

IMAGE_PAD, VISION_START, VISION_END = 248056, 248053, 248054
EOS = (248044, 248046)
LINE_TIMEOUT_S = 900.0

TEXTS = [
    "Explain how a hash map handles collisions, with an example.",
    "Write a short Python function that checks whether a string is a palindrome, then explain it.",
    "What were the main causes of the First World War?",
    "Describe the water cycle step by step.",
    "Compare TCP and UDP in a few sentences.",
    "Give a recipe for a simple tomato soup.",
    "Explain what a B-tree is and why databases use it.",
    "Describe the life cycle of a star like the Sun.",
]
Q_IMAGE = "Describe this picture: which shapes, which colours, and where are they?"
Q_FOLLOW = "Now describe the same picture again, but list the colours first and then the shapes."
FILLER = ("The committee reviewed item %d of the long agenda, noting the budget, the schedule and the open risks, and "
          "asked the staff to report back next quarter. ")


# ---------------------------------------------------------------------------------------------------------------- pictures
def picture_png(kind, size=(640, 480)):
    """A synthetic picture (needs Pillow): two different ones give two different image layouts and embeddings."""
    from PIL import Image, ImageDraw
    im = Image.new("RGB", size, "white")
    d = ImageDraw.Draw(im)
    w, h = size
    if kind == "A":
        d.rectangle([w * .1, h * .15, w * .45, h * .6], fill=(220, 20, 20))
        d.ellipse([w * .55, h * .4, w * .9, h * .9], fill=(20, 50, 220))
    else:
        d.polygon([(w * .5, h * .05), (w * .9, h * .8), (w * .1, h * .8)], fill=(20, 160, 40))
        d.rectangle([w * .65, h * .1, w * .95, h * .3], fill=(240, 200, 0))
    buf = io.BytesIO()
    im.save(buf, "PNG")
    return buf.getvalue()


class Encoder:
    """The image encoder (the config's "vision" block, strata-vision): ENC <image> <out> -> OK <image tokens>."""

    def __init__(self, vcfg, workdir, env, log):
        args = [vcfg["exe"], "--mmproj", vcfg["mmproj"], "--model", vcfg["model"]]
        if vcfg.get("gpu"):
            args.append("--gpu")
        if vcfg.get("threads"):
            args += ["--threads", str(vcfg["threads"])]
        if vcfg.get("max_tokens"):
            args += ["--max-tokens", str(vcfg["max_tokens"])]
        if vcfg.get("min_tokens"):
            args += ["--min-tokens", str(vcfg["min_tokens"])]
        self.dir = Path(workdir)
        self.p = subprocess.Popen(args, cwd=self.dir, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log,
                                  text=True, bufsize=1, env=env)
        line = self.p.stdout.readline()
        if not line.startswith("READY"):
            raise SystemExit(f"the image encoder did not start: {line.strip()!r}")

    def encode(self, png, name):
        img, out = self.dir / f"{name}.png", self.dir / f"{name}.sve"
        img.write_bytes(png)
        self.p.stdin.write(f"ENC {img.name} {out.name}\n")
        self.p.stdin.flush()
        line = self.p.stdout.readline().strip()
        if not line.startswith("OK"):
            raise SystemExit(f"the image encoder failed: {line!r}")
        return out, int(line.split()[1])

    def close(self):
        try:
            self.p.stdin.write("QUIT\n")
            self.p.stdin.flush()
            self.p.wait(timeout=30)
        except Exception:  # noqa: BLE001
            self.p.kill()


# ---------------------------------------------------------------------------------------------------------------- prompts
class Prompts:
    def __init__(self, tok, pics, n_text, long_tokens):
        self.tok = tok
        self.pics = pics                     # name -> (embeddings file, image tokens)
        self.texts = [self.chat(q) for q in TEXTS[:n_text]]
        per = max(1, len(tok.encode(FILLER % 100)))
        self.filler = "".join(FILLER % i for i in range(long_tokens // per + 1)) if long_tokens else ""

    def chat(self, text):
        return self.tok.encode(f"<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
                               parse_special=True)

    def image(self, name, text, long=False):
        """The prompt of one picture and a question (one <|image_pad|> per image token, as the server writes it)."""
        body = (self.filler + "\n\n" if long else "") + text
        ids = self.tok.encode(f"<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>{body}<|im_end|>\n"
                              f"<|im_start|>assistant\n<think>\n\n</think>\n\n", parse_special=True)
        k = ids.index(IMAGE_PAD)
        if k == 0 or ids[k - 1] != VISION_START:
            raise SystemExit("the tokenizer did not give <|vision_start|><|image_pad|> as two special tokens")
        return ids[:k] + [IMAGE_PAD] * self.pics[name][1] + ids[k + 1:]

    def follow_up(self, turn1, reply, text):
        """turn 1's prompt, its reply (end-of-turn tokens dropped) and a new user turn, as one history."""
        r = list(reply)
        while r and r[-1] in EOS:
            r.pop()
        return list(turn1) + r + self.tok.encode(
            f"<|im_end|>\n<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
            parse_special=True)


# ---------------------------------------------------------------------------------------------------------------- engine
class Engine:
    """The engine's --serve protocol: one line per request, its answers on stdout (tools/batch_test.py's Engine, with
    the slots' tokens kept apart while admissions run)."""

    def __init__(self, exe, cfg, batch, vision, extra_args, env, log_path):
        args = [a for a in cfg["args"] if a != "--vision"]
        if vision:
            args.append("--vision")
        if len(cfg.get("gpu") or []) > 1:
            args += ["--layer-split", str(cfg.get("layer_split") or "auto")]
        if batch:
            args += ["--batch", str(batch)]
        args += extra_args
        self.log_path = log_path
        self.log = open(log_path, "w")
        self.p = subprocess.Popen([exe, "--serve", *args], cwd=cfg.get("cwd"), stdin=subprocess.PIPE,
                                  stdout=subprocess.PIPE, stderr=self.log, text=True, bufsize=1, env=env)
        self.q = queue.Queue()
        threading.Thread(target=self._read, daemon=True).start()
        self.info = {}
        while True:
            line = self.next()
            if line.startswith("INFO "):
                for kv in line.split()[1:]:
                    k, _, v = kv.partition("=")
                    self.info[k] = v
            if line.startswith("READY"):
                break
        self.slot_tok = {}                   # slot -> its request's tokens (the admission's T, then its BT lines)
        self.slot_done = {}                  # slot -> its BDONE line

    def _read(self):
        for line in self.p.stdout:
            self.q.put(line.rstrip("\n"))
        self.q.put(None)

    def next(self):
        try:
            line = self.q.get(timeout=LINE_TIMEOUT_S)
        except queue.Empty:
            raise SystemExit(f"the engine said nothing for {LINE_TIMEOUT_S:.0f} s (see {self.log_path})") from None
        if line is None:
            raise SystemExit(f"the engine ended (see {self.log_path})")
        return line

    def send(self, line):
        self.p.stdin.write(line + "\n")
        self.p.stdin.flush()

    def _slot_line(self, line):
        if line.startswith("BT "):
            _, s, y = line.split()
            self.slot_tok.setdefault(int(s), []).append(int(y))
            return True
        if line.startswith("BDONE "):
            self.slot_done[int(line.split()[1])] = line
            return True
        return False

    @staticmethod
    def _req(head, max_new, emb, ids):
        return " ".join(x for x in (head, str(max_new), str(emb) if emb else "", ",".join(map(str, ids))) if x)

    def solo(self, max_new, ids, emb=None):
        """GEN / GENI alone (no slot may be active): its tokens."""
        self.send(self._req("GENI" if emb else "GEN", max_new, emb, ids))
        got = []
        while True:
            line = self.next()
            if line.startswith("T "):
                got.append(int(line.split()[1]))
            elif line.startswith("DONE"):
                return got
            elif line.startswith("ERR"):
                raise SystemExit(f"engine: {line}")
            else:
                self._slot_line(line)

    def admit(self, slot, max_new, ids, emb=None):
        """BGEN / BGENI into `slot`; the other slots' lines meanwhile are kept.  -> True when it decodes on."""
        self.slot_done.pop(slot, None)
        self.slot_tok[slot] = []
        self.send(self._req(f"BGENI {slot}" if emb else f"BGEN {slot}", max_new, emb, ids))
        while True:
            line = self.next()
            if line.startswith("T "):
                self.slot_tok[slot].append(int(line.split()[1]))
            elif line.startswith("BADM "):
                if line.split()[2] != "1":
                    self.slot_done[slot] = line
                return line.split()[2] == "1"
            elif line.startswith("ERR"):
                raise SystemExit(f"engine: {line}")
            else:
                self._slot_line(line)

    def drain(self, slots):
        """Until every slot of `slots` has ended (BDONE)."""
        while any(s not in self.slot_done for s in slots):
            line = self.next()
            if line.startswith("ERR"):
                raise SystemExit(f"engine: {line}")
            self._slot_line(line)
        return {s: self.slot_tok.get(s, []) for s in slots}

    def quit(self):
        try:
            self.send("QUIT")
            self.p.wait(timeout=300)              # the next run needs the GPUs back
        except Exception:  # noqa: BLE001
            self.p.kill()
            self.p.wait(timeout=60)
        self.log.close()


def tokenizer(path):
    from batch_test import tokenizer as tk   # (tools/batch_test.py)
    return tk(path)


def slot_order(batch):
    """Slots spread over the pipelined groups (half and half), so both groups hold image and text rows."""
    if batch < 4:
        return list(range(batch))
    half = batch // 2
    out = []
    for a, b in zip(range(half), range(half, batch)):
        out += [a, b]
    return out + list(range(2 * half, batch))


class Results:
    def __init__(self):
        self.rows = []

    def check(self, name, got, want, kind=""):
        same = got == want
        diff = next((k for k in range(min(len(got), len(want))) if got[k] != want[k]),
                    None if len(got) == len(want) else min(len(got), len(want)))
        self.rows.append({"check": name, "kind": kind, "ok": same, "tokens": len(got), "ref_tokens": len(want),
                          "first_diff": diff})
        print(f"  {'OK  ' if same else 'FAIL'} {name}: {len(got)} tokens, reference {len(want)}"
              f"{'' if same else f', differs at {diff}'}", flush=True)
        return same

    def ok(self, kind=None):
        return all(r["ok"] for r in self.rows if kind is None or r["kind"] == kind)


# ---------------------------------------------------------------------------------------------------------------- engine mode
def engine_env(a, cfg, tier_root, extra=None):
    env = dict(os.environ)
    if not a.no_config_env:
        env.update({k: str(v) for k, v in (cfg.get("env") or {}).items()})
    env["LD_LIBRARY_PATH"] = ":".join(cfg.get("lib_dirs", []) + [env.get("LD_LIBRARY_PATH", "")])
    if a.mt_min:
        env["STRATA_IQ_MT_MIN"] = a.mt_min
    if "STRATA_CONV_TIER_DIR" in env:       # a fresh disk tier per engine run: no earlier run answers a reference
        d = Path(tempfile.mkdtemp(prefix="tier-", dir=tier_root))
        env["STRATA_CONV_TIER_DIR"] = str(d)
    env.pop("STRATA_MROPE_REGION_SKIP", None)
    env.update(extra or {})
    return env


def start(a, cfg, exe, vision, tier_root, name, extra_env=None):
    log = str(Path(a.logs) / f"{name}.log")
    print(f"[{name}] starting {exe} ({'--vision' if vision else 'no --vision'}, --batch {a.batch}); log {log}",
          flush=True)
    e = Engine(exe, cfg, a.batch, vision, a.extra.split(), engine_env(a, cfg, tier_root, extra_env), log)
    print(f"[{name}] ready: batch_slots={e.info.get('batch_slots')} batch_groups={e.info.get('batch_groups')} "
          f"mrope_slots={e.info.get('mrope_slots', '-')}", flush=True)
    return e


def run_engine_mode(a, cfg, tok, res):
    work = Path(tempfile.mkdtemp(prefix="vision-batch-check-"))
    venv = dict(os.environ)
    venv["LD_LIBRARY_PATH"] = ":".join(cfg.get("lib_dirs", []) + [venv.get("LD_LIBRARY_PATH", "")])
    enc = Encoder(cfg["vision"], work, venv, open(Path(a.logs) / "encoder.log", "w"))
    pics = {k: enc.encode(picture_png(k), f"pic{k}") for k in ("A", "B")}
    enc.close()                             # (its VRAM back before the engine sizes its expert cache)
    print(f"pictures: A {pics['A'][1]} image tokens, B {pics['B'][1]}", flush=True)
    P = Prompts(tok, pics, a.texts, a.long_tokens)
    img_a, img_b = P.image("A", Q_IMAGE), P.image("B", Q_IMAGE)
    img_long = P.image("A", Q_IMAGE, long=True)
    M = a.max_new
    emb = {k: str(v[0]) for k, v in pics.items()}

    # ---- process 1: the solo references
    e = start(a, cfg, a.exe, True, work, "refs")
    ref = {"text": [e.solo(M, ids) for ids in P.texts], "A": e.solo(M, img_a, emb["A"]),
           "B": e.solo(M, img_b, emb["B"]), "long": e.solo(M, img_long, emb["A"])}
    follow = P.follow_up(img_a, ref["A"], Q_FOLLOW)
    e.quit()
    print(f"references: {len(P.texts)} texts, images A/B/long; the follow-up turn is {len(follow)} tokens", flush=True)

    # ---- process 2: the checks
    skip = {"STRATA_MROPE_REGION_SKIP": "1"} if a.negative_control else None
    e = start(a, cfg, a.exe, True, work, "checks" + ("-negative" if skip else ""), skip)
    if e.info.get("mrope_slots") != "1":
        print("WARNING: the engine does not say mrope_slots=1: it has no per-slot M-RoPE regions (an older engine, or "
              "they did not fit) and refuses an image beside other slots", flush=True)
    ref["follow"] = e.solo(M, follow, emb["A"])                 # a fresh engine: nothing of it is cached
    order = slot_order(a.batch)
    nt = len(P.texts)
    if a.batch < 4 or nt < 3:
        raise SystemExit("needs --batch 4 or more and --texts 3 or more")

    # (i) + (ii): texts decode; a long image prompt is read beside them, then decodes beside a late text admission
    print("(i)/(ii): a long image prompt read and decoded beside text slots", flush=True)
    k = min(nt - 1, len(order) - 2)
    text_slots = order[:k]
    for s, ids in zip(text_slots, P.texts):
        e.admit(s, M, ids)
    img_slot, late_slot = order[k], order[k + 1]
    e.admit(img_slot, M, img_long, emb["A"])
    e.admit(late_slot, M, P.texts[k])
    got = e.drain(text_slots + [img_slot, late_slot])
    for j, s in enumerate(text_slots):
        res.check(f"(i) text {j} in slot {s} while the image is read and decodes", got[s], ref["text"][j], "text")
    res.check(f"(i) text {k} admitted in slot {late_slot} while the image slot decodes", got[late_slot],
              ref["text"][k], "text")
    res.check(f"(ii) the long image request in slot {img_slot}, in a busy batch", got[img_slot], ref["long"], "image")

    # (iii): two different images at once, a text slot beside them
    print("(iii): two different images in two slots at once", flush=True)
    sa, sb, st = order[0], order[1], order[2]
    e.admit(sa, M, img_a, emb["A"])
    e.admit(sb, M, img_b, emb["B"])
    e.admit(st, M, P.texts[0])
    got = e.drain([sa, sb, st])
    res.check(f"(iii) image A in slot {sa}", got[sa], ref["A"], "image")
    res.check(f"(iii) image B in slot {sb}", got[sb], ref["B"], "image")
    res.check(f"(iii) text 0 in slot {st} beside them", got[st], ref["text"][0], "text")

    # reset: text into the slots the images used (their regions back to the identity)
    print("reset: text requests into the slots the images used", flush=True)
    e.admit(sa, M, P.texts[1])
    e.admit(sb, M, P.texts[2])
    got = e.drain([sa, sb])
    res.check(f"reset: text 1 in slot {sa} (image A's before)", got[sa], ref["text"][1], "text")
    res.check(f"reset: text 2 in slot {sb} (image B's before)", got[sb], ref["text"][2], "text")

    # (iv): the follow-up turn of the image conversation
    print("(iv): the image conversation's follow-up turn", flush=True)
    s1, s2, s3, s4 = order[0], order[1], order[2], order[3]
    e.admit(s1, M, img_a, emb["A"])                             # turn 1 decodes in a slot
    e.admit(s2, M, P.texts[1])
    got = e.drain([s1, s2])
    res.check(f"(iv) turn 1 in slot {s1}", got[s1], ref["A"], "image")
    e.admit(s3, M, P.texts[2])                                  # a text slot decodes ...
    e.admit(s4, M, follow, emb["A"])                            # ... the follow-up beside it, in another slot
    got = e.drain([s3, s4])
    res.check(f"(iv) follow-up in slot {s4} beside a text slot, turn 1 having decoded in slot {s1}", got[s4],
              ref["follow"], "image")
    res.check(f"(iv) text 2 in slot {s3} beside it", got[s3], ref["text"][2], "text")
    for j in range(a.flush):                                    # unrelated requests: the follow-up's start leaves the
        e.solo(8, P.chat(f"Say a number between 1 and {100 + j}."))   # live session (cache / slot park / tier)
    e.admit(s1, M, P.texts[0])
    e.admit(s2, M, follow, emb["A"])
    got = e.drain([s1, s2])
    res.check(f"(iv) follow-up in slot {s2} after {a.flush} unrelated requests (the conversation cache)", got[s2],
              ref["follow"], "image")
    res.check(f"(iv) text 0 in slot {s1} beside it", got[s1], ref["text"][0], "text")
    res.check("(iv) follow-up on the solo path once more", e.solo(M, follow, emb["A"]), ref["follow"], "image")
    e.quit()
    shutil.rmtree(work, ignore_errors=True)


def text_run(a, cfg, tok, exe, vision, tier_root, name):
    """A text-only run: solo references, then the batch (a long prompt read beside the slots, a late admission)."""
    P = Prompts(tok, {}, a.texts, a.long_tokens)
    long_ids = P.chat(P.filler + "\n\nSummarize the agenda above in three sentences.")
    e = start(a, cfg, exe, vision, tier_root, name)
    M = a.max_new
    solo = [e.solo(M, ids) for ids in P.texts] + [e.solo(M, long_ids)]
    order = slot_order(a.batch)
    k = min(len(P.texts) - 1, len(order) - 2)
    for s, ids in zip(order[:k], P.texts):
        e.admit(s, M, ids)
    e.admit(order[k], M, long_ids)
    e.admit(order[k + 1], M, P.texts[k])
    got = e.drain(order[:k + 2])
    batch = [got[s] for s in order[:k]] + [got[order[k]], got[order[k + 1]]]
    batch_ref = solo[:k] + [solo[-1], solo[k]]
    e.quit()
    return {"solo": solo, "batch": batch, "batch_ref": batch_ref}


def run_text_ab(a, cfg, tok, res):
    tier_root = Path(tempfile.mkdtemp(prefix="vision-batch-ab-"))
    runs = {}
    for exe, tag in ((a.text_ab, "old"), (a.exe, "new")):
        for vision in (True, False):
            name = f"text-{tag}-{'vision' if vision else 'novision'}"
            runs[name] = text_run(a, cfg, tok, exe, vision, tier_root, name)
            Path(a.logs, f"{name}.tokens.json").write_text(json.dumps(runs[name]))
    base = "text-old-novision"
    for name, r in runs.items():
        for j, (b, s) in enumerate(zip(r["batch"], r["batch_ref"])):
            res.check(f"(v) {name}: batch row {j} == its solo", b, s, "text")
        if name != base:
            for j, (x, y) in enumerate(zip(r["solo"], runs[base]["solo"])):
                res.check(f"(v) {name} == {base}: solo {j}", x, y, "ab")
            for j, (x, y) in enumerate(zip(r["batch"], runs[base]["batch"])):
                res.check(f"(v) {name} == {base}: batch row {j}", x, y, "ab")
    shutil.rmtree(tier_root, ignore_errors=True)


# ---------------------------------------------------------------------------------------------------------------- server mode
def post(base, body, timeout=1800):
    req = urllib.request.Request(base.rstrip("/") + "/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json", "Authorization": "Bearer dummy"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        j = json.load(r)
    return j["choices"][0]["message"].get("content") or ""


def run_server_mode(a, res):
    def png_url(kind):
        return "data:image/png;base64," + base64.b64encode(picture_png(kind)).decode()

    def body(messages):
        return {"model": a.model, "messages": messages, "max_tokens": a.max_new, "temperature": 0,
                "chat_template_kwargs": {"enable_thinking": False}}

    def text_msg(q):
        return [{"role": "user", "content": q}]

    def image_msg(kind, q):
        return [{"role": "user", "content": [{"type": "text", "text": q},
                                             {"type": "image_url", "image_url": {"url": png_url(kind)}}]}]

    filler = "".join(FILLER % i for i in range(a.long_tokens // 30 + 1))
    texts = [text_msg(q) for q in TEXTS[:a.texts]]
    img_a, img_b = image_msg("A", Q_IMAGE), image_msg("B", Q_IMAGE)
    img_long = image_msg("A", filler + "\n\n" + Q_IMAGE)

    def together(named, stagger=0.3):
        out, err = {}, []

        def one(name, msgs):
            try:
                out[name] = post(a.url, body(msgs))
            except Exception as x:  # noqa: BLE001
                err.append((name, repr(x)))
        th = []
        for name, msgs in named:
            th.append(threading.Thread(target=one, args=(name, msgs)))
            th[-1].start()
            time.sleep(stagger)
        for t in th:
            t.join()
        if err:
            raise SystemExit(f"requests failed: {err}")
        return out

    def cmp(name, got, want, kind):
        g, w = list(got), list(want)            # (characters: a text comparison)
        return res.check(name, g, w, kind)

    print("references: every request alone", flush=True)
    ref = {f"t{j}": post(a.url, body(m)) for j, m in enumerate(texts)}
    ref.update({"A": post(a.url, body(img_a)), "B": post(a.url, body(img_b)), "long": post(a.url, body(img_long))})
    follow = img_a + [{"role": "assistant", "content": ref["A"]}, {"role": "user", "content": Q_FOLLOW}]
    ref["follow"] = post(a.url, body(follow))
    k = min(len(texts) - 1, 6)
    print("control: text only, at the same concurrency", flush=True)
    ctl = together([(f"t{j}", texts[j]) for j in range(k)] + [("x", text_msg(filler + "\n\nSummarize the agenda."))])
    control_ok = all(cmp(f"control: text {j}", ctl[f"t{j}"], ref[f"t{j}"], "control") for j in range(k))
    print("(i)/(ii): a long image prompt beside text requests", flush=True)
    got = together([(f"t{j}", texts[j]) for j in range(k)] + [("long", img_long), ("late", texts[k])])
    for j in range(k):
        cmp(f"(i) text {j} while the image is read and decodes", got[f"t{j}"], ref[f"t{j}"], "text")
    cmp(f"(i) text {k} admitted while the image decodes", got["late"], ref[f"t{k}"], "text")
    cmp("(ii) the long image request in a busy batch", got["long"], ref["long"], "image")
    print("(iii): two images at once", flush=True)
    got = together([("A", img_a), ("B", img_b), ("t0", texts[0])])
    cmp("(iii) image A", got["A"], ref["A"], "image")
    cmp("(iii) image B", got["B"], ref["B"], "image")
    cmp("(iii) text 0 beside them", got["t0"], ref["t0"], "text")
    print("(iv): the follow-up turn in a busy batch", flush=True)
    got = together([("t1", texts[1]), ("follow", follow), ("t2", texts[2])])
    cmp("(iv) follow-up beside text requests", got["follow"], ref["follow"], "image")
    cmp("(iv) text 1 beside it", got["t1"], ref["t1"], "text")
    if not control_ok:
        print("NOTE: the text-only control already differs from solo at this concurrency: the server's settings are "
              "not exact, so a difference below is INCONCLUSIVE, not a failure", flush=True)
    return control_ok


# ---------------------------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exe", help="the engine to check (engine mode)")
    ap.add_argument("--config", help="strata-<model>.json with a \"vision\" block (engine mode)")
    ap.add_argument("--url", help="server mode: a running server's /v1 base URL")
    ap.add_argument("--model", default="strata", help="server mode: the model name to send")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--texts", type=int, default=6, help="text prompts (at most %d)" % len(TEXTS))
    ap.add_argument("--max-new", type=int, default=96)
    ap.add_argument("--long-tokens", type=int, default=3000,
                    help="filler tokens in the long image prompt: several prompt chunks, read beside the text slots")
    ap.add_argument("--flush", type=int, default=6, help="(iv): unrelated solo requests before the follow-up again")
    ap.add_argument("--extra", default="--pcie-frac 0 --adapt-every 1000000 --no-prefill-borrow",
                    help="more engine arguments in one string")
    ap.add_argument("--mt-min", default="1", help="STRATA_IQ_MT_MIN for the engine (1: exact; empty: the default)")
    ap.add_argument("--no-config-env", action="store_true", help="leave the config's \"env\" out")
    ap.add_argument("--text-ab", metavar="OLD_EXE", help="(v) only: text tokens of OLD_EXE vs --exe, --vision on and off")
    ap.add_argument("--negative-control", action="store_true",
                    help="STRATA_MROPE_REGION_SKIP=1: the image checks must fail (exit 0 only if they do)")
    ap.add_argument("--logs", default="", help="directory for the engine logs and token dumps")
    ap.add_argument("--out", default="", help="write the results as JSON")
    a = ap.parse_args()
    a.texts = max(3, min(a.texts, len(TEXTS)))
    res = Results()
    if a.url:
        control_ok = run_server_mode(a, res)
        ok = res.ok("text") and res.ok("image")
        verdict = "PASS" if ok else ("INCONCLUSIVE" if not control_ok else "FAIL")
    else:
        if not (a.exe and a.config):
            ap.error("engine mode needs --exe and --config (or --url for server mode)")
        cfg = json.loads(Path(a.config).read_text())
        a.logs = a.logs or tempfile.mkdtemp(prefix="vision-batch-logs-")
        Path(a.logs).mkdir(parents=True, exist_ok=True)
        tok = tokenizer(cfg["tokenizer"])
        if a.text_ab:
            run_text_ab(a, cfg, tok, res)
            ok = res.ok()
        else:
            if not cfg.get("vision"):
                raise SystemExit("the config has no \"vision\" block (the image encoder)")
            run_engine_mode(a, cfg, tok, res)
            ok = res.ok()
            if a.negative_control:
                # the control passes when the image checks FAIL (and the text slots stay right)
                ok = not res.ok("image") and res.ok("text")
                print(f"negative control: image checks {'failed as they must' if not res.ok('image') else 'PASSED'}"
                      f" with the region copy skipped", flush=True)
        verdict = "PASS" if ok else "FAIL"
        print(f"engine logs: {a.logs}", flush=True)
    print(f"{verdict}: {sum(r['ok'] for r in res.rows)} of {len(res.rows)} checks identical", flush=True)
    if a.out:
        Path(a.out).write_text(json.dumps({"verdict": verdict, "checks": res.rows}, indent=1))
    return 0 if verdict == "PASS" else 2


if __name__ == "__main__":
    sys.exit(main())
