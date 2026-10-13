#!/usr/bin/env python3
"""A fake Strata engine and image encoder for tools/vision_check_smoke.py: CPU only, no model, no GPU.

It speaks the part of the engine's --serve protocol that tools/vision_batch_check.py uses, in the formats
src/program/generate.cpp prints - each format below names the line it was copied from (branch g3-143 at d6d896fb, the
STRATA_MROPE_SLOTS opt-in on top of bb01c8a8's per-slot regions) - so that protocol and bookkeeping bugs of the harness
show up before a GPU window:

  in   GEN / GENI <max_new> [key=value ...] [<embeddings file>] <id,id,...>    a request on the solo path (ckpt=0 known)
       BGEN / BGENI <slot> <max_new> ...   read as a "GEN 1" (11636-11651), then the request goes on in batch slot
              <slot>
       STOP   read on the stdin thread, never queued (9795): the running request tests it once per window, after the
              window's T lines and the next window's drafts (14440, 14455, 14505), and ends there (finish "cancel"; a
              STOP in the decode keeps the session live, 14538-14543); a STOP that came between two requests is
              dropped when the next one starts (11670)
       BSTOP <slot>: that slot ends at its next window (11606-11610; BDONE ... cancel, 11035), QUIT
  out  INFO ... (9877-9898), READY <context> stop (9954)                    once, at the start
       RESUME <n> (12628), PP <pos> <total> <ms> <tok/s> (9021, 12817), REUSED <n> (13568), T <id> (14440),
       DONE <generated> <prompt> <prompt ms> <decode ms> <finish> <drafts accepted> <drafts offered> <reused> ...
       (14675-14693), and after an admission's DONE: BADM <slot> <0|1> (14799)
       BT <slot> <id> (11030), BDONE <slot> <n> <stop|length|cancel> <ms> (11040)   between any two request lines
       ERR <message> (11643, 11911, 11992, 11997, 12013, 12019, 12117)
  log  the startup lines the harness reads (4862, 4902, 4923, 4980, 4985, 5419, 5701, 5833, 5887, 5893, 7234, 9829) and
       per request: parked / skip parking / restored / gave back / takes (8570, 8676, 11557, 12374, 12446, 12512,
       14782), then 'strata serve: prompt <n> tokens = <r> reused + ...' after the DONE / BADM (14820)

What it models, as generate.cpp does it (simplified where the harness cannot tell the difference):
  - where a request starts reading: the live session (12224), the checkpoints at turn boundaries and the root
    (12225-12229, 13405-13431), an idle batch slot holding the start of a text prompt (12258-12278), a parked
    conversation (12279, 12330; restored 12457-12515); the outgoing conversation is parked at a switch (12367-12384)
    when it has --conversation-cache-min-tokens or more (8569) and no batch slot parks it (STRATA_SLOT_PARK, 12370);
    ckpt=0 takes no turn checkpoint and leaves nothing to continue or park, but still saves the root (11927-11931,
    13423-13431, 14543)
  - a session holds the prompt and every token a window fed (14432, 14541): all but the last token after a STOP
  - the M-RoPE table: region 0 is rebuilt by every request (12026-12114); with STRATA_MROPE_SLOTS=1, --vision and
    --batch one region per slot (4936-4987), copied at an admission when either side holds an image layout
    (10209-10243), not with STRATA_MROPE_REGION_SKIP=1; without the regions image and text rows are refused side by
    side (12005-12025).  A slot decodes the right tokens only while the region it reads is its request's layout; the
    layout is what the positions after the pictures depend on, sum(rows - max(nx, ny)) over the pictures (12082)
  - greedy tokens as a function of the context (tokens and pictures) only, so solo and slot runs agree exactly
  - the solo path's windows: 1 + up to --spec - 1 drafts, all accepted.  --suffix-draft (default 3) makes --spec N
    into N + 2 with the MTP at N (2938-2941) and lets the draft policy, learned over the process's timed rounds
    (9964, 14292-14297, 14498-14503), choose the window: in this fake the window sizes then depend on how many timed
    rounds the process has run; with --suffix-draft 0 they follow from the state alone.  An admission's one window is
    untimed (14317).  Batch slots take one token per window (the pipelined path)
  - the pipelined groups (4846-4863): one group's window per tick, its active slots' BT lines in slot order

Faults (VISION_CHECK_FAKE_FAULT=a,b,...; the new binary only, never with --fake-old):
  beside     an image slot's tokens are wrong while any text slot decodes
  stale      with the regions on, every slot reads the identity region (the copy never reaches it; no warning): a fault
             every process with STRATA_MROPE_SLOTS=1 shares, which only the anchors without the regions can see
  resumed    a request that resumed and then read past its resume point decodes wrong tokens in a slot while another
             slot decodes - text and image alike (an older path difference: the harness must say INCONCLUSIVE)
  pathdep    a session decoded in multi-row windows carries their shapes into what continues from it (TESTLOG #59 (4)):
             a request resumed from such a state decodes other tokens than one resumed from a state of other windows
  textdrift  a text slot's tokens are wrong while another slot decodes (batch != solo)
  noregions  STRATA_MROPE_SLOTS=1 does not turn the regions on
  nopark     the conversation cache never parks (the RAM floor's skip)
  slots      CUDA0's expert cache has fewer slots than the production engine's
  warning    one more WARNING line at the start
  samegrid   (the encoder) every picture on a 16 x 16 grid, so two pictures sit alike after them
  encstall   (the encoder) never answers an ENC line
--fake-old: the last release - no regions whatever the environment, no faults.
VISION_CHECK_FAKE_TICK_MS: what a batch window takes (default 0.3 ms); VISION_CHECK_FAKE_WINDOW_MS: what the next solo
window's drafts take, the time a STOP has to arrive in (default 5 ms); VISION_CHECK_FAKE_EOS_AFTER: the reply token
from which an end-of-turn may come (default: never).

  (image encoder) run with --mmproj ...: READY <n_embd>, then ENC <png> <out> -> OK <n_tokens> <nx> <ny> <ms>, the file
  a strata-vision SVE1 record (tools/vision/strata_vision.cpp 11-14, 183, 226)
"""
import os, queue, re, struct, sys, threading, time, hashlib

TURN, IMAGE_PAD = 248045, 248056
EOS = (248044, 248046)
N_EMBD = 8                 # the fake model's width (a real record's n_embd must be the model's: 12048)
CHUNK = 512                # the config's --prefill 512
SHORT_READ = 64            # o.short_read (857): a part this short goes through the verify windows
PROMPT_CACHE = 6           # o.prompt_cache (826): checkpoints kept
CACHE_ROOT = 2048          # o.prompt_cache_root (845)
M64 = (1 << 64) - 1
WRONG = 0x5DEECE66D        # what a wrong position does to a token's context
PATH = 0x2545F4914F6CDD1D  # what another window history does to the numbers (pathdep)


def mix(h, tok, extra=0):
    h ^= (tok * 0x9E3779B97F4A7C15 + extra) & M64
    h = (h * 0xBF58476D1CE4E5B9) & M64
    return h ^ (h >> 29)


def token_of(h, k, eos_after):
    """The greedy token after a context of hash h; k: its place in the reply (an end-of-turn only from eos_after on)."""
    if k >= eos_after and (h >> 7) % 23 == 0:
        return EOS[1]
    return 1000 + (h >> 20) % 60000


def prompt_hash(ids, rows):
    """The hash of a prompt: its tokens, and at an <|image_pad|> run the picture's rows (rows: position -> row key)."""
    h = 0
    for i, t in enumerate(ids):
        h = mix(h, t, rows.get(i, 0))
    return h


def out(line):
    sys.stdout.write(line + "\n")
    sys.stdout.flush()


def log(line):
    sys.stderr.write(line + "\n")
    sys.stderr.flush()


# ---------------------------------------------------------------------------------------------------------------- encoder
def png_size(data):
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        return 640, 480
    return struct.unpack(">II", data[16:24])


def encoder():
    faults = set(filter(None, os.environ.get("VISION_CHECK_FAKE_FAULT", "").split(",")))
    out(f"READY {N_EMBD}")                                                                   # strata_vision.cpp 183
    for line in sys.stdin:
        p = line.split()
        if not p:
            continue
        if p[0] == "QUIT":
            return 0
        if p[0] != "ENC" or len(p) != 3:
            out("ERR expected: ENC <image> <output>")                                        # strata_vision.cpp 191
            continue
        if "encstall" in faults:             # an encoder that never answers (the harness's --line-timeout)
            time.sleep(3600)
        t0 = time.time()
        data = open(p[1], "rb").read()
        w, h = png_size(data)
        nx, ny = max(1, w // 32), max(1, h // 32)
        if "samegrid" in faults:             # every picture on one grid: their positions after them are the same
            nx = ny = 16
        n = nx * ny
        seed = hashlib.sha256(data).digest()
        with open(p[2], "wb") as f:              # the record generate.cpp reads (12045-12058): SVE1 n nx ny n_embd
            f.write(struct.pack("<5i", 0x31455653, n, nx, ny, N_EMBD))
            f.write(struct.pack(f"<{n * N_EMBD}f", *[((seed[i % 32] + i) % 251) / 251.0 for i in range(n * N_EMBD)]))
        out(f"OK {n} {nx} {ny} {1000 * (time.time() - t0):.0f}")                            # strata_vision.cpp 226
    return 0


# ---------------------------------------------------------------------------------------------------------------- engine
class Slot:
    def __init__(self):
        self.active = self.cached = self.img = self.stop = self.parks = self.resumed_read = False
        self.ids, self.h, self.x, self.k, self.produced, self.max_new, self.layout, self.t0 = [], 0, 0, 0, 0, 0, 0, 0.0
        self.wh = 0                          # the window history of the state it continues (pathdep)


class Fake:
    def __init__(self, argv):
        def opt(name, default, conv=int):    # the last one given wins, as generate.cpp reads its arguments in order
            at = [j for j, x in enumerate(argv) if x == name and j + 1 < len(argv)]
            return conv(argv[at[-1] + 1]) if at else default
        self.old = "--fake-old" in argv
        faults = set(filter(None, os.environ.get("VISION_CHECK_FAKE_FAULT", "").split(",")))
        self.faults = set() if self.old else faults
        self.batch = opt("--batch", 0)
        split = opt("--layer-split", "", str)
        self.split = [int(x) for x in split.split(",") if x.strip().isdigit()] if split and split != "auto" else []
        self.vision = "--vision" in argv
        self.max_context = opt("--max-context", 262144)
        self.min_tokens = opt("--conversation-cache-min-tokens", 0)
        self.cache_mib = opt("--conversation-cache-mib", 0)
        self.cache_slots = opt("--conversation-cache-slots", 0)
        self.min_free = opt("--conversation-cache-min-free-mib", 0)
        self.kv_resident = opt("--kv-resident", 0)
        spec = opt("--spec", 1)
        self.suffix_draft = max(0, opt("--suffix-draft", 3))          # (868: on by default)
        self.mtp_max_t = 0
        if self.suffix_draft > 0 and spec >= 2:                         # 2938-2941
            self.mtp_max_t, spec = spec, min(spec + 2, 8)
        self.spec = spec
        self.spec_mtp = self.mtp_max_t if self.mtp_max_t > 0 else spec
        self.rounds = 0                      # timed solo rounds so far: what the draft policy has learned from
        env = os.environ
        self.slot_park = env.get("STRATA_SLOT_PARK", "0") not in ("", "0")
        self.skip = env.get("STRATA_MROPE_REGION_SKIP", "0") not in ("", "0")
        on = env.get("STRATA_MROPE_SLOTS", "0") not in ("", "0")   # mrope_slots_env (generate.cpp 209-215)
        self.regions = on and self.vision and self.batch > 0 and not self.old and "noregions" not in self.faults
        self.tick_s = float(env.get("VISION_CHECK_FAKE_TICK_MS", "0.3")) / 1000.0
        self.window_s = float(env.get("VISION_CHECK_FAKE_WINDOW_MS", "5")) / 1000.0
        self.eos_after = int(env.get("VISION_CHECK_FAKE_EOS_AFTER", str(1 << 40)))
        n_st = len(self.split) + 1
        self.groups = n_st if self.batch and n_st > 1 and self.batch % n_st == 0 else 1
        self.gs = max(1, self.batch // self.groups)
        self.slots = [Slot() for _ in range(self.batch)]
        self.region0 = 0                     # the layout region 0 holds (0: the identity)
        self.region = [0] * self.batch       # each slot's region (with the regions)
        self.region_img = [False] * self.batch   # slot_mrope_img (10129)
        self.live, self.live_imgs, self.live_ok, self.live_wh = [], (), False, 0
        self.checks = []                     # (ids, imgs, wh) of each conversation checkpoint
        self.parked = []                     # {"live": (ids, imgs, wh), "checks": [...]}, oldest first
        self.main_slot, self.rr = -1, 0
        self.stop = threading.Event()
        self.q = queue.Queue()

    # ---- start
    def startup(self):
        n_st = len(self.split) + 1
        if n_st > 1:
            later = ",".join(map(str, range(1, n_st)))
            log(f"strata generate: layer split across {n_st} GPUs: CUDA0, then CUDA{later} "
                f"(split {','.join(map(str, self.split))})")
        if self.batch:
            g, gs = self.groups, self.gs
            log(f"strata generate: --batch-groups auto: {g} group{'' if g == 1 else 's'} of {gs} slot"
                f"{'' if gs == 1 else 's'}")                                                            # 4862
            for k in range(n_st):
                log(f"strata generate: --batch: {self.batch} slot sessions on CUDA{k} (0.29 GiB each); "
                    f"16.69 GiB free")                                                                  # 4902
            log(f"strata generate: --batch {self.batch}: the slot sessions take {0.29 * self.batch:.2f} GiB of VRAM on "
                f"CUDA0 that the expert cache would otherwise hold")                                    # 4923
        if self.regions:
            stride = (self.max_context + 64 + 63) // 64 * 64
            log(f"strata generate: --vision with --batch {self.batch}: the image position table holds {1 + self.batch} "
                f"regions of {stride} rows (the main sessions' and one per slot), "
                f"{(1 + self.batch) * stride * 12 / 1048576.0:.1f} MiB per GPU; image and text requests share the "
                f"batch")                                                                                # 4980
            if self.skip:
                log("strata generate: WARNING: STRATA_MROPE_REGION_SKIP=1 (a test's negative control): image slots "
                    "decode at the wrong positions")                                                    # 4985
        if "warning" in self.faults:
            log("strata generate: WARNING: a warning the production engine does not print (VISION_CHECK_FAKE_FAULT)")
        slots = [6144, 6656, 6144, 5632][:n_st] + [5000] * max(0, n_st - 4)
        if "slots" in self.faults:
            slots[0] = 6100
        self.expert_slots = slots
        log(f"strata generate: expert cache auto: 15.62 GiB free, 700 MiB reserved (+0 MiB for the draft head) -> "
            f"{slots[0]} slots")                                                                        # 5419
        log(f"strata generate: expert cache {slots[0]} slots, 14.27 GiB of VRAM; policy is")           # 5701
        bounds = [0] + self.split + [48]
        for k in range(1, n_st):
            log(f"strata generate: layer split, CUDA{k}: 17.80 GiB free of 23.56, room for experts 16.40 GiB")  # 5833
            log(f"strata generate: layer split: CUDA{k} runs layers {bounds[k]}-{bounds[k + 1] - 1}, expert cache "
                f"{slots[k]} slots (14.44 GiB), {slots[k]} of its {slots[k]} profiled pairs; slot 0 verified")  # 5887
        if n_st > 1:
            log(f"strata generate: layer split: CUDA0 runs layers 0-{bounds[1] - 1}")                   # 5893
            log("strata serve: layer split: 100% of the experts resident, the prompt path's streamed ring "
                "96 slots")                                                                             # 7234
        log("strata serve: 1415 MiB of VRAM free with everything loaded")                             # 9829
        batch = (f" batch_slots={self.batch} slot_cache=1 batch_groups={self.groups}" +
                 (" mrope_slots=1" if self.regions else "")) if self.batch else ""
        out(f"INFO logit_bias=1 context={self.max_context} kv=int8 kv_resident={self.kv_resident} "
            f"expert_slots={sum(slots)} expert_cache_mib=0 expert_slots_primary={slots[0]} expert_cache_primary_mib=0 "
            f"spec={self.spec} mtp_max={self.mtp_max_t} lookup={self.suffix_draft} vram_free_mib=1415 cvec=none "
            f"arena_mib=0 pool_workers=63 pcie_frac=0.00 spec_min_p=0.50 conversation_cache_mib={self.cache_mib} "
            f"conversation_cache_slots={self.cache_slots} conversation_cache_min_free_mib={self.min_free} "
            f"conversation_cache_min_tokens={self.min_tokens} tail_role_token=-1 vram_elastic=0{batch} "
            f"engine=0.1.43")                                                                         # 9877-9898
        out(f"READY {self.max_context} stop")                                                         # 9954

    # ---- the stdin thread: STOP is never queued (9795)
    def reader(self):
        for line in sys.stdin:
            line = line.rstrip("\n").rstrip("\r")
            if line == "STOP":
                self.stop.set()
                continue
            self.q.put(line)
        self.q.put(None)

    def run(self):
        threading.Thread(target=self.reader, daemon=True).start()
        self.startup()
        while True:
            if any(s.active for s in self.slots):
                try:
                    line = self.q.get_nowait()
                except queue.Empty:
                    self.tick()
                    time.sleep(self.tick_s)
                    continue
            else:
                line = self.q.get()
            if line is None or line == "QUIT":
                return 0
            if line.startswith("BSTOP "):                                                              # 11606-11610
                b = int(line.split()[1])
                if 0 <= b < len(self.slots):
                    self.slots[b].stop = True
                continue
            # a request reads its prompt through every stage: the groups in flight finish first (11583-11590), so
            # their BT / BDONE lines come before the request's RESUME
            for g in range(self.groups):
                self.tick(g)
            self.request(line)

    # ---- the slots: one group's window per tick (the pipelined path, 11011-11051)
    def tick(self, group=None):
        for j in range(self.groups):
            g = (self.rr + j) % self.groups if group is None else group
            act = [t for t in range(self.gs) if self.slots[g * self.gs + t].active]
            if act or group is not None:
                break
        else:
            return
        if not act:
            return
        self.rr = g + 1
        width = max(act) + 1
        others = [(b, s.img) for b, s in enumerate(self.slots) if s.active]
        for t in range(width):
            b = g * self.gs + t
            s = self.slots[b]
            if not s.active:
                s.cached = False                 # its pad row writes its state (11121)
                continue
            beside = [img for c, img in others if c != b]
            reads = self.region[b] if self.regions else self.region0
            if self.regions and "stale" in self.faults:
                reads = 0
            wrong = reads != s.layout
            if s.img and "beside" in self.faults and any(not img for img in beside):
                wrong = True
            if s.resumed_read and "resumed" in self.faults and beside:
                wrong = True
            if not s.img and "textdrift" in self.faults and beside:
                wrong = True
            hh = mix(s.h, s.x)
            y = token_of(hh ^ WRONG if wrong else hh, s.k, self.eos_after)
            out(f"BT {b} {y}")                                                                       # 11030
            s.produced += 1
            s.k += 1
            s.ids.append(s.x)                    # the token the window fed joins the slot's tokens (11037)
            s.h = hh
            fin = ("stop" if y in EOS else "cancel" if s.stop else "length" if s.produced >= s.max_new else None)
            if fin:                                                                                  # 11035
                out(f"BDONE {b} {s.produced} {fin} {1000 * (time.time() - s.t0):.1f}")               # 11040
                s.active = False
                s.cached = not s.img             # (11045: an image slot is never a cache)
            else:
                s.x = y

    # ---- a request
    def request(self, line):
        admit_slot, admit_max_new = -1, 0
        if line.startswith("BGEN ") or line.startswith("BGENI "):                                    # 11636-11651
            m = re.match(r"BGENI? (-?\d+) (-?\d+)(.*)$", line)
            b = int(m.group(1)) if m else -1
            if not m or b < 0 or b >= len(self.slots) or self.slots[b].active or int(m.group(2)) < 1:
                out(f"ERR BGEN: no such free slot (--batch {self.batch}) or a bad max_new")          # 11643
                return
            admit_slot, admit_max_new = b, int(m.group(2))
            line = ("GENI 1" if line.startswith("BGENI ") else "GEN 1") + m.group(3)
        self.stop.clear()                        # a STOP that arrived between requests is stale (11670)
        geni = line.startswith("GENI ")
        if not geni and not line.startswith("GEN "):
            out("ERR expected: GEN <max_new> <id,id,...> or GENI <max_new> <file> <id,id,...>")      # 11911
            return
        parts = line.split()[1:]
        keys, i = {}, 1
        while i < len(parts) and "=" in parts[i]:
            k, _, v = parts[i].partition("=")
            keys[k] = v
            i += 1
        emb = None
        if geni and i < len(parts):
            emb, i = parts[i], i + 1
        try:
            max_new = int(parts[0])
            ids = [int(x) for x in parts[i].split(",")] if i < len(parts) else []
        except ValueError:
            max_new, ids = 0, []
        if max_new < 1 or not ids or (geni and not emb):
            out("ERR bad request: max_new")                                                          # 11992
            return
        req_ckpt = keys.get("ckpt", "1") != "0"
        n = len(ids)
        if geni and not self.vision:
            out("ERR this engine was started without --vision")                                     # 11997
            return
        if self.vision and self.slots and not self.regions:                                         # 12005-12025
            busy = next((b for b, s in enumerate(self.slots) if s.active), -1)
            busy_img = next((b for b, s in enumerate(self.slots) if s.active and s.img), -1)
            if geni and busy >= 0:
                out(f"ERR vision: an image request is not taken while batch slot {busy} decodes (image and text rows "
                    f"share one M-RoPE table); try again once the slots are idle")                   # 12013
                return
            if not geni and busy_img >= 0:
                out(f"ERR vision: a text request is not taken while batch slot {busy_img} decodes an image request "
                    f"(image and text rows share one M-RoPE table); try again once it ends")         # 12019
                return
        imgs, rows, layout = [], {}, 0           # the pictures: (start, key); their rows; the positions after them
        if geni:
            err = ""
            try:
                recs = self.read_embeddings(emb)
            except (OSError, ValueError) as x:
                recs, err = [], str(x)
            k = j = 0
            while not err and j < n:
                if ids[j] != IMAGE_PAD:
                    j += 1
                    continue
                if k >= len(recs):
                    err = "the prompt has more images than the embeddings file"
                    break
                cnt, nx, ny, key = recs[k]
                k += 1
                if ids[j:j + cnt] != [IMAGE_PAD] * cnt:
                    err = f"image {k} has {cnt} rows but fewer <|image_pad|> tokens"
                    break
                imgs.append((j, key))
                for r in range(cnt):
                    rows[j + r] = (key + r) & M64
                layout += cnt - max(nx, ny)
                j += cnt
            if not err and k != len(recs):
                err = "the embeddings file has more images than the prompt"
            if err:                              # the table back to the identity, then ERR (12104-12111)
                self.region0 = 0
                out(f"ERR {err}")
                return
        if geni or self.region0 != 0:            # positions for every cell; the identity again for text (12026-12114)
            self.region0 = layout
        if n + max_new + 8 > self.max_context:
            out(f"ERR prompt ({n} tokens) + max_new ({max_new}) exceeds the context ({self.max_context})")   # 12117
            return
        imgs = tuple(imgs)
        t0 = time.time()

        def below(L):
            return tuple(x for x in imgs if x[0] < L)

        def starts_with(pre, pre_imgs):          # 12143-12149: tokens and pictures, at most n - 1 of them
            L = len(pre)
            return 1 <= L <= n - 1 and list(pre) == ids[:L] and below(L) == tuple(pre_imgs)

        resume, from_live, src_wh = 0, False, 0  # 12221-12230
        if self.live_ok and starts_with(self.live, self.live_imgs):
            resume, from_live, src_wh = len(self.live), True, self.live_wh
        for c_ids, c_imgs, c_wh in self.checks:
            if len(c_ids) > resume and starts_with(c_ids, c_imgs):
                resume, from_live, src_wh = len(c_ids), False, c_wh
        slot_source, slot_tokens = -1, 0         # 12254-12278: text requests only
        if not imgs:
            for b, s in enumerate(self.slots):
                if not s.active and s.cached and len(s.ids) > max(resume, slot_tokens) and starts_with(s.ids, ()):
                    slot_source, slot_tokens = b, len(s.ids)
        best, best_n, best_live, best_wh = -1, 0, False, 0   # conversations.best (conversation_cache.hpp 326-341)
        for idx in range(len(self.parked) - 1, -1, -1):
            e = self.parked[idx]
            for (c_ids, c_imgs, c_wh), is_live in [(e["live"], True)] + [(c, False) for c in e["checks"]]:
                if len(c_ids) > best_n and starts_with(c_ids, c_imgs):
                    best, best_n, best_live, best_wh = idx, len(c_ids), is_live, c_wh
        incoming = self.parked.pop(best) if best >= 0 and best_n > max(resume, slot_tokens) else None   # 12330
        if incoming:
            slot_source = -1
        park_due = not from_live or incoming is not None or slot_source >= 0                         # 12367
        owner = self.main_slot if self.main_slot >= 0 and self.slots[self.main_slot].parks else -1  # 12370
        self.main_slot = -1
        if park_due and owner >= 0:
            log(f"strata serve: conversation cache: {len(self.live)} tokens not parked here: slot {owner} parks them "
                f"with its reply")                                                                    # 12374
        elif park_due:
            self.park()
        if slot_source >= 0:                     # 12385-12450: the slot's sessions back into the main ones
            s = self.slots[slot_source]
            self.checks = []
            self.live, self.live_imgs, self.live_wh = list(s.ids), (), s.wh
            resume, from_live, src_wh = slot_tokens, True, s.wh
            log(f"strata batch: slot {slot_source} gave back {slot_tokens} tokens of this conversation (all it holds) "
                f"in 0.5 ms")                                                                         # 12446
        if incoming:                             # 12457-12515
            self.live, self.live_imgs, self.live_wh = list(incoming["live"][0]), incoming["live"][1], incoming["live"][2]
            self.checks = list(incoming["checks"])
            resume, from_live, src_wh = best_n, best_live, best_wh
            log(f"strata serve: conversation cache: restored {resume} tokens ({'live' if from_live else 'checkpoint'}) "
                f"in 3.0 ms; parked={len(self.parked)} bytes={len(self.parked) * 1048576}")             # 12512
        self.checks = [c for c in self.checks if len(c[0]) <= resume and starts_with(c[0], c[1])]   # 12525-12531
        self.live_ok = False                     # until this request has finished (12532)
        if resume == 0:
            self.checks = []
        out(f"RESUME {resume}")                                                                       # 12628
        # the read: up to the root, up to the last turn boundary (the checkpoints there), up to the last token
        turn_at = next((j for j in range(n - 1, resume, -1) if ids[j] == TURN), -1)                   # 13405-13408
        last_turn = turn_at
        if not req_ckpt:
            turn_at = -1                         # ckpt=0 drops the split there (13424)
        root_at = -1
        if resume == 0:                          # 13425-13431: the root is still looked for before it
            for j in range(1, max(last_turn, 0)):
                if ids[j] == TURN:
                    root_at = j if j >= CACHE_ROOT else -1
                    break
        at, cancelled = resume, False
        for to in sorted({c for c in (root_at, turn_at, n - 1) if c > resume}):
            if self.stop.is_set():
                cancelled = True
                break
            step = to - at if to - at <= SHORT_READ else CHUNK
            for q in range(at, to, step):
                out(f"PP {min(to, q + step)} {n} {1000 * (time.time() - t0):.0f} 1000.0")              # 9021, 12817
            at = to
            if to in (root_at, turn_at):
                self.checks.append((tuple(ids[:to]), below(to), src_wh))
                if len(self.checks) > PROMPT_CACHE:
                    self.checks.pop(0)
        prompt_ms = 1000 * (time.time() - t0)
        out(f"REUSED {resume}")                                                                       # 13568
        # the windows: the first holds the last prompt token alone (untimed, 14317), then 1 + drafts, all accepted
        consumed = list(ids[:-1])
        hf = prompt_hash(ids[:-1], rows)         # the session's tokens so far; x: the token the next window feeds
        if "pathdep" in self.faults and src_wh:
            hf = mix(hf, src_wh, PATH)           # the state it resumed carries its windows' numbers (TESTLOG #59 (4))
        x, k, produced, finish, first = ids[-1], 0, 0, "cancel" if cancelled else "length", True
        wh, accepted, offered = src_wh, 0, 0
        d0 = time.time()
        while not cancelled and produced < max_new:
            if first:
                T = 1
            elif self.suffix_draft > 0:          # the draft policy, learned over the process's timed rounds (9964)
                T = 1 + ((hf >> 11) + min(self.rounds // 8, 3)) % max(1, self.spec)
            else:                                # the MTP's windows alone: from the state
                T = 1 + (hf >> 11) % max(1, self.spec_mtp)
            if not first:
                offered += T - 1                 # (14432-14435: every draft is accepted here)
                accepted += T - 1
                if admit_slot < 0:
                    self.rounds += 1
            if T > 1:
                wh = mix(wh, T, 0x77) | 1
            first = False
            feeds, outs, hh = [x], [], hf
            for r in range(T):
                hh = mix(hh, feeds[r])
                outs.append(token_of(hh, k + r, self.eos_after))
                if r + 1 < T:
                    feeds.append(outs[-1])
            consumed += feeds                    # every token the window fed is in the session (14432)
            hf, x, k = hh, outs[-1], k + T
            eos = False
            for y in outs:
                if produced >= max_new or eos:
                    break
                out(f"T {y}")                                                                         # 14440
                produced += 1
                eos = y in EOS
            if eos:
                finish = "stop"
                break
            if admit_slot < 0 and produced < max_new:
                self.stop.wait(self.window_s)    # the next window's drafts (14455) before the STOP test (14505)
            if self.stop.is_set():
                finish = "cancel"
                break
        if not cancelled:                        # 14538-14544
            self.live, self.live_imgs, self.live_ok, self.live_wh = consumed, below(len(consumed)), req_ckpt, wh
        decode_ms = 1000 * (time.time() - d0)
        out(f"DONE {produced} {n} {prompt_ms:.1f} {decode_ms:.1f} {finish} {accepted} {offered} {resume} 0 0 0 0 0.0 "
            f"{n - resume} 0")                                                                        # 14687
        if admit_slot >= 0:                      # 14695-14799
            cont = (not cancelled and produced == 1 and admit_max_new > 1 and finish == "length" and
                    len(self.live) == len(consumed))
            if cont:
                b = admit_slot
                if self.regions and not self.skip and (self.region0 != 0 or self.region_img[b]):   # 10214-10243
                    self.region[b] = self.region0
                    self.region_img[b] = self.region0 != 0
                s = self.slots[b]
                s.active, s.stop, s.cached = True, False, False
                s.ids, s.h, s.x, s.k, s.wh = list(self.live), hf, x, k, wh
                s.produced, s.max_new, s.t0 = 1, admit_max_new, time.time()
                s.img = bool(self.live_imgs)     # (14734)
                s.layout = layout
                s.resumed_read = resume > 0 and n - resume >= 8
                s.parks = self.slot_park and not s.img   # (14742)
                if s.parks:
                    self.main_slot = b
                log(f"strata batch: slot {b} takes {len(self.live)} tokens (copied in 0.5 ms)")       # 14782
            out(f"BADM {admit_slot} {1 if cont else 0}")                                             # 14799
        log(f"strata serve: prompt {n} tokens = {resume} reused + {n - resume} read in {prompt_ms:.0f} ms (1000.0 "
            f"tok/s), {produced} generated in {decode_ms:.0f} ms (100.0 tok/s), drafts accepted {accepted} of "
            f"{offered}, {len(self.checks)} checkpoints")                                            # 14820

    def park(self):
        """The outgoing conversation into the conversation cache (park_current_body, 8563-8756), if it is worth a
        slot; the copies of it a turn back go first (STRATA_SUPERSEDE_PREFIX)."""
        if not (self.cache_mib > 0 and self.cache_slots > 0) or not self.live_ok or not self.live:
            return                                                                                    # 8564
        if "nopark" in self.faults:
            log(f"strata serve: conversation cache: skip parking (physical RAM admission; need 1 MiB plus "
                f"{self.min_free} MiB floor, a fake engine's fault; evicted 0, {len(self.parked)} still "
                f"parked)")                                                                             # 8676
            return
        if self.min_tokens > 0 and len(self.live) < self.min_tokens:
            log(f"strata serve: conversation cache: skip parking ({len(self.live)} tokens, below the "
                f"{self.min_tokens} minimum; parked={len(self.parked)})")                             # 8570
            return
        mine = (tuple(self.live), tuple(self.live_imgs), self.live_wh)
        self.parked = [e for e in self.parked
                       if not (len(e["live"][0]) <= len(mine[0]) and mine[0][:len(e["live"][0])] == e["live"][0])]
        while len(self.parked) >= self.cache_slots:
            self.parked.pop(0)
        self.parked.append({"live": mine, "checks": list(self.checks)})
        log(f"strata serve: conversation cache: parked {len(self.live)} tokens in 2.0 ms; parked={len(self.parked)} "
            f"bytes={len(self.parked) * 1048576} evictions=0 snapshot_bytes=1048576 reused_kv_bytes=0 capture=async "
            f"capture_ms=1.0 wait_ms=0.0")                                                            # 11557

    @staticmethod
    def read_embeddings(path):
        recs = []
        with open(path, "rb") as f:
            while True:
                hdr = f.read(20)
                if not hdr:
                    break
                magic, cnt, nx, ny, width = struct.unpack("<5i", hdr)
                if len(hdr) != 20 or magic != 0x31455653 or nx * ny != cnt or width != N_EMBD:
                    raise ValueError(f"bad embeddings file (expected strata-vision records of width {N_EMBD})")
                body = f.read(cnt * width * 4)
                if len(body) != cnt * width * 4:
                    raise ValueError("short embeddings file")
                recs.append((cnt, nx, ny, int.from_bytes(hashlib.sha256(struct.pack("<3i", cnt, nx, ny) + body)
                                                         .digest()[:8], "little")))
        return recs


def main():
    if "--mmproj" in sys.argv[1:]:
        return encoder()
    if "--serve" not in sys.argv[1:]:
        print("the fake engine runs with --serve (or as the image encoder, with --mmproj)", file=sys.stderr)
        return 2
    return Fake(sys.argv[1:]).run()


if __name__ == "__main__":
    sys.exit(main())
