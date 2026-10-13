#!/usr/bin/env python3
"""CPU-only smoke test of tools/vision_batch_check.py: the harness against tools/vision_check_fake_engine.py, a fake
engine and image encoder that speak the same stdin/stdout protocol (no model, no GPU, no production process touched).

Every mode of the harness runs once against a correct fake engine (GO expected) and then against faulty ones, each of
which must give the verdict and exit code the harness promises (NO-GO 2, INCONCLUSIVE 3, a usage error 64) and name the
right criterion.  text-ab runs first: its --out JSON is the --expect-from of the engine-mode cases, as in the GPU window.
  text-ab               GO (with the vision-off arms, and replies that end early)
  engine                GO
  negative              GO (every image comparison fails with STRATA_MROPE_REGION_SKIP=1)
  fault-beside          NO-GO: an image slot is wrong beside text slots (the text controls match)
  fault-stale           NO-GO: every image slot reads the identity in every process with the regions (refs and checks
                        agree); only the anchors without the regions see it
  text-ab-stale         NO-GO: the same fault against the old binary's image requests
  fault-resumed         INCONCLUSIVE: the follow-ups differ, and so do their text controls
  fault-nopark          NO-GO: the RAM-cache follow-up is not restored (REUSED, no 'restored' line)
  fault-noregions       NO-GO: no regions despite STRATA_MROPE_SLOTS=1 (and the image scenarios are refused)
  fault-slots           NO-GO: fewer expert-cache slots than the old binary's under the same flags (--expect-from)
  fault-slots-default   INCONCLUSIVE: the same against production's k5 counts (other flags)
  fault-warning         NO-GO: a WARNING line at the start
  suffix-on             INCONCLUSIVE: the suffix drafter on (its windows learned per process) and a state that carries
                        its windows' shapes: the promotions' solo legs end apart and the continuations differ
  coverage              INCONCLUSIVE: every reply may end at once, so the scenarios are not exercised (no NO-GO)
  samegrid              INCONCLUSIVE: the encoder puts both pictures on one grid, so a mix-up of their regions would
                        not show ((iii)'s layout criterion)
  encoder-stall         INCONCLUSIVE in about 2 s: the encoder never answers, --line-timeout ends it, no engine starts
  text-ab-drift         NO-GO: the new binary's text slots differ from solo and from the old binary
  usage                 exit 64 for an unknown argument
  server mode           GO against a stand-in /v1/chat/completions on an ephemeral 127.0.0.1 port, and INCONCLUSIVE
                        against one whose replies change under load (the text-only control differs already)

The config is production's (--template, read only) with the engine, the encoder and the working directory pointed at
the fake; the tokenizer is the real one (--tokenizer, read only), so the prompts are the ones the GPU window sends.

  python3 tools/vision_check_smoke.py [--only NAME] [--keep]
"""
import argparse, hashlib, http.server, json, os, shlex, shutil, subprocess, sys, tempfile, threading, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
HARNESS = HERE / "vision_batch_check.py"
FAKE = HERE / "vision_check_fake_engine.py"
TEMPLATE = "/media/raid0/StrataQ38FN/linux/configs/T16-143-prod262r-vision-cpu.json"
# what the smoke test needs of the template when it is not there: production's shape (4 GPUs, a layer split, 16 slots)
FALLBACK = {
    "args": ["--prefill", "512", "--spec", "4", "--max-context", "262144", "--kv-resident", "32768",
             "--conversation-cache-mib", "98304", "--conversation-cache-slots", "320",
             "--conversation-cache-min-free-mib", "16384", "--conversation-cache-min-tokens", "1024", "--vision"],
    "tokenizer": "/media/raid0/StrataQ38FN/Strata-data/packs/unsloth-ud-iq4_xs/tokenizer",
    "gpu": [0, 1, 2, 3], "layer_split": "12,25,37", "parallel": 16,
    "env": {"STRATA_SLOT_PARK": "1", "STRATA_PARK_ASYNC": "1", "STRATA_CONV_TIER_DIR": "/tmp/unused",
            "STRATA_CONV_TIER_MIN_TOKENS": "8192"},
    "vision": {"exe": "", "mmproj": "", "model": "", "gpu": False, "threads": 32, "max_tokens": 1024},
}
SMALL = ["--batch", "16", "--max-new", "24", "--text-max-new", "64", "--long-tokens", "1200", "--promote-after", "10",
         "--line-timeout", "60", "--start-timeout", "60"]
EXPECT = ["--expect-from", "@EXPECT"]
GO_ENGINE = {"startup: the slot regions": "GO", "startup: the expert-cache": "GO", "opt-in:": "GO",
             "text slots beside": "GO", "image slots == their references": "GO", "(iii) the two pictures": "GO",
             "(iv) live": "GO", "(iv) RAM cache": "GO", "promotion, the server's order": "GO",
             "promotion, text first": "GO", "requests resumed": "GO", "text controls": "GO", "coverage:": "GO",
             "every engine run": "GO"}

# name, harness arguments, fake engine environment, expected exit, {criterion substring: expected status}
CASES = [
    ("text-ab", ["--text-ab", "@OLD", "--vision-off-arms"], {"VISION_CHECK_FAKE_EOS_AFTER": "20"}, 0,
     {"batch == solo": "GO", "== its old-binary arm's": "GO", "a slot's state back": "GO", "image requests alone": "GO",
      "startup: the slot regions": "GO", "startup: the expert-cache": "GO", "requests resumed where the A/B": "GO"}),
    ("engine", EXPECT, {}, 0, GO_ENGINE),
    ("negative", ["--negative-control"] + EXPECT, {}, 0,
     {"every image comparison FAILS": "GO", "text slots and the text": "GO", "startup: the slot regions": "GO",
      "no WARNING": "GO", "requests resumed": "GO"}),
    ("fault-beside", EXPECT, {"VISION_CHECK_FAKE_FAULT": "beside"}, 2,
     {"image slots == their references": "NO-GO", "text slots beside": "GO", "(iv) live": "NO-GO",
      "(iv) RAM cache": "NO-GO", "text controls": "GO"}),
    ("fault-stale", EXPECT, {"VISION_CHECK_FAKE_FAULT": "stale"}, 2,
     {"image slots == their references": "NO-GO", "(iv) live": "NO-GO", "(iv) RAM cache": "NO-GO",
      "text slots beside": "GO", "text controls": "GO", "requests resumed": "GO"}),
    ("text-ab-stale", ["--text-ab", "@OLD"], {"VISION_CHECK_FAKE_FAULT": "stale"}, 2,
     {"image requests alone": "NO-GO", "batch == solo": "GO", "== its old-binary arm's": "GO",
      "a slot's state back": "GO"}),
    ("fault-resumed", EXPECT, {"VISION_CHECK_FAKE_FAULT": "resumed"}, 3,
     {"(iv) live": "INCONCLUSIVE", "(iv) RAM cache": "INCONCLUSIVE", "text controls": "INCONCLUSIVE",
      "image slots == their references": "GO", "text slots beside": "GO", "promotion, text first": "GO"}),
    ("fault-nopark", EXPECT, {"VISION_CHECK_FAKE_FAULT": "nopark"}, 2, {"requests resumed": "NO-GO"}),
    ("fault-noregions", EXPECT, {"VISION_CHECK_FAKE_FAULT": "noregions"}, 2,
     {"startup: the slot regions": "NO-GO", "every engine run": "NO-GO"}),
    ("fault-slots", EXPECT, {"VISION_CHECK_FAKE_FAULT": "slots"}, 2,
     {"startup: the expert-cache": "NO-GO", "image slots == their references": "GO"}),
    ("fault-slots-default", [], {"VISION_CHECK_FAKE_FAULT": "slots"}, 3,
     {"startup: the expert-cache": "INCONCLUSIVE", "image slots == their references": "GO"}),
    ("fault-warning", EXPECT, {"VISION_CHECK_FAKE_FAULT": "warning"}, 2,
     {"no WARNING": "NO-GO", "image slots == their references": "GO"}),
    ("suffix-on", EXPECT + ["--extra=--pcie-frac 0 --adapt-every 1000000 --no-prefill-borrow"],
     {"VISION_CHECK_FAKE_FAULT": "pathdep"}, 3,
     {"promotion, the server's order": "INCONCLUSIVE", "promotion, text first": "INCONCLUSIVE",
      "image slots == their references": "GO", "(iv) live": "GO", "(iv) RAM cache": "GO", "text slots beside": "GO",
      "requests resumed": "GO"}),
    ("coverage", EXPECT, {"VISION_CHECK_FAKE_EOS_AFTER": "0"}, 3,
     {"coverage:": "INCONCLUSIVE", "image slots == their references": "GO", "text slots beside": "GO"}),
    ("samegrid", EXPECT, {"VISION_CHECK_FAKE_FAULT": "samegrid"}, 3,
     {"(iii) the two pictures": "INCONCLUSIVE", "image slots == their references": "GO", "(iv) live": "GO"}),
    ("encoder-stall", EXPECT + ["--line-timeout", "2"], {"VISION_CHECK_FAKE_FAULT": "encstall"}, 3,
     {"every engine run": "INCONCLUSIVE"}),
    ("text-ab-drift", ["--text-ab", "@OLD"], {"VISION_CHECK_FAKE_FAULT": "textdrift"}, 2,
     {"batch == solo": "NO-GO", "== its old-binary arm's": "NO-GO"}),
    ("usage", ["--no-such-argument"], {}, 64, {}),
    ("server", ["--url", "@URL"], {}, 0, {"text requests beside images": "GO", "image requests in a busy": "GO"}),
    ("server-drift", ["--url", "@URL-DRIFT"], {}, 3,
     {"text requests beside images": "INCONCLUSIVE", "image requests in a busy": "INCONCLUSIVE"}),
]


class FakeServer:
    """A stand-in for serve/server.py's /v1/chat/completions on an ephemeral 127.0.0.1 port (server mode): the reply is
    a hash of the request's messages - the same alone and under load - unless `drift`, where a request that arrives
    while another is in flight gets another reply (a server whose settings are not exact)."""

    def __init__(self, drift=False):
        self.busy, self.lock, self.drift = 0, threading.Lock(), drift
        fake = self

        class H(http.server.BaseHTTPRequestHandler):
            def log_message(self, *_):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
                with fake.lock:
                    fake.busy += 1
                    crowded = fake.busy > 1
                time.sleep(0.4)              # (long enough for the next staggered request to arrive meanwhile)
                text = hashlib.sha256(json.dumps(body.get("messages"), sort_keys=True).encode()).hexdigest()
                if fake.drift and crowded:
                    text = text[::-1]
                with fake.lock:
                    fake.busy -= 1
                out = json.dumps({"choices": [{"message": {"role": "assistant", "content": text}}]}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(out)))
                self.end_headers()
                self.wfile.write(out)

        self.httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}/v1"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self):
        self.httpd.shutdown()


def wrapper(path, extra=""):
    """An executable that runs the fake (the harness starts an engine binary by its path)."""
    path.write_text(f"#!/bin/sh\nexec {shlex.quote(sys.executable)} {shlex.quote(str(FAKE))} {extra} \"$@\"\n")
    path.chmod(0o755)
    return str(path)


def config(work, template, tok):
    cfg = json.loads(Path(template).read_text()) if template and Path(template).is_file() else json.loads(
        json.dumps(FALLBACK))
    new = wrapper(work / "fake-new")
    cfg.update({"exe": new, "cwd": str(work / "cwd"), "lib_dirs": [], "tokenizer": tok or cfg["tokenizer"]})
    cfg.pop("log", None)
    cfg["vision"] = dict(cfg.get("vision") or {}, exe=new, mmproj="fake.mmproj", model="fake.gguf", gpu=False)
    env = dict(cfg.get("env") or {})
    if "STRATA_CONV_TIER_DIR" in env:
        env["STRATA_CONV_TIER_DIR"] = str(work / "tier")
    cfg["env"] = env
    (work / "cwd").mkdir(exist_ok=True)
    (work / "tier").mkdir(exist_ok=True)
    path = work / "fake-config.json"
    path.write_text(json.dumps(cfg, indent=1))
    return path, new


def rows_like(rows, kind, key):
    return [r for r in rows if r["kind"] == kind and key in r["check"]]


def structure(name, j, problems):
    """What a run's --out must hold beside the verdict: the checks, the runs, every request with its REUSED."""
    rows, runs = j.get("checks") or [], j.get("runs") or {}
    if name == "encoder-stall":              # (no engine starts: the encoder gives up first, in about 2 s)
        if runs or not any("image encoder said nothing" in (r.get("detail") or "") for r in rows):
            problems.append(f"encoder-stall: an engine started, or the encoder's timeout is not named: {rows[:2]}")
        return
    if not rows or not (runs or name.startswith("server")):
        problems.append("no checks or no runs in --out")
        return
    for proc, run in runs.items():           # every stdout line of the fake was one the harness reads
        if isinstance(run, dict) and run.get("unexpected_lines"):
            problems.append(f"run {proc}: stdout lines the harness did not place: {run['unexpected_lines'][:3]}")
        if isinstance(run, dict) and run.get("desync"):
            problems.append(f"run {proc}: the harness failed in it: {run['desync']}")
    if name in ("engine", "negative", "fault-beside", "fault-stale", "fault-resumed", "suffix-on", "coverage"):
        reqs = [r for p in ("refs", "checks", "checks-negative", "optin") for r in (runs.get(p) or {}).get("requests", [])]
        if any(r.get("reused") is None for r in reqs if not r.get("err")):
            problems.append("a request without its REUSED line")
        if any(r.get("done") and not any("strata serve: prompt" in l for l in r.get("log", [])) for r in reqs):
            problems.append("a request whose log lines miss its 'strata serve: prompt' line")
        pics = runs.get("_pictures") or {}
        if not all(isinstance(pics.get(k), dict) and pics[k].get("nx") for k in ("A", "B")):
            problems.append(f"the pictures' grids are not in --out: {pics}")
    if name == "engine":
        for proc in ("refs", "checks", "optin"):
            if not (runs.get(proc) or {}).get("requests"):
                problems.append(f"run {proc}: no requests recorded")
        want = {"ivb.follow: REUSED": 2, "ivc.follow: REUSED": 2, "ivc.follow: the engine log says": 2,
                "ivb.follow (on the solo path): REUSED": 1, "ivc.follow (on the solo path): the engine log says": 1,
                "prom.first.img (continued in a slot): the engine log says": 2,
                "prom.first.img (continued on the solo path): the engine log says": 1,
                "prom.first.ctl (continued in a slot): the engine log says": 2,
                "prom.server.img (continued in a slot): REUSED": 2, "ivb.ctl.follow: REUSED": 2,
                "ivc.ctl.follow: the engine log says": 2, "ivc.ctl.follow (on the solo path): REUSED": 1}
        for key, n in want.items():
            got = rows_like(rows, "reuse", key)
            if len(got) != n or not all(r["ok"] for r in got):
                problems.append(f"{n} passing reuse rows '{key}' wanted, got {[(r['check'], r['ok']) for r in got]}")
        for kind, n in (("image", 14), ("image-resumed", 4), ("image-promoted", 4), ("control", 8), ("optin", 3),
                        ("layout", 1)):
            got = [r for r in rows if r["kind"] == kind]
            if len(got) < n or not all(r["ok"] for r in got):
                problems.append(f"{n} or more passing '{kind}' rows wanted, got {[(r['check'], r['ok']) for r in got]}")
        for r in rows:                       # every promotion decided: the solo legs ended alike
            if r["kind"] in ("image-promoted",) and r.get("detail"):
                problems.append(f"a promotion row is undecided against a correct fake: {r['check']}: {r['detail']}")
        groups = [r for r in rows if r["kind"] == "coverage" and r["scenario"] == "groups"]
        if len(groups) != 2 or not all(r["ok"] for r in groups):
            problems.append(f"the group coverage rows: {[(r['check'], r['ok']) for r in groups]}")
        if not (j.get("expectation") or {}).get("deciding"):
            problems.append(f"the expert-cache expectation is not the old binary's under the same flags: "
                            f"{j.get('expectation')}")
    if name == "negative":
        imgs = [r for r in rows if r["kind"] in ("image", "image-resumed", "image-promoted")]
        if not imgs or any(r.get("same") is not False for r in imgs):
            problems.append(f"every image row must differ in the negative control: "
                            f"{[(r['check'], r.get('same')) for r in imgs if r.get('same') is not False]}")
    if name == "fault-stale":                # refs and checks agree: only the rows against the anchors fail
        alone = [r for r in rows if r["kind"] in ("image-resumed", "image-promoted") and r.get("pair", "").endswith(
            ":alone")]
        solo = [r for r in rows if r["kind"] == "image-resumed" and r.get("pair", "").endswith(":solo")]
        if not alone or not all(r["ok"] for r in alone):
            problems.append(f"fault-stale: the rows against the same steps alone must match: "
                            f"{[(r['check'], r['ok']) for r in alone]}")
        if not solo or any(r["ok"] is not False for r in solo):
            problems.append(f"fault-stale: the rows against the solo path must fail: {[(r['check'], r['ok']) for r in solo]}")
    if name == "suffix-on":
        prom = [r for r in rows if r["kind"] == "image-promoted"]
        if not prom or any(r["ok"] is not None or not r.get("detail") for r in prom):
            problems.append(f"suffix-on: every promotion row must be undecided (its solo legs ended apart): "
                            f"{[(r['check'], r['ok'], r.get('detail', '')[:40]) for r in prom]}")
    if name == "text-ab":
        arms = [k for k in runs if k.startswith("text-")]
        if len(arms) != 5:
            problems.append(f"5 text arms wanted, got {arms}")
        if not rows_like(rows, "ab-image", "image part"):
            problems.append("no image-part rows in text-ab")
        if not rows_like(rows, "ab-back", "BSTOPped mid-reply"):
            problems.append("no give-back rows in text-ab")
        if [r for r in rows if r["kind"] == "ab" and "novision" in r["scenario"] and "old-novision" not in r["check"]]:
            problems.append("a vision-off arm is compared with old-vision as deciding")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--template", default=TEMPLATE, help="the production config the fake one is made from (read only)")
    ap.add_argument("--tokenizer", default="", help="the tokenizer directory (default: the template's)")
    ap.add_argument("--only", action="append", default=[], help="run only these cases (by name)")
    ap.add_argument("--keep", action="store_true", help="keep the working directory (logs, --out files)")
    a = ap.parse_args()
    for p in (HARNESS, FAKE):
        if not p.is_file():
            raise SystemExit(f"missing {p}")
    work = Path(tempfile.mkdtemp(prefix="vision-check-smoke-"))
    cfg, _ = config(work, a.template, a.tokenizer)
    old = wrapper(work / "fake-old", "--fake-old")
    expect = work / "expect-from.json"       # (a copy of the text-ab case's --out)
    failed = []
    t_all = time.time()
    todo = [c for c in CASES if not a.only or c[0] in a.only]
    if any("@EXPECT" in c[1] for c in todo) and not any(c[0] == "text-ab" for c in todo):
        todo = [c for c in CASES if c[0] == "text-ab"] + todo     # (--expect-from needs text-ab's --out)
    for name, args, fenv, want_exit, want in todo:
        out = work / f"{name}.json"
        server = FakeServer(drift="@URL-DRIFT" in args) if any(x.startswith("@URL") for x in args) else None
        sub = {"@OLD": old, "@URL": server.url if server else "", "@URL-DRIFT": server.url if server else "",
               "@EXPECT": str(expect)}
        cmd = [sys.executable, str(HARNESS), "--exe", str(work / "fake-new"), "--config", str(cfg), *SMALL,
               "--logs", str(work / f"logs-{name}"), "--out", str(out)] + [sub.get(x, x) for x in args]
        env = {k: v for k, v in os.environ.items() if not k.startswith("VISION_CHECK_FAKE_")}
        env.pop("STRATA_MROPE_SLOTS", None)
        env.pop("STRATA_MROPE_REGION_SKIP", None)
        env.update(fenv)
        t0 = time.time()
        try:
            p = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=900)
        finally:
            if server:
                server.close()
        problems = []
        if p.returncode != want_exit:
            problems.append(f"exit {p.returncode}, expected {want_exit}")
        j = {}
        if want_exit != 64:
            try:
                j = json.loads(out.read_text())
            except (OSError, ValueError) as x:
                problems.append(f"no --out JSON: {x}")
        elif out.exists():
            problems.append("a usage error wrote --out")
        crit = j.get("criteria") or []
        for key, status in want.items():
            got = [c["status"] for c in crit if key in c["criterion"] and c["status"] != "info"]
            if not got or any(s != status for s in got):
                problems.append(f"criterion '{key}': {got or 'missing'}, expected {status}")
        if want_exit == 0 and any(c["status"] not in ("GO", "info") for c in crit):
            problems.append("not every criterion GO: " + "; ".join(f"{c['status']} {c['criterion']}" for c in crit
                                                                   if c["status"] not in ("GO", "info")))
        if j:
            structure(name, j, problems)
        if name == "text-ab" and out.exists():
            shutil.copyfile(out, expect)
        dt = time.time() - t0
        print(f"{'PASS' if not problems else 'FAIL'} {name}: exit {p.returncode}, verdict {j.get('verdict')}, "
              f"{len(j.get('checks') or [])} rows, {dt:.1f} s", flush=True)
        if problems:
            failed.append(name)
            for x in problems:
                print(f"     - {x}", flush=True)
            tail = (p.stdout + p.stderr).splitlines()[-40:]
            print("     harness output (last lines):\n       " + "\n       ".join(tail), flush=True)
    print(f"{'ALL PASS' if not failed else 'FAILED: ' + ', '.join(failed)} in {time.time() - t_all:.0f} s"
          f"{f'; files in {work}' if a.keep or failed else ''}", flush=True)
    if not (a.keep or failed):
        shutil.rmtree(work, ignore_errors=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
