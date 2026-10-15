"""
Queued LLM scoring. Run this when the machine is free (see NOTE below).

Emits one JSONL line per item with the label-token score

    s = log P("1") - log P("0")

taken from the top-logprobs of a single forced output token. That score, not
the token the model emitted, is what the decision-rule analysis needs: the
token is argmax(s > 0), i.e. one arbitrary operating point on s.

Resumable: re-running skips ids already present in the output file.

NOTE on this machine. Measured 2026-10-13: swap was 100% full, ~850 MB RAM
free against a 3.5 GB model, load average 27 from an unrelated dev stack
(Cypress, ng serve, 2 JVMs, mysqld). One 900-token prompt did not return in
120 s. Check `free -m` shows >5 GB available and `uptime` load < 4 before
launching, and set the CPU governor to performance first (needs sudo):
    sudo cpupower frequency-set -g performance
FINDINGS.md in ../llm_prior_shrinkage records a 2.7x speedup from that alone.

Usage
    python3 -P -u score_llm.py --task bensparx --model qwen2.5:3b-instruct-q8_0 \
        --prompt list --limit 0 --out ../results/llm_bensparx_qwen_list.jsonl
"""
import argparse, csv, json, os, sys, time, urllib.request, collections

URL = "http://localhost:11434/api/chat"
BENSPARX = "/mnt/extra-storage/data/bensparx/BenSParX.csv"
CKD = "/mnt/extra-storage/data/diabetic_ckd/DiabeticCKD_dataset.csv"


# ------------------------------------------------------------------ prompting
# Several serialisations, because LLM4CKD (arXiv:2609.04013) names prompt
# design, feature serialisation and model version as the things its own
# reliability depends on. Each (model, prompt) pair is a separate cell.
def serialise(pairs, style):
    if style == "list":
        return ", ".join(f"{k}={v}" for k, v in pairs)
    if style == "lines":
        return "\n".join(f"- {k}: {v}" for k, v in pairs)
    if style == "sentence":
        return ". ".join(f"The {k} is {v}" for k, v in pairs) + "."
    raise ValueError(style)


PROMPTS = {
    "bensparx": ("You are a clinical speech analyst. Decide whether the speaker "
                 "has Parkinson's disease from these acoustic features of a "
                 "conversational recording.\n{body}\n"
                 "Reply with exactly one character: 1 for Parkinson's disease, "
                 "0 for healthy control."),
    "ckd": ("You are a nephrology screening assistant. Decide whether this "
            "diabetic patient has chronic kidney disease from these demographic "
            "and lifestyle factors.\n{body}\n"
            "Reply with exactly one character: 1 for CKD, 0 for no CKD."),
}


def ask(model, msg, retries=3):
    body = json.dumps({"model": model,
                       "messages": [{"role": "user", "content": msg}],
                       "stream": False,
                       "options": {"temperature": 0, "num_predict": 1},
                       "logprobs": True, "top_logprobs": 20}).encode()
    for a in range(retries):
        try:
            req = urllib.request.Request(URL, body, {"Content-Type": "application/json"})
            return json.load(urllib.request.urlopen(req, timeout=1800))
        except Exception as e:
            if a == retries - 1:
                raise
            time.sleep(5 * (a + 1))


def score_from(d):
    """s = logP("1") - logP("0") from the top-logprobs of the single token.
    Tokens absent from the top-k get a floor, recorded so it is auditable."""
    tl = {x["token"]: x["logprob"] for x in d["logprobs"][0]["top_logprobs"]}
    floor = min(tl.values()) - 2.0 if tl else -30.0
    l1 = tl.get("1", tl.get("１", floor))
    l0 = tl.get("0", tl.get("０", floor))
    return l1 - l0, {"emitted": d["message"]["content"], "lp1": l1, "lp0": l0,
                     "both_in_topk": ("1" in tl and "0" in tl)}


# -------------------------------------------------------------------- loaders
def load_bensparx():
    """900 segments, 120 subjects (60 PD / 60 HC), 71 features. Subject id is
    the segment id minus its trailing _N, so predictions can be aggregated to
    subject level the way arXiv:2605.24806 does."""
    rows = list(csv.reader(open(BENSPARX)))
    hdr, data = rows[0], rows[1:]
    feats = hdr[1:-1]
    out = []
    for r in data:
        out.append({"id": r[0], "group": r[0].rsplit("_", 1)[0],
                    "y": int(r[-1]), "pairs": list(zip(feats, r[1:-1]))})
    return out


def load_ckd():
    """4000 records = 400 patients x 10 yearly visits. The rows are NOT
    independent, so `group` is the patient id and any split must honour it.
    Record-level CKD prevalence is 9.65%; 185/400 patients are ever positive."""
    rows = list(csv.reader(open(CKD)))
    hdr, data = rows[0], rows[1:]
    feats = hdr[1:-1]
    out = []
    for i, r in enumerate(data):
        out.append({"id": f"{r[0]}_v{i % 10}", "group": r[0],
                    "y": int(r[-1]), "pairs": list(zip(feats, r[1:-1]))})
    return out


LOADERS = {"bensparx": load_bensparx, "ckd": load_ckd}


def subsample(items, limit=0, groups=0, seed=0):
    """Grouped subsample. `groups` takes that many whole subjects/patients,
    balanced across the label so the cohort keeps the source's 50/50 split;
    `limit` instead caps the item count. Neither ever splits a group, because
    the segments of one subject are not independent observations."""
    import random
    rng = random.Random(seed)
    by = collections.defaultdict(list)
    for it in items:
        by[it["group"]].append(it)
    if groups:
        # A group's label is "positive if ANY record in it is positive".
        # Using the FIRST record instead is wrong for longitudinal data: in the
        # CKD cohort onset is almost always at the last visit, so 0 of 400
        # patients are positive on visit 1 and the balancing silently became a
        # no-op. For BenSParX every segment of a subject shares one label, so
        # `any` and `first` agree there and this change is inert.
        pos = sorted(g for g, v in by.items() if any(r["y"] == 1 for r in v))
        neg = sorted(g for g, v in by.items() if not any(r["y"] == 1 for r in v))
        rng.shuffle(pos); rng.shuffle(neg)
        half = groups // 2
        keep = sorted(pos[:half] + neg[:groups - half])
        return [it for g in keep for it in by[g]]
    if not limit or limit >= len(items):
        return items
    gs = sorted(by); rng.shuffle(gs)
    out = []
    for g in gs:
        if len(out) >= limit:
            break
        out.extend(by[g])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=sorted(LOADERS))
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt", default="list", choices=("list", "lines", "sentence"))
    ap.add_argument("--limit", type=int, default=0, help="0 = all items")
    ap.add_argument("--groups", type=int, default=0,
                    help="take this many whole subjects/patients, label-balanced")
    ap.add_argument("--nfeat", type=int, default=0,
                    help="use only the first N features (0 = all). Paper 3 names its "
                         "fixed feature pipeline as a limitation, so this is a factor.")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    items = subsample(LOADERS[a.task](), limit=a.limit, groups=a.groups)
    if a.nfeat:
        for it in items:
            it["pairs"] = it["pairs"][:a.nfeat]
    done = set()
    if os.path.exists(a.out):
        for line in open(a.out):
            try:
                done.add(json.loads(line)["id"])
            except Exception:
                pass
    todo = [it for it in items if it["id"] not in done]
    print(f"{a.task}/{a.model}/{a.prompt}: {len(items)} items, "
          f"{len(done)} already done, {len(todo)} to go", flush=True)

    t0 = time.time()
    with open(a.out, "a") as fh:
        for n, it in enumerate(todo, 1):
            msg = PROMPTS[a.task].format(body=serialise(it["pairs"], a.prompt))
            d = ask(a.model, msg)
            s, meta = score_from(d)
            fh.write(json.dumps({"id": it["id"], "group": it["group"],
                                 "y": it["y"], "s": s, "model": a.model,
                                 "prompt": a.prompt, **meta}) + "\n")
            fh.flush()
            if n % 10 == 0 or n == len(todo):
                el = time.time() - t0
                print(f"  {n}/{len(todo)}  {el/n:.1f}s/item  "
                      f"eta {(len(todo)-n)*el/n/60:.0f}min", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
