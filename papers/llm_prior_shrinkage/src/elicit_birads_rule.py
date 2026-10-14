"""Does the local LLM know the BI-RADS lexicon's malignancy associations?

Validated NON-CIRCULARLY: we elicit P(malignant | descriptor) from the LLM using text only,
then compare against the EMPIRICAL rate computed from the training split of CBIS-DDSM.
The LLM never sees the data. If its textbook knowledge tracks the empirical rates, the
concept rule is independently supported rather than asserted.

Only the TRAIN split is used for the empirical side, so nothing here touches test labels.
"""
import json, os, subprocess, sys, time
import numpy as np, yaml
from scipy.stats import spearmanr, pearsonr

D = "/mnt/extra-storage/data/medimeta_extracted"
OLLAMA = "http://127.0.0.1:11434/api/generate"
SCHEMA = {"type": "object",
          "properties": {"probability_malignant": {"type": "number"}},
          "required": ["probability_malignant"]}

PROMPT = """In the BI-RADS lexicon for mammography, a breast {kind} may be described as
"{desc}".

Based on standard radiology knowledge, what is the approximate probability that a mass with
this {kind} is MALIGNANT (cancerous)? Give a number between 0 and 1.

For calibration: a circumscribed margin is strongly associated with benign disease
(low probability), while a spiculated margin is strongly associated with malignancy
(high probability).

Answer with JSON only: {{"probability_malignant": <number>}}"""


def ask(model, desc, kind, n_vote=3):
    vals = []
    for i in range(n_vote):
        pl = {"model": model, "prompt": PROMPT.format(desc=desc, kind=kind),
              "stream": False, "format": SCHEMA,
              "options": {"temperature": 0.0 if i == 0 else 0.3, "num_predict": 60, "seed": i}}
        p = subprocess.run(["curl", "-s", OLLAMA, "-d", json.dumps(pl)],
                           capture_output=True, text=True, timeout=600)
        try:
            v = float(json.loads(json.loads(p.stdout)["response"])["probability_malignant"])
            if 0.0 <= v <= 1.0:
                vals.append(v)
        except Exception:
            pass
    return (float(np.median(vals)), len(vals)) if vals else (float("nan"), 0)


def empirical(ds, concept_task):
    """P(malignant | descriptor present) from the TRAIN split only."""
    sp = np.array([int(os.path.splitext(os.path.basename(l.strip()))[0])
                   for l in open(f"{D}/{ds}/splits/train.txt") if l.strip()])
    path = np.load(f"{D}/{ds}/task_labels/pathology.npy").astype(int).ravel()
    conc = np.load(f"{D}/{ds}/task_labels/{concept_task}.npy")
    if conc.ndim == 1:
        conc = np.eye(int(conc.max()) + 1)[conc.astype(int)]
    conc = conc[sp].astype(bool); mal = path[sp].astype(bool)
    out = {}
    for j in range(conc.shape[1]):
        m = conc[:, j]
        out[j] = (float(mal[m].mean()), int(m.sum())) if m.sum() >= 5 else (float("nan"), int(m.sum()))
    return out


def main(model="llama3.2:3b"):
    res = {}
    for ds, concept_task, kind in [("mammo_mass", "mass margins", "margin"),
                                   ("mammo_mass", "mass shape", "shape"),
                                   ("mammo_calc", "calc type", "calcification type"),
                                   ("mammo_calc", "calc distribution", "calcification distribution")]:
        info = yaml.safe_load(open(f"{D}/{ds}/info.yaml"))
        labels = next(t["labels"] for t in info["tasks"] if t["task_name"] == concept_task)
        emp = empirical(ds, concept_task)
        print(f"\n=== {ds} / {concept_task} ===")
        print(f"{'descriptor':<28} {'LLM P(mal)':>11} {'empirical':>10} {'n_train':>8} {'votes':>6}")
        print("-" * 68)
        rows = []
        for j in sorted(labels, key=int):
            desc = labels[j].replace("_", " ").lower()
            llm, nv = ask(model, desc, kind)
            e, n = emp.get(int(j), (float("nan"), 0))
            print(f"{labels[j][:27]:<28} {llm:>11.3f} "
                  f"{(f'{e:.3f}' if e==e else 'n/a'):>10} {n:>8} {nv:>6}")
            rows.append({"descriptor": labels[j], "llm": llm, "empirical": e, "n_train": n})
        ok = [(r["llm"], r["empirical"]) for r in rows
              if r["empirical"] == r["empirical"] and r["llm"] == r["llm"]]
        if len(ok) >= 3:
            a = np.array([x[0] for x in ok]); b = np.array([x[1] for x in ok])
            rho = spearmanr(a, b).statistic; pr = pearsonr(a, b)
            print(f"  LLM vs empirical over {len(ok)} descriptors: "
                  f"Spearman={rho:+.3f}  Pearson={pr.statistic:+.3f} (p={pr.pvalue:.3f})")
            res[f"{ds}/{concept_task}"] = {"rows": rows, "spearman": float(rho),
                                           "pearson": float(pr.statistic),
                                           "pearson_p": float(pr.pvalue), "n": len(ok)}
        else:
            res[f"{ds}/{concept_task}"] = {"rows": rows, "note": "too few usable descriptors"}
    os.makedirs("results", exist_ok=True)
    json.dump(res, open("results/birads_rule_elicitation.json", "w"), indent=2)
    print("\nwrote results/birads_rule_elicitation.json")
    print("\nIf Spearman is high, the LLM's textbook lexicon knowledge tracks the empirical")
    print("malignancy rates it never saw. That independently supports the concept rule.")
    print("If it is near zero, the LLM adds nothing here and the concept bottleneck should")
    print("use empirical train-split rates instead - which we would report plainly.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "llama3.2:3b")
