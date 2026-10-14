"""Label-geometry elicitation from a local LLM — all four prompt designs in one place.

Replaces test_geometry_elicitation.py / elicit_geometry_v2.py / elicit_geometry_v3.py /
elicit_geometry_final.py, which were four near-duplicate files for one question.

    python elicit_geometry.py <variant> [model ...]
    variants: v1_simple  v2_complex  v3_decomposed  v4_order_only

Findings (see PHASE0_RESULTS.md Phase 0.8) — llama3.2:3b:
  v1  3/3 scales ordered, 0/3 controls rejected  (says everything is ordinal)
  v2  1/3 scales,         3/3 controls           (collapses to 'anatomical location')
  v3  3/3 scales,         0/3 controls           (collapses to 'Clinical Severity')
  v4  3/3 modal orderings correct, tau=1.0       (single calls shuffle-sensitive)
Conclusion: reliable sorter, useless detector -> take detection from dataset metadata,
use the LLM only for ordering, under a 5-shuffle majority vote.
"""
import json, subprocess, sys, time, re, random
from collections import Counter

OLLAMA = "http://127.0.0.1:11434/api/generate"

SCHEMAS = {
    "ordinal_flag": {"type": "object", "properties": {
        "is_ordinal": {"type": "boolean"},
        "ordered_classes": {"type": "array", "items": {"type": "string"}}},
        "required": ["is_ordinal", "ordered_classes"]},
    "axis_full": {"type": "object", "properties": {
        "axis_name": {"type": "string"}, "axis_is_severity": {"type": "boolean"},
        "ordered_classes": {"type": "array", "items": {"type": "string"}},
        "justification": {"type": "string"}},
        "required": ["axis_name", "axis_is_severity", "ordered_classes", "justification"]},
    "axis_only": {"type": "object", "properties": {
        "axis_name": {"type": "string"}, "is_severity_scale": {"type": "boolean"}},
        "required": ["axis_name", "is_severity_scale"]},
    "order_only": {"type": "object", "properties": {
        "ordered_classes": {"type": "array", "items": {"type": "string"}}},
        "required": ["ordered_classes"]},
}

SEVERITY_TERMS = {"severity","severe","grade","grading","stage","staging","progression",
                  "malignancy","malignant","aggressiveness","aggressive","risk","suspicion",
                  "differentiation","dysplasia","invasiveness","extent","advancement","degree",
                  "assessment","category","score"}
ANATOMY_TERMS = {"organ","anatomy","anatomical","location","site","position","region",
                 "structure","type","kind"}

def gate(axis):
    """Code-side severity-axis gate: WE decide, not the model's own boolean."""
    t = set(re.findall(r"[a-z]+", axis.lower()))
    if t & ANATOMY_TERMS and not (t & SEVERITY_TERMS):
        return False
    return bool(t & SEVERITY_TERMS)

P = {
 "v1_simple": """You are given the class labels of a medical image classification task.
Labels (in arbitrary order): {labels}

Decide whether these labels form an ORDINAL scale (a clinically ordered severity/grading
scale). If ordinal, return them sorted from least to most severe. If they are unordered
categories, set is_ordinal to false and return the labels unchanged.
Answer with JSON only.""",

 "v2_complex": """You are given the class labels of a medical image classification task.

Labels (arbitrary order): {labels}

Some label sets lie on a single underlying clinical SEVERITY or PROGRESSION axis (for
example a cancer grading scale, or a disease staging scale). Others are simply unordered
categories (for example different anatomical organs, or unrelated diseases).

1. axis_name: name the single underlying axis these labels vary along. If they do not
   share one ordered axis, name what distinguishes them instead.
2. axis_is_severity: true only if that axis is a clinical severity/progression/grading axis.
3. ordered_classes: if axis_is_severity, sort from least to most severe. Else unchanged.
4. justification: one short sentence.
Answer with JSON only.""",

 "v3_detect": """Medical image classification labels: {labels}

Question: do these labels form a clinical SEVERITY or GRADING scale (one ordered axis from
least to most severe), or are they unordered categories?

Set is_severity_scale true only for an ordered clinical severity/grading/staging scale.
Set it false for unordered categories such as different organs, tissue types, or distinct
unrelated diagnoses. Set axis_name to a short name for what distinguishes the labels.
JSON only.""",

 "sort": """Medical image classification labels: {labels}

These labels form a clinical severity scale. Return them sorted from LEAST severe to MOST
severe. Include every label exactly once.
JSON only.""",
}

CASES = [
    ("gleason", ["Gleason grade 5","benign","Gleason grade 3","Gleason grade 4"], True,
     ["benign","Gleason grade 3","Gleason grade 4","Gleason grade 5"]),
    ("dr_icdr", ["severe NPDR","no DR","proliferative DR","mild NPDR","moderate NPDR"], True,
     ["no DR","mild NPDR","moderate NPDR","severe NPDR","proliferative DR"]),
    ("birads", ["BI-RADS 4","BI-RADS 2","BI-RADS 5","BI-RADS 1","BI-RADS 3"], True,
     ["BI-RADS 1","BI-RADS 2","BI-RADS 3","BI-RADS 4","BI-RADS 5"]),
    ("ctrl_organs", ["liver","spleen","left kidney","pancreas"], False, None),
    ("ctrl_skin_dx", ["melanoma","basal cell carcinoma","seborrheic keratosis","nevus"], False, None),
    ("ctrl_tissue", ["adipose","lymphocytes","mucus","smooth muscle","debris"], False, None),
]

def ask(model, prompt, schema, npred=400):
    pl = {"model": model, "prompt": prompt, "stream": False, "format": schema,
          "options": {"temperature": 0, "num_predict": npred}}
    t0 = time.time()
    p = subprocess.run(["curl","-s",OLLAMA,"-d",json.dumps(pl)],
                       capture_output=True, text=True, timeout=900)
    dt = time.time() - t0
    try:
        return json.loads(json.loads(p.stdout)["response"]), dt
    except Exception as e:
        return {"__error__": str(e)}, dt

def run(variant, model):
    print(f"\n{'='*74}\nMODEL {model}   VARIANT {variant}\n{'='*74}")
    res, npass = {}, 0
    for name, labels, exp_ord, exp_order in CASES:
        if variant == "v4_order_only":
            if not exp_ord:          # v4 does no detection by design; metadata supplies it
                print(f"  [skip] {name:<13} (detection comes from metadata, not the LLM)")
                continue
            rng = random.Random(0); answers = []; secs = 0.0
            for _ in range(5):
                sh = exp_order[:]; rng.shuffle(sh)
                o, dt = ask(model, P["sort"].format(labels=sh), SCHEMAS["order_only"], 300)
                secs += dt; answers.append(tuple(o.get("ordered_classes") or []) or None)
            modal, nmod = Counter(answers).most_common(1)[0]
            ok = modal == tuple(exp_order)
            print(f"  [{'PASS' if ok else 'FAIL'}] {name:<13} {secs:5.1f}s "
                  f"exact {sum(a==tuple(exp_order) for a in answers)}/5  stability {nmod}/5")
            res[name] = {"pass": bool(ok), "modal": list(modal) if modal else None,
                         "stability": nmod, "secs": round(secs,1)}
            npass += ok
            continue

        if variant == "v1_simple":
            o, dt = ask(model, P["v1_simple"].format(labels=labels), SCHEMAS["ordinal_flag"])
            said = bool(o.get("is_ordinal")); order = o.get("ordered_classes"); axis = ""
        elif variant == "v2_complex":
            o, dt = ask(model, P["v2_complex"].format(labels=labels), SCHEMAS["axis_full"], 500)
            axis = o.get("axis_name",""); said = bool(o.get("axis_is_severity")) and gate(axis)
            order = o.get("ordered_classes")
        elif variant == "v3_decomposed":
            a, dt = ask(model, P["v3_detect"].format(labels=labels), SCHEMAS["axis_only"], 300)
            axis = a.get("axis_name",""); said = bool(a.get("is_severity_scale")) and gate(axis)
            order = None
            if said:
                b, dt2 = ask(model, P["sort"].format(labels=labels), SCHEMAS["order_only"], 300)
                dt += dt2; order = b.get("ordered_classes")
        else:
            raise SystemExit(f"unknown variant {variant}")

        ok = (said and order == exp_order) if exp_ord else (not said)
        npass += ok
        print(f"  [{'PASS' if ok else 'FAIL'}] {name:<13} {dt:5.1f}s "
              f"said_ordinal={said} axis={axis!r}")
        if exp_ord and order != exp_order:
            print(f"          got     : {order}\n          expected: {exp_order}")
        res[name] = {"pass": bool(ok), "said_ordinal": said, "axis_name": axis,
                     "ordered_classes": order, "secs": round(dt,1)}
    n = len([c for c in CASES if variant != "v4_order_only" or c[2]])
    print(f"  --> {npass}/{n} passed")
    res["_score"] = f"{npass}/{n}"
    return res

if __name__ == "__main__":
    variant = sys.argv[1] if len(sys.argv) > 1 else "v4_order_only"
    models = sys.argv[2:] or ["llama3.2:3b"]
    out = {m: run(variant, m) for m in models}
    path = f"phase0/geometry_{variant}.json"
    json.dump(out, open(path, "w"), indent=2); print(f"\nwrote {path}")
