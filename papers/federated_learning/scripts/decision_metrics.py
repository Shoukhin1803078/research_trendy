#!/usr/bin/env python3
"""
Turns the per-candidate raw records from full_pipeline.py into a single top-1 decision per
query for the M1+M2+M3 system, so it can be compared on equal (P/R/F1, one-decision-per-
query) footing with M1, M1+M2, and the literature systems in Table II.

Decision policy: keep the M1 anchor (the top BM25 guess) as the default answer, UNLESS a
different (non-anchor) surviving candidate is labeled "equivalent" by M3, in which case take
the one with the highest M3 confidence among those. This directly operationalizes "M3 only
overrides M1 when it is confident there is a better match" -- the natural role of an
adjudicator layered on top of a retriever.
"""
import json, collections, sys

def load(path):
    recs = collections.defaultdict(list)
    anchor_gold = {}
    for line in open(path):
        r = json.loads(line)
        recs[r['src']].append(r)
        anchor_gold[r['src']] = r['anchor_is_gold']
    return recs, anchor_gold

def decide(task_name):
    path = f'full_pilot_raw_{task_name}.jsonl'
    by_query, anchor_gold = load(path)
    n = len(by_query)
    m1_correct = 0
    m1m2m3_correct = 0
    overrides = 0
    override_correct = 0
    for src, cands in by_query.items():
        a_gold = anchor_gold[src]
        if a_gold:
            m1_correct += 1
        equiv_cands = [c for c in cands if c['m3_relation'] == 'equivalent']
        decision_is_anchor = True
        if equiv_cands:
            best = max(equiv_cands, key=lambda c: c['m3_confidence'])
            decision_is_anchor = False
            overrides += 1
            if best['gold_any']:
                override_correct += 1
                m1m2m3_correct += 1
        if decision_is_anchor and a_gold:
            m1m2m3_correct += 1
    return {
        'task': task_name, 'n_queries': n,
        'm1_correct_of_n': m1_correct,
        'm1_precision': m1_correct/n, 'm1_recall': m1_correct/n, 'm1_f1': m1_correct/n,
        'm1m2m3_correct_of_n': m1m2m3_correct,
        'm1m2m3_precision': m1m2m3_correct/n, 'm1m2m3_recall': m1m2m3_correct/n, 'm1m2m3_f1': m1m2m3_correct/n,
        'n_overrides': overrides, 'n_override_correct': override_correct,
        'override_precision': override_correct/overrides if overrides else None,
    }

if __name__ == '__main__':
    out = {}
    for t in ('ncit-doid', 'omim-ordo'):
        r = decide(t)
        out[t] = r
        print(json.dumps(r, indent=1))
    json.dump(out, open('decision_metrics_results.json', 'w'), indent=1)
