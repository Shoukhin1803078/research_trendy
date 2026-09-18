#!/usr/bin/env python3
"""
Full-scale, no-LLM ablation: M1-only vs. M1+M2, single top-1 decision per query,
evaluated against the OAEI Bio-ML gold equivalence+subsumption references on the
COMPLETE query pool of both public tasks (no sampling -- this needs no LLM calls,
so there is no cost reason to bound it).

Decision policies:
  M1      : the top BM25-ranked target class (the "anchor").
  M1+M2   : same anchor UNLESS M2 finds it disjoint from itself (impossible) or finds a
            union-generator candidate that M2 asserts is a real equivalentClass/subclass/
            superclass match the anchor itself is NOT already; in practice, since M2 only
            ever *removes* disjoint candidates from contention (Prop. 1) and never promotes
            a different candidate to the decision without M3's judgment, the only way this
            row can differ from the M1 row is if the anchor itself is typed disjoint from
            the record it is retrieved for (impossible, since M2 relations are computed
            against the anchor, never used to veto the anchor) -- so this ablation also
            reports, honestly, how often M2 had any opportunity to change anything at all.
"""
import sys, json, time, random, collections
sys.path.insert(0, __file__.rsplit('/', 1)[0])
from full_pipeline import parse_onto, ancestor_closure, BM25, m2_relation, read_tsv_pairs, short

random.seed(20260918)
TOPK = 10

def evaluate_task(name, src_owl, tgt_owl, equiv_tsv, subs_tsv):
    t0 = time.time()
    print(f"=== {name} ===", flush=True)
    S = parse_onto(src_owl); T = parse_onto(tgt_owl)
    print(f"  src={len(S)} tgt={len(T)} ({time.time()-t0:.0f}s)", flush=True)
    T_anc = ancestor_closure(T)
    docs = [(uri, lb) for uri, d in T.items() for lb in d['labels'] if lb]
    idx = BM25(docs)
    children = collections.defaultdict(set)
    for u, d in T.items():
        for p in d['parents']:
            children[p].add(u)

    def cls_rank(text, topk):
        hits = idx.query(text, topk*4)
        best = {}
        for i, sc in hits:
            u = idx.keys[i]
            if sc > best.get(u, -1): best[u] = sc
        return sorted(best.items(), key=lambda x: -x[1])[:topk]

    gold_e = read_tsv_pairs(equiv_tsv)
    gold_s = read_tsv_pairs(subs_tsv) if subs_tsv else {}
    # Query pool and correctness are scored against the EQUIVALENCE reference only, exactly
    # matching Table VI / the OAEI Bio-ML unsupervised-equivalence metric other systems in
    # Table II are scored against -- this is what makes the P/R/F1 below directly comparable
    # to LogMap/BERTMap/MILA/etc. Subsumption gold (gold_s) is used only as an extra "don't
    # count this as wrongly dropped" signal for M2's safety-check diagnostics, never for the
    # headline P/R/F1, since Table II's numbers are not scored against it either.
    queries = sorted(s for s in gold_e if s in S and any(t in T for t in gold_e[s]))
    print(f"  {len(queries)} queries with equivalence gold in this ontology pair", flush=True)

    n_disjoint_would_veto_anchor = 0
    n_m2_had_opportunity = 0  # candidate other than anchor was itself gold AND anchor wasn't
    m1_correct = 0
    m1_m2_correct = 0
    m2_disjoint_drops = 0
    m2_total_candidates = 0

    def label_of(src):
        return ' . '.join(S[src]['labels']) or short(src)

    for n, s in enumerate(queries):
        if n % 500 == 0 and n:
            print(f"    {n}/{len(queries)} ({time.time()-t0:.0f}s)", flush=True)
        g_e = {t for t in gold_e.get(s, set()) if t in T}
        g_s = {t for t in gold_s.get(s, set()) if t in T}
        g_any = g_e | g_s
        lex = [u for u, _ in cls_rank(label_of(s), TOPK)]
        if not lex:
            continue
        anchor = lex[0]
        grp = set([anchor]) | T[anchor]['parents'] | children[anchor]
        grp = [u for u in grp if u in T]
        seen = []; sset = set()
        for u in lex+grp:
            if u not in sset:
                sset.add(u); seen.append(u)
        others = [c for c in seen if c != anchor][:TOPK]

        anchor_is_gold = anchor in g_e  # equivalence-only correctness, matching Table II/VI
        anchor_is_gold_any = anchor in g_any  # broader safety-net check for M2 diagnostics
        if anchor_is_gold:
            m1_correct += 1

        # M2 pass: type each "other" candidate; drop any typed disjoint from contention.
        survivors = []
        for c in others:
            rel, ev = m2_relation(anchor, c, T_anc, T)
            m2_total_candidates += 1
            if rel == 'disjoint':
                m2_disjoint_drops += 1
                continue
            survivors.append((c, rel))

        if any(c in g_any for c, _ in survivors) and not anchor_is_gold_any:
            n_m2_had_opportunity += 1

        # M1+M2 decision policy: M2 alone has no adjudication model, so without M3 it can
        # only ever *remove* candidates, never re-rank or promote one over the anchor; the
        # decision therefore stays the anchor unless the anchor itself was typed disjoint
        # against every other structural neighbor found for it (impossible by construction,
        # since disjointness is only ever computed relative to the anchor, never against it).
        m1_m2_decision = anchor
        if m1_m2_decision in g_e:
            m1_m2_correct += 1

    n = len(queries)
    return {
        'task': name, 'n_queries': n,
        'm1_precision': m1_correct/n, 'm1_recall': m1_correct/n, 'm1_f1': m1_correct/n,
        'm1_m2_precision': m1_m2_correct/n, 'm1_m2_recall': m1_m2_correct/n, 'm1_m2_f1': m1_m2_correct/n,
        'm2_candidates_typed': m2_total_candidates,
        'm2_disjoint_drops': m2_disjoint_drops,
        'm2_disjoint_drop_rate': m2_disjoint_drops/m2_total_candidates if m2_total_candidates else 0.0,
        'm2_had_opportunity_to_help': n_m2_had_opportunity,
        'wall_clock_sec': time.time()-t0,
    }

if __name__ == '__main__':
    base = __file__.rsplit('/', 1)[0] + '/bioml'
    tasks = [
        ('ncit-doid', f'{base}/ncit-doid/ncit-doid/ncit.owl', f'{base}/ncit-doid/ncit-doid/doid.owl',
         f'{base}/ncit-doid/ncit-doid/refs_equiv/test.tsv', f'{base}/ncit-doid/ncit-doid/refs_subs/train.tsv'),
        ('omim-ordo', f'{base}/omim-ordo/omim-ordo/omim.owl', f'{base}/omim-ordo/omim-ordo/ordo.owl',
         f'{base}/omim-ordo/omim-ordo/refs_equiv/test.tsv', f'{base}/omim-ordo/omim-ordo/refs_subs/train.tsv'),
    ]
    out = {}
    for name, src, tgt, eq, sub in tasks:
        r = evaluate_task(name, src, tgt, eq, sub)
        out[name] = r
        print(json.dumps(r, indent=1), flush=True)
    outpath = __file__.rsplit('/', 1)[0] + '/ablation_m1_m2_results.json'
    json.dump(out, open(outpath, 'w'), indent=1)
    print(f"dumped {outpath}")
