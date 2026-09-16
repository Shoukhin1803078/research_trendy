#!/usr/bin/env python3
"""M1/M2 evaluation on OAEI Bio-ML. Retrieval-only, no LLM."""
import re, math, sys, time, json, collections
import xml.etree.ElementTree as ET

RDF='{http://www.w3.org/1999/02/22-rdf-syntax-ns#}'
RDFS='{http://www.w3.org/2000/01/rdf-schema#}'
OWL='{http://www.w3.org/2002/07/owl#}'
OBO='{http://www.geneontology.org/formats/oboInOwl#}'
SYN = {OBO+'hasExactSynonym', OBO+'hasRelatedSynonym', OBO+'hasNarrowSynonym', OBO+'hasBroadSynonym'}

def parse_onto(path):
    cls={}
    ctx=ET.iterparse(path, events=('start','end'))
    _, root = next(ctx)
    for ev, el in ctx:
        if ev=='end' and el.tag==OWL+'Class':
            uri=el.get(RDF+'about')
            if uri:
                labels=[]; parents=set()
                for ch in el:
                    if ch.tag==RDFS+'label' and ch.text:
                        labels.append(ch.text.strip())
                    elif ch.tag in SYN and ch.text:
                        labels.append(ch.text.strip())
                    elif ch.tag==RDFS+'subClassOf':
                        r=ch.get(RDF+'resource')
                        if r: parents.add(r)
                cls[uri]={'labels':labels,'parents':parents}
            el.clear(); root.clear()
    return cls

TOK=re.compile(r'[a-z0-9]+')
def tok(s): return TOK.findall(s.lower())

class BM25:
    def __init__(self, docs, k1=1.2, b=0.75):
        # docs: list of (key, text)
        self.k1, self.b = k1, b
        self.keys=[]; self.post=collections.defaultdict(list); self.dl=[]
        for i,(k,t) in enumerate(docs):
            toks=tok(t); self.keys.append(k)
            tf=collections.Counter(toks); self.dl.append(len(toks))
            for term,f in tf.items(): self.post[term].append((i,f))
        self.N=len(self.keys); self.avgdl=(sum(self.dl)/self.N) if self.N else 1.0
        self.idf={t: math.log(1+(self.N-len(p)+0.5)/(len(p)+0.5)) for t,p in self.post.items()}
    def query(self, text, topk):
        scores=collections.defaultdict(float)
        for term in set(tok(text)):
            if term not in self.post: continue
            idf=self.idf[term]
            for i,tf in self.post[term]:
                dl=self.dl[i]
                scores[i]+= idf*tf*(self.k1+1)/(tf+self.k1*(1-self.b+self.b*dl/self.avgdl))
        return sorted(scores.items(), key=lambda x:-x[1])[:topk]

def read_tsv(p):
    m=collections.defaultdict(set)
    with open(p, encoding='utf-8') as f:
        next(f)
        for ln in f:
            parts=ln.rstrip('\n').split('\t')
            if len(parts)>=2 and parts[0] and parts[1]:
                m[parts[0]].add(parts[1])
    return m

def evaluate(name, src_owl, tgt_owl, ref_tsv, ks=(1,3,5,10), restrict_to_ref=True):
    t0=time.time()
    print(f"\n=== {name} ===  parsing ontologies...", flush=True)
    S=parse_onto(src_owl); print(f"  source classes: {len(S)}  ({time.time()-t0:.0f}s)", flush=True)
    T=parse_onto(tgt_owl); print(f"  target classes: {len(T)}  ({time.time()-t0:.0f}s)", flush=True)

    # target label index (per-label docs)
    docs=[]; label2cls=[]
    for uri,d in T.items():
        for lb in d['labels']:
            if lb: docs.append((uri,lb))
    print(f"  target label docs: {len(docs)}", flush=True)
    idx=BM25(docs); print(f"  index built ({time.time()-t0:.0f}s)", flush=True)

    children=collections.defaultdict(set)
    for u,d in T.items():
        for p in d['parents']: children[p].add(u)

    gold=read_tsv(ref_tsv)
    queries=[s for s in gold if s in S and any(t in T for t in gold[s])]
    print(f"  test queries: {len(queries)}", flush=True)

    res={k:collections.defaultdict(lambda: [0.0,0.0,0.0,0]) for k in ks}  # gen -> k -> [P,R,F1,n]
    gens=['lexical','graph','union']
    cand_sizes=collections.defaultdict(list)

    def cls_rank(text, topk):
        hits=idx.query(text, topk*4)
        best={}
        for i,sc in hits:
            u=idx.keys[i]
            if sc>best.get(u,-1): best[u]=sc
        return sorted(best.items(), key=lambda x:-x[1])[:topk]

    for n,s in enumerate(queries):
        if n%500==0 and n: print(f"    ..{n}/{len(queries)} ({time.time()-t0:.0f}s)", flush=True)
        g={t for t in gold[s] if t in T}
        # generator A: lexical
        lex=[u for u,_ in cls_rank(' . '.join(S[s]['labels']) or s, 10)]
        # generator B: graph expansion from lexical top-1
        base=lex[0] if lex else None
        grp=set()
        if base:
            grp.add(base); grp.update(T[base]['parents']); grp.update(children[base])
        grp=[u for u in grp if u in T]
        for gen in gens:
            if gen=='lexical': ranked=lex
            elif gen=='graph': ranked=grp
            else:
                seen=[]; sset=set()
                for u in lex+grp:
                    if u not in sset: sset.add(u); seen.append(u)
                ranked=seen
            cand_sizes[gen].append(len(ranked))
            if not g: continue
            for k in ks:
                top=set(ranked[:k])
                tp=len(top&g)
                P=tp/len(top) if top else 0.0
                R=tp/len(g)
                F=2*P*R/(P+R) if (P+R)>0 else 0.0
                cell=res[k][gen]; cell[0]+=P; cell[1]+=R; cell[2]+=F; cell[3]+=1

    print(f"\n  --- results ({name}) ---")
    print(f"  {'gen':8s} {'k':>3s} {'P':>7s} {'R':>7s} {'F1':>7s}   mean|cand|")
    for k in ks:
        for gen in gens:
            P,R,F,c=res[k][gen]
            if c: print(f"  {gen:8s} {k:3d} {P/c:7.3f} {R/c:7.3f} {F/c:7.3f}   {sum(cand_sizes[gen])/len(cand_sizes[gen]):8.1f}")
    return res, cand_sizes

if __name__=='__main__':
    base=sys.argv[1] if len(sys.argv)>1 else 'ncit-doid'
    d=f"bioml/{base}/{base}"
    if base=='ncit-doid': src,tgt=f"{d}/ncit.owl",f"{d}/doid.owl"
    else: src,tgt=f"{d}/omim.owl",f"{d}/ordo.owl"
    res, cs = evaluate(base, src, tgt, f"{d}/refs_equiv/test.tsv")
    KS=[1,3,5,10]
    out={'task':base,'ks':KS,
         'gens':{g:{str(k):[res[k][g][0]/res[k][g][3], res[k][g][1]/res[k][g][3], res[k][g][2]/res[k][g][3], res[k][g][3]] for k in KS} for g in ['lexical','graph','union']},
         'mean_cand':{g: sum(cs[g])/len(cs[g]) for g in cs}}
    json.dump(out, open(f'pilot_{base}.json','w'), indent=1)
    print('  dumped pilot_%s.json'%base)
