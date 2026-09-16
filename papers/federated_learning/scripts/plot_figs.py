#!/usr/bin/env python3
"""Figures for the FedHarm evaluation section, from measured pilot JSON."""
import json, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figs")
HERE = os.path.dirname(os.path.abspath(__file__))
plt.rcParams.update({
    'font.size': 7, 'axes.labelsize': 7, 'axes.titlesize': 7.5,
    'legend.fontsize': 6.3, 'xtick.labelsize': 6.5, 'ytick.labelsize': 6.5,
    'font.family': 'serif', 'axes.linewidth': 0.6, 'lines.linewidth': 1.1,
    'figure.dpi': 300, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.02,
})

STYLE = {  # grayscale-safe: linestyle + marker carry the signal
    'lexical': dict(color='#1a1a1a', ls='-',  marker='o', label='lexical only'),
    'graph':   dict(color='#7a7a7a', ls='--', marker='s', label='graph expansion'),
    'union':   dict(color='#9a6a2f', ls=':',  marker='^', label='union'),
}
D = {t: json.load(open(os.path.join(HERE, f'pilot_{t}.json'))) for t in ('ncit-doid', 'omim-ordo')}
KS = [1, 3, 5, 10]

# ---------- Fig 1: recall@k, both tasks ----------
fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.35), sharey=True)
for ax, task, title in zip(axes, ('ncit-doid', 'omim-ordo'),
                           ('NCIT--DOID (3,245 queries)', 'OMIM--ORDO (2,595 queries)')):
    for g in ('lexical', 'graph', 'union'):
        r = [D[task]['gens'][g][str(k)][1] for k in KS]
        ax.plot(KS, r, **STYLE[g])
    ax.set_title(title)
    ax.set_xlabel('candidates retrieved, $k$')
    ax.set_xticks(KS)
    ax.grid(alpha=0.25, lw=0.4)
    ax.set_ylim(0.25, 1.0)
axes[0].set_ylabel('recall vs. reference alignment')
axes[0].legend(loc='lower right', framealpha=0.9, borderpad=0.3)
fig.savefig(f"{OUT}/fig_recall_k.pdf")
fig.savefig(f"{OUT}/fig_recall_k.png", dpi=600)
plt.close(fig)

# ---------- Fig 2: precision-recall plane, NCIT-DOID ----------
PUB = [  # OAEI Bio-ML, unsupervised; values as published (see Table II)
    ('LogMap',        0.934, 0.668, 's', '#4a4a4a'),
    ('BERTMap',       0.888, 0.878, 'D', '#4a4a4a'),
    ('SORBETMatcher', 0.920, 0.907, 'v', '#4a4a4a'),
    ('OLaLa',         0.913, 0.864, 'P', '#4a4a4a'),
    ('MILA',          0.964, 0.932, '*', '#b00020'),
]
fig, ax = plt.subplots(figsize=(3.45, 2.85))
lex = [(D['ncit-doid']['gens']['lexical'][str(k)][0],
        D['ncit-doid']['gens']['lexical'][str(k)][1], k) for k in KS]
ax.plot([p[1] for p in lex], [p[0] for p in lex], '-o', color='#9a6a2f',
        ms=3.6, label='our retrieval only (no adjudication)', zorder=3)
for r, p, k in lex[:-1]:
    ax.annotate(f'$k$={k}', (r, p), textcoords='offset points', xytext=(3, -8),
                fontsize=5.6, color='#7a5322')
ax.annotate(f'$k$={lex[-1][2]}', (lex[-1][1], lex[-1][0]), textcoords='offset points',
            xytext=(-19, 4), fontsize=5.6, color='#7a5322')
for name, p, r, mk, col in PUB:
    ax.plot(r, p, marker=mk, color=col, ms=5.8, ls='none', zorder=4,
            label=f'{name} (pub.)')
ax.set_xlabel('recall'); ax.set_ylabel('precision')
ax.set_xlim(0.62, 1.01); ax.set_ylim(0.0, 1.08)
ax.grid(alpha=0.25, lw=0.4)
ax.legend(loc='lower left', framealpha=0.94, borderpad=0.3, handletextpad=0.4,
          labelspacing=0.25, columnspacing=0.7, ncol=2, fontsize=5.9)
ax.set_title('NCIT--DOID: published points, our curve', fontsize=7.0)
fig.savefig(f"{OUT}/fig_pr_plane.pdf")
fig.savefig(f"{OUT}/fig_pr_plane.png", dpi=600)
plt.close(fig)

# ---------- Fig 3: cost of union ----------
fig, ax = plt.subplots(figsize=(3.45, 2.2))
import numpy as np
tasks = ['ncit-doid', 'omim-ordo']
x = np.arange(len(tasks)); w = 0.26
for i, g in enumerate(('lexical', 'graph', 'union')):
    vals = [D[t]['mean_cand'][g] for t in tasks]
    ax.bar(x + (i - 1) * w, vals, w, color=STYLE[g]['color'], edgecolor='white', lw=0.5,
           hatch=['', '///', '...'][i], label=STYLE[g]['label'])
for i, t in enumerate(tasks):
    for j, g in enumerate(('lexical', 'graph', 'union')):
        r10 = D[t]['gens'][g]['10'][1]
        ax.annotate(f'{r10:.3f}', (i + (j - 1) * w, D[t]['mean_cand'][g]),
                    textcoords='offset points', xytext=(0, 2), ha='center', fontsize=5.4)
ax.set_xticks(x); ax.set_xticklabels(['NCIT--DOID', 'OMIM--ORDO'])
ax.set_ylabel('mean candidates per query')
ax.set_ylim(0, 16.5)
ax.grid(alpha=0.22, lw=0.4, axis='y')
ax.legend(loc='upper left', framealpha=0.9, borderpad=0.3)
ax.set_title('Union costs more candidates; annotation = recall@10', fontsize=6.8)
fig.savefig(f"{OUT}/fig_cost.pdf")
fig.savefig(f"{OUT}/fig_cost.png", dpi=600)
plt.close(fig)

print("figures written:")
for f in sorted(os.listdir(OUT)):
    print("  ", f, os.path.getsize(f"{OUT}/{f}"), "bytes")
