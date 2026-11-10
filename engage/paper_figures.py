#!/usr/bin/env python3
"""Charts for the CVPR paper (matplotlib, LaTeX text in Times so they match the paper body).

    ~/miniconda3/envs/lgd/bin/python paper_figures.py

Reads results/analysis.json (sweep, noise) and engage/fuzzy.py (FOU illustration) and writes
vector PDFs at exact CVPR column width to ../../cvpr2027_paper/figures/{fou,sweep,noise}.pdf.
Include them with \\includegraphics[width=\\linewidth]{figures/<name>} (no rescaling).
Palette: blue #2391FF, yellow #FFF83F, pink #F70063, plus greys. No orange.
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from engage.fuzzy import EngageInput, EngagementFLS, sigmoid_um

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
FIG = os.path.join(HERE, "..", "..", "cvpr2027_paper", "figures")

COLW = 3.25  # CVPR column width, inches (text width 6.875 in)
BLUE, YELLOW, PINK = "#2391FF", "#FFF83F", "#F70063"
GREY, INK = "#8A8A86", "#1A1A1A"
THETA_E = 0.60  # frozen motion-command threshold (sec. 4)

plt.rcParams.update({
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage{times}\usepackage{amsmath}",
    "font.family": "serif", "font.size": 8, "axes.labelsize": 8.5, "legend.fontsize": 7.5,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": INK, "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5, "lines.linewidth": 1.4,
    "legend.frameon": False, "legend.handlelength": 2.2, "legend.columnspacing": 1.4,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.01, "pdf.fonttype": 42,
})


def nolead(x, _):
    """Tick labels like the reference figure: 0, .25, .5, 1."""
    s = f"{x:g}"
    return s[1:] if s.startswith("0.") else s


def panel(ax, tag):
    ax.text(0.97, 0.05, rf"\textbf{{({tag})}}", transform=ax.transAxes, ha="right", va="bottom", fontsize=8.5)


def top_legend(fig, handles, labels, gap=0.015, **kw):
    """One legend centred just above the top row of axes."""
    y = max(ax.get_position().y1 for ax in fig.axes) + gap
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, y), **kw)


def save(fig, name):
    os.makedirs(FIG, exist_ok=True)
    fig.savefig(os.path.join(FIG, name + ".pdf"))
    plt.close(fig)


def fig_fou(dist=2.2, sigmas=(0.08, 0.20)):
    fs = [i / 200 for i in range(201)]
    styles = {0.08: ("--", 0.22), 0.20: (":", 0.10)}
    fig, (a, b) = plt.subplots(1, 2, figsize=(COLW, 1.45), gridspec_kw={"wspace": 0.42})
    # (a) facing-HIGH set: type-1 line, FOU band and lower membership for each sigma
    for s in sigmas[::-1]:
        ls, alpha = styles[s]
        lo = [sigmoid_um(f, 0.62, 12.0, s)[0] for f in fs]
        up = [sigmoid_um(f, 0.62, 12.0, s)[1] for f in fs]
        a.fill_between(fs, lo, up, color=PINK, alpha=alpha, lw=0)
        a.plot(fs, lo, ls, color=PINK)
    a.plot(fs, [sigmoid_um(f, 0.62, 12.0, 0.0)[0] for f in fs], "-", color=BLUE)
    a.set(xlim=(0, 1), ylim=(0, 1.03), xlabel=r"facing score $f$", ylabel=r"$\mu_{\mathrm{HIGH}}(f)$",
          xticks=[0, .25, .5, .75, 1], yticks=[0, .5, 1])
    panel(a, "a")
    # (b) engagement vs facing for a still person at `dist`: type-1 E and IT2 lower bound
    t1, it2 = EngagementFLS(type1=True), EngagementFLS()
    b.axhspan(THETA_E, 0.85, color=YELLOW, alpha=0.45, lw=0)
    b.axhline(THETA_E, color=GREY, ls=(0, (3, 2)), lw=0.7)
    b.text(0.02, 0.83, r"accept if $\underline{E}\geq\theta_E$", color=INK, fontsize=6.5, va="top")
    h = [b.plot(fs, [t1(EngageInput(dist, 0.0, f, 0.0, 0.0)).e_low for f in fs], "-", color=BLUE)[0]]
    for s in sigmas:
        h.append(b.plot(fs, [it2(EngageInput(dist, 0.0, f, s, 0.0)).e_low for f in fs],
                        styles[s][0], color=PINK)[0])
    b.set(xlim=(0, 1), ylim=(0, 0.85), xlabel=rf"facing score $f$ ($d={dist}$\,m)", ylabel=r"engagement",
          xticks=[0, .25, .5, .75, 1], yticks=[0, .2, .4, .6, .8])
    panel(b, "b")
    for ax in (a, b):
        ax.xaxis.set_major_formatter(FuncFormatter(nolead))
        ax.yaxis.set_major_formatter(FuncFormatter(nolead))
    top_legend(fig, h, [r"type-1 ($\sigma_f{=}0$)", r"IT2, $\sigma_f{=}0.08$", r"IT2, $\sigma_f{=}0.20$"], ncol=3)
    save(fig, "fou")


def fig_sweep(an):
    fig, ax = plt.subplots(figsize=(COLW, 1.75))
    ax.axvspan(0, 2, color=YELLOW, alpha=0.55, lw=0)
    ax.text(4.0, 8, r"FT\,$\leq$\,2\,\%", fontsize=7, color=INK)
    style = {"it2": ("IT2 engagement (ours)", BLUE, "-"), "type1": ("Type-1 engagement", PINK, "--"),
             "facing_only": ("Facing only", GREY, ":")}
    h = []
    for m, (lab, c, ls) in style.items():
        cur = an["sweep"]["curves"][m]
        h.append(ax.plot([100 * x[1] for x in cur], [100 * x[2] for x in cur], ls, color=c, label=lab)[0])
    op = an["operating_point"]
    for m, mk, c in (("it2", "o", BLUE), ("type1", "s", PINK), ("gesture_only", "D", GREY)):
        ax.plot(100 * op[m][0]["p"], 100 * op[m][1]["p"], mk, color=c, ms=4.5, mec="white", mew=0.7, zorder=5)
    g = op["gesture_only"]
    ax.annotate("gesture only", (100 * g[0]["p"], 100 * g[1]["p"]), xytext=(-4, -10),
                textcoords="offset points", ha="right", fontsize=7, color=INK)
    o = op["it2"]
    ax.annotate(r"frozen $\theta_E$", (100 * o[0]["p"], 100 * o[1]["p"]), xytext=(12, -16),
                textcoords="offset points", fontsize=7, color=INK,
                arrowprops=dict(arrowstyle="-", color=GREY, lw=0.5))
    ax.set(xlim=(0, 65), ylim=(0, 80), xlabel=r"Bystander false-trigger rate (\%)",
           ylabel=r"Commander recall (\%)")
    top_legend(fig, h, [x.get_label() for x in h], ncol=3, handlelength=1.8, columnspacing=1.0)
    save(fig, "sweep")


def fig_noise(an):
    lv, res = an["noise"]["levels"], an["noise"]["results"]
    series = (("type1", "Type-1", PINK, "--", "s"), ("it2", r"IT2 ($\sigma$ updated)", BLUE, "-", "o"),
              ("it2_fixed_sigma", r"IT2 ($\sigma$ not updated)", BLUE, ":", "x"),
              ("facing_only", "Facing only", GREY, "-.", "^"))
    fig, axs = plt.subplots(2, 2, figsize=(COLW, 2.35), sharey="row", sharex="col",
                            gridspec_kw={"hspace": 0.12, "wspace": 0.08})
    for col, var, xl in ((0, "facing", "facing noise $s$"), (1, "distance", "distance noise $s$ (m)")):
        for row, key in ((0, "ft"), (1, "recall")):
            ax = axs[row, col]
            for m, lab, c, ls, mk in series:
                v = [res[var][str(s)][m] for s in lv[var]]
                ax.errorbar(lv[var], [100 * x[key + "_mean"] for x in v], yerr=[100 * x[key + "_sd"] for x in v],
                            fmt=ls, marker=mk, color=c, lw=1.1, ms=3.2, capsize=1.5, elinewidth=0.6, label=lab)
            if row:
                ax.set_xlabel(xl)
            ax.xaxis.set_major_formatter(FuncFormatter(nolead))
        axs[1, col].set_xticks([0, .1, .2, .3] if var == "facing" else [0, .5, 1])
    axs[0, 0].set_ylabel(r"false triggers (\%)")
    axs[1, 0].set_ylabel(r"recall (\%)")
    axs[0, 0].set_ylim(bottom=0)
    axs[0, 0].set_yticks([0, 2, 4, 6, 8])
    h, lab = axs[0, 0].get_legend_handles_labels()
    top_legend(fig, h, lab, ncol=2, handlelength=2.6)
    save(fig, "noise")


def main():
    an = json.load(open(os.path.join(RESULTS, "analysis.json")))
    fig_fou()
    fig_sweep(an)
    fig_noise(an)
    print("wrote fou, sweep, noise ->", os.path.abspath(FIG))


if __name__ == "__main__":
    main()
