"""
Replication across data sets: overlay the pooled reliability curves and compare the
stabilization constants k of several runs of uncertainty_analysis.py.

Usage:
  python scripts/replication_figures.py --out results_replication \
      --run "Men 2015/16 (4 leagues)=results_uncertainty" \
      --run "Women 2023/24 (5 leagues)=results_women2023" \
      --run "Indian Super League 2021/22=results_isl"

Output: <out>/fig_replication_curves.png, <out>/fig_replication_k.png, <out>/k_comparison.csv
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

FAMILY_ORDER = ["attacking", "progression", "defending"]
STYLES = ["-", "--", ":", "-."]
COLORS = ["#1f3a5f", "#c05621", "#4a7c59", "#7b1e1e"]


def load_runs(specs: list) -> dict:
    runs = {}
    for spec in specs:
        name, path = spec.split("=", 1)
        p = Path(path)
        runs[name] = {"pooled": pd.read_csv(p / "reliability_curves_pooled.csv"), "k": pd.read_csv(p / "stabilization_constants.csv")}
        if (p / "k_vs_frequency.csv").exists():
            runs[name]["freq"] = pd.read_csv(p / "k_vs_frequency.csv")
    return runs


def fig_k_vs_frequency_all(runs: dict, out: Path) -> pd.DataFrame:
    """k against event frequency for every metric of every run on one plot, with a power-law fit."""
    from matplotlib.ticker import ScalarFormatter, NullFormatter
    rows = []
    for name, run in runs.items():
        if "freq" not in run:
            continue
        for _, r in run["freq"].iterrows():
            rows.append({"dataset": name, "metric": r["label"], "freq": r["events_per_appearance"], "k": r["k_matches"]})
    df = pd.DataFrame(rows)
    rho = df["freq"].corr(df["k"], method="spearman")
    b, a = np.polyfit(np.log(df["freq"]), np.log(df["k"]), 1)
    pred = a + b * np.log(df["freq"])
    r2 = 1 - ((np.log(df["k"]) - pred) ** 2).sum() / ((np.log(df["k"]) - np.log(df["k"]).mean()) ** 2).sum()
    fig, ax = plt.subplots(figsize=(6.2, 4.8))
    markers = ["o", "s", "^", "D"]
    for (name, g), mk, c in zip(df.groupby("dataset", sort=False), markers, COLORS):
        ax.scatter(g["freq"], g["k"], marker=mk, color=c, s=42, label=name, zorder=3, alpha=0.9)
    xs = np.logspace(np.log10(df["freq"].min() * 0.8), np.log10(df["freq"].max() * 1.3), 50)
    ax.plot(xs, np.exp(a) * xs ** b, color="grey", lw=1, ls="--", label=f"k = {np.exp(a):.1f} f^{b:.2f}")
    ax.set_xscale("log"); ax.set_yscale("log")
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_formatter(ScalarFormatter()); axis.set_minor_formatter(NullFormatter())
    ax.set_yticks([1, 2, 3, 5, 10, 20, 30]); ax.set_xticks([0.2, 0.5, 1, 2, 5, 10, 20, 50])
    ax.set_xlabel("Mean events per appearance, f (log scale)")
    ax.set_ylabel("Stabilization constant k (matches, log scale)")
    ax.set_title(f"k against event frequency across data sets (Spearman {rho:.2f})", fontsize=10)
    ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=7.5, frameon=False)
    fig.tight_layout(); fig.savefig(out, dpi=220); plt.close(fig)
    print(f"k vs frequency, all runs: n = {len(df)}, Spearman {rho:.2f}, k = {np.exp(a):.2f} f^{b:.2f}, R2 (log) = {r2:.2f}")
    return df


def fig_curves(runs: dict, out: Path) -> None:
    first = next(iter(runs.values()))["pooled"]
    metrics = (first[["metric", "label", "family"]].drop_duplicates()
               .assign(order=lambda d: d["family"].map({f: i for i, f in enumerate(FAMILY_ORDER)}))
               .sort_values(["order", "label"]))
    n = len(metrics)
    cols = 5
    rows_n = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows_n, cols, figsize=(2.9 * cols, 2.6 * rows_n), sharex=True, sharey=True)
    axes = axes.ravel()
    for ax, (_, m) in zip(axes, metrics.iterrows()):
        for (name, run), ls, c in zip(runs.items(), STYLES, COLORS):
            g = run["pooled"][run["pooled"]["metric"] == m["metric"]].sort_values("matches")
            ax.plot(g["matches"], g["reliability"], ls=ls, color=c, marker="o", ms=2.5, lw=1.4, label=name)
        ax.axhline(0.7, color="black", lw=0.6, ls="--")
        ax.set_title(m["label"], fontsize=8.5)
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.3)
        ax.tick_params(labelsize=7)
    for ax in axes[n:]:
        ax.axis("off")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower right", bbox_to_anchor=(0.98, 0.04), fontsize=8, frameon=False)
    fig.supxlabel("Matches in each half-sample", fontsize=9)
    fig.supylabel("Split-half reliability (Spearman r)", fontsize=9)
    fig.suptitle("The same metrics stabilise at the same speed in different competitions", fontsize=11)
    fig.tight_layout(rect=[0.02, 0.02, 1, 0.95])
    fig.savefig(out, dpi=220)
    plt.close(fig)


def k_table(runs: dict) -> pd.DataFrame:
    frames = []
    for name, run in runs.items():
        frames.append(run["k"][["metric", "label", "family", "k_matches"]].rename(columns={"k_matches": name}))
    df = frames[0]
    for f in frames[1:]:
        df = df.merge(f.drop(columns=["label", "family"]), on="metric", how="outer")
    names = list(runs)
    df["rank_" + names[0]] = df[names[0]].rank()
    for nm in names[1:]:
        df["rank_" + nm] = df[nm].rank()
    return df.sort_values(names[0])


def fig_k(df: pd.DataFrame, runs: dict, out: Path) -> None:
    """Stabilization constants of the other data sets against the reference set, on log axes."""
    from matplotlib.ticker import ScalarFormatter, NullFormatter
    from matplotlib.lines import Line2D
    names = list(runs)
    base = names[0]
    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    colors = {"attacking": "#c05621", "progression": "#1f3a5f", "defending": "#4a7c59"}
    markers = ["o", "s", "D", "^"]
    for nm, mk in zip(names[1:], markers):
        for _, r in df.iterrows():
            if pd.notna(r[base]) and pd.notna(r[nm]):
                ax.scatter(r[base], r[nm], color=colors[r["family"]], marker=mk, s=44, zorder=3, alpha=0.9, edgecolor="white", linewidth=0.5)
    vals = pd.concat([df[n] for n in names]).dropna()
    lo, hi = max(1.0, vals.min() * 0.8), vals.max() * 1.25
    ax.plot([lo, hi], [lo, hi], color="grey", lw=0.8, ls="--")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_formatter(ScalarFormatter()); axis.set_minor_formatter(NullFormatter())
    ticks = [t for t in [1, 2, 3, 5, 10, 20, 30] if lo <= t <= hi]
    ax.set_xticks(ticks); ax.set_yticks(ticks)
    ax.set_xlabel(f"k (matches), {base}")
    ax.set_ylabel("k (matches), other data sets")
    # label each metric once, at the point of the second run (the main replication)
    texts = [ax.text(r[base], r[names[1]], r["label"].replace(" /90", ""), fontsize=7)
             for _, r in df.iterrows() if pd.notna(r[base]) and pd.notna(r[names[1]])]
    try:
        from adjustText import adjust_text
        adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle="-", color="grey", lw=0.5), expand=(1.3, 1.6), force_text=(0.6, 0.9))
    except ImportError:
        pass
    handles = [Line2D([], [], marker=mk, color="grey", ls="", markersize=7, label=nm) for nm, mk in zip(names[1:], markers)]
    handles += [Line2D([], [], marker="o", color=c, ls="", markersize=7, label=f.capitalize()) for f, c in colors.items()]
    ax.legend(handles=handles, fontsize=7.5, frameon=False, loc="upper left")
    ax.set_title("Stabilization constant k across data sets", fontsize=10)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(out, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", required=True, help='"Name=results_dir" (first run is the reference)')
    parser.add_argument("--out", default="results_replication")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    runs = load_runs(args.run)
    fig_curves(runs, out / "fig_replication_curves.png")
    df = k_table(runs)
    df.to_csv(out / "k_comparison.csv", index=False)
    names = list(runs)
    print("Stabilization constant k (matches) per data set:")
    print(df[["label", "family"] + names].round(1).to_string(index=False))
    for nm in names[1:]:
        rho = df[names[0]].corr(df[nm], method="spearman")
        print(f"Spearman correlation of k across metrics, {names[0]} vs {nm}: {rho:.2f}")
    if len(names) > 1:
        fig_k(df, runs, out / "fig_replication_k.png")
    if any("freq" in r for r in runs.values()):
        fig_k_vs_frequency_all(runs, out / "fig_k_vs_frequency_all.png").to_csv(out / "k_vs_frequency_all.csv", index=False)
    print(f"\nFigures and table written to {out}/")


if __name__ == "__main__":
    main()