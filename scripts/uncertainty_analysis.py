"""
Sampling uncertainty of per-90 scouting metrics.

Input:  data/derived/player_match.parquet (from build_player_match.py)
Output: results_uncertainty/
  reliability_curves.csv        split-half reliability of each metric as a function of matches
  matches_to_stability.csv      matches (and minutes) needed for reliability >= 0.7 per metric
  percentile_intervals.csv      bootstrap 90% interval width of positional percentiles
  rank_stability.csv            share of the top-10 after k matches still in the top-10 at season end
  fig1_reliability_curves.png
  fig2_honest_pizza.png         the usual percentile chart next to the same chart with intervals

Usage:
  python scripts/uncertainty_analysis.py --derived data/derived --out results_uncertainty
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RNG = np.random.default_rng(2027)

# (column, label, numerator count, denominator: "minutes" -> per 90, or a count column -> ratio)
METRICS = [
    ("npxg_p90", "npxG /90", "npxg", "minutes"),
    ("shots_p90", "Shots /90", "shots", "minutes"),
    ("xa_p90", "xA /90", "xa", "minutes"),
    ("key_passes_p90", "Key passes /90", "key_passes", "minutes"),
    ("touches_box_p90", "Box touches /90", "touches_box", "minutes"),
    ("progressive_passes_p90", "Progressive passes /90", "progressive_passes", "minutes"),
    ("progressive_carries_p90", "Progressive carries /90", "progressive_carries", "minutes"),
    ("dribbles_completed_p90", "Successful dribbles /90", "dribbles_completed", "minutes"),
    ("pass_completion_pct", "Pass completion %", "passes_completed", "passes"),
    ("pressures_p90", "Pressures /90", "pressures", "minutes"),
    ("tackles_interceptions_p90", "Tackles + interceptions /90", "tackles_interceptions", "minutes"),
    ("ball_recoveries_p90", "Ball recoveries /90", "ball_recoveries", "minutes"),
    ("aerials_won_p90", "Aerial duels won /90", "aerials_won", "minutes"),
]
FAMILY = {"npxg_p90": "attacking", "shots_p90": "attacking", "xa_p90": "attacking", "key_passes_p90": "attacking",
          "touches_box_p90": "attacking", "progressive_passes_p90": "progression", "progressive_carries_p90": "progression",
          "dribbles_completed_p90": "progression", "pass_completion_pct": "progression", "pressures_p90": "defending",
          "tackles_interceptions_p90": "defending", "ball_recoveries_p90": "defending", "aerials_won_p90": "defending"}
POS_GROUPS = ["CB", "FB", "DM", "CM", "AM", "W", "FW"]
MIN_APPEARANCE = 30        # minutes for a match to count as an appearance
MIN_SEASON_MINUTES = 900   # reference population threshold
N_GRID = [3, 5, 8, 10, 13, 16, 19]   # a 38-match season allows two disjoint halves of up to 19
N_SPLITS = 40
N_BOOT = 400
RANK_CUTOFFS = [5, 10, 15, 20]
TOP_K = 10


# ----------------------------------------------------------------------------- helpers
COUNT_COLS = sorted({num for _, _, num, _ in METRICS} | {den for _, _, _, den in METRICS})
NUM_IDX = np.array([COUNT_COLS.index(num) for _, _, num, _ in METRICS])
DEN_IDX = np.array([COUNT_COLS.index(den) for _, _, _, den in METRICS])
SCALE = np.array([90.0 if den == "minutes" else 100.0 for _, _, _, den in METRICS])
METRIC_COLS = [col for col, _, _, _ in METRICS]


def rates(df: pd.DataFrame) -> pd.Series:
    """Metric rates for a set of rows (ratio of sums, never mean of ratios)."""
    return pd.Series(rates_from_counts(df[COUNT_COLS].to_numpy(dtype=float)), index=METRIC_COLS)


def rates_from_counts(counts: np.ndarray) -> np.ndarray:
    """Rates from a (matches x COUNT_COLS) array: ratio of column sums."""
    tot = counts.sum(axis=0)
    den = tot[DEN_IDX]
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(den > 0, tot[NUM_IDX] / den * SCALE, np.nan)


def count_matrices(pm: pd.DataFrame, keys: list, sort_by_date: bool = False) -> dict:
    """One (matches x COUNT_COLS) float array per player-season."""
    out = {}
    for k, g in pm.groupby(keys):
        if sort_by_date:
            g = g.sort_values("match_date")
        out[k] = g[COUNT_COLS].to_numpy(dtype=float)
    return out


def add_derived(pm: pd.DataFrame) -> pd.DataFrame:
    pm = pm.copy()
    pm["tackles_interceptions"] = pm["tackles"] + pm["interceptions"]
    pm["match_date"] = pd.to_datetime(pm["match_date"])
    return pm


def player_seasons(pm: pd.DataFrame) -> pd.DataFrame:
    keys = ["player_id", "player", "competition", "season"]
    g = pm.groupby(keys)
    ps = g.apply(rates, include_groups=False).reset_index()
    ps["minutes"] = g["minutes"].sum().values
    ps["matches"] = g.size().values
    pos = pm.groupby(keys + ["pos_group"])["minutes"].sum().reset_index().sort_values("minutes", ascending=False).drop_duplicates(keys)
    return ps.merge(pos[keys + ["pos_group"]], on=keys)


def percentile_of(value: float, reference: np.ndarray) -> float:
    return 100.0 * np.searchsorted(reference, value, side="right") / len(reference)


# ----------------------------------------------------------------------------- A. reliability curves
def reliability_curves(pm: pd.DataFrame) -> pd.DataFrame:
    """Split-half reliability: for n matches, correlation across players between the rates
    computed on two disjoint random samples of n matches each. Within position group."""
    rows = []
    keys = ["player_id", "competition", "season"]
    logs = count_matrices(pm, keys)
    pos_of = pm.groupby(keys)["pos_group"].agg(lambda s: s.value_counts().index[0])
    for pos in POS_GROUPS:
        ids = [k for k, p in pos_of.items() if p == pos]
        for n in N_GRID:
            eligible = [k for k in ids if len(logs[k]) >= 2 * n]
            if len(eligible) < 20:
                continue
            corr = []
            for _ in range(N_SPLITS):
                a = np.empty((len(eligible), len(METRICS)))
                b = np.empty_like(a)
                for i, k in enumerate(eligible):
                    m = logs[k]
                    idx = RNG.permutation(len(m))[: 2 * n]
                    a[i] = rates_from_counts(m[idx[:n]])
                    b[i] = rates_from_counts(m[idx[n:]])
                a, b = pd.DataFrame(a, columns=METRIC_COLS), pd.DataFrame(b, columns=METRIC_COLS)
                corr.append(a.corrwith(b, method="spearman").values)
            corr = np.nanmean(np.array(corr), axis=0)
            for (col, label, _, _), r in zip(METRICS, corr):
                rows.append({"pos_group": pos, "metric": col, "label": label, "family": FAMILY[col],
                             "matches": n, "players": len(eligible), "reliability": float(r)})
    return pd.DataFrame(rows)


def matches_to_stability(curves: pd.DataFrame, minutes_per_appearance: float, threshold: float = 0.7) -> pd.DataFrame:
    """Matches needed for reliability >= threshold, pooled across positions (player-weighted mean),
    with linear interpolation on the n grid."""
    pooled = (curves.assign(w=curves["players"]).groupby(["metric", "label", "family", "matches"])
              .apply(lambda g: np.average(g["reliability"], weights=g["w"]), include_groups=False)
              .rename("reliability").reset_index())
    out = []
    for (metric, label, family), g in pooled.groupby(["metric", "label", "family"]):
        g = g.sort_values("matches")
        x, y = g["matches"].values, g["reliability"].values
        if (y >= threshold).any():
            i = int(np.argmax(y >= threshold))
            n_star = x[i] if i == 0 else x[i - 1] + (threshold - y[i - 1]) * (x[i] - x[i - 1]) / (y[i] - y[i - 1])
        else:
            n_star = np.nan
        out.append({"metric": metric, "label": label, "family": family, "matches_to_r70": n_star,
                    "minutes_to_r70": n_star * minutes_per_appearance if pd.notna(n_star) else np.nan,
                    "reliability_at_5": float(np.interp(5, x, y)), "reliability_at_10": float(np.interp(10, x, y)),
                    "reliability_at_19": float(np.interp(19, x, y))})
    return pooled, pd.DataFrame(out).sort_values("matches_to_r70")


# ----------------------------------------------------------------------------- B. percentile intervals
def percentile_intervals(pm: pd.DataFrame, ps: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    ref = ps[ps["minutes"] >= MIN_SEASON_MINUTES]
    reference = {(pos, col): np.sort(ref.loc[ref["pos_group"] == pos, col].dropna().values)
                 for pos in POS_GROUPS for col, _, _, _ in METRICS}
    keys = ["player_id", "competition", "season"]
    logs = count_matrices(pm, keys)
    rows, detail = [], {}
    for _, r in ref.iterrows():
        k = (r["player_id"], r["competition"], r["season"])
        m = logs[k]
        if r["pos_group"] not in POS_GROUPS:
            continue
        idx = RNG.integers(0, len(m), (N_BOOT, len(m)))
        boot = pd.DataFrame(np.array([rates_from_counts(m[i]) for i in idx]), columns=METRIC_COLS)
        pct = {}
        for col, label, _, _ in METRICS:
            refv = reference[(r["pos_group"], col)]
            if len(refv) < 20:
                continue
            vals = boot[col].dropna().values
            dist = 100.0 * np.searchsorted(refv, vals, side="right") / len(refv)
            lo, hi = np.percentile(dist, [5, 95])
            point = percentile_of(r[col], refv)
            pct[col] = (point, lo, hi)
            rows.append({"player_id": r["player_id"], "player": r["player"], "pos_group": r["pos_group"], "minutes": r["minutes"],
                         "metric": col, "label": label, "family": FAMILY[col], "percentile": point, "p05": lo, "p95": hi, "width": hi - lo})
        detail[k] = (r, pct)
    df = pd.DataFrame(rows)
    df["minutes_bucket"] = pd.cut(df["minutes"], [899, 1500, 2500, 10000], labels=["900-1500", "1500-2500", "2500+"])
    summary = (df.groupby(["family", "label", "minutes_bucket"], observed=True)["width"]
               .median().unstack("minutes_bucket").round(1).reset_index())
    return df, summary, detail


# ----------------------------------------------------------------------------- C. ranking stability
def rank_stability(pm: pd.DataFrame, ps: pd.DataFrame) -> pd.DataFrame:
    ref = ps[ps["minutes"] >= MIN_SEASON_MINUTES]
    keys = ["player_id", "competition", "season"]
    logs = count_matrices(pm, keys, sort_by_date=True)
    rows = []
    for (comp, season), rs in ref.groupby(["competition", "season"]):
        for pos in POS_GROUPS:
            sub = rs[rs["pos_group"] == pos]
            if len(sub) < 2 * TOP_K:
                continue
            for k in RANK_CUTOFFS:
                # Same player set on both sides: only players with at least k appearances,
                # so that the overlap is not distorted by who is eligible at each cutoff.
                eligible = sub[sub["player_id"].map(lambda pid: len(logs[(pid, comp, season)]) >= k)]
                if len(eligible) < 2 * TOP_K:
                    continue
                early = pd.DataFrame([rates_from_counts(logs[(pid, comp, season)][:k]) for pid in eligible["player_id"]],
                                     columns=METRIC_COLS, index=eligible["player_id"].values)
                for col, label, _, _ in METRICS:
                    final_top = set(eligible.nlargest(TOP_K, col)["player_id"])
                    early_top = set(early[col].nlargest(TOP_K).index)
                    rows.append({"competition": comp, "season": season, "pos_group": pos, "metric": col, "label": label, "family": FAMILY[col],
                                 "matches": k, "players": len(eligible), "top10_overlap": len(early_top & final_top) / TOP_K})
    df = pd.DataFrame(rows)
    return df, df.groupby(["family", "label", "matches"])["top10_overlap"].mean().unstack("matches").round(2).reset_index()


# ----------------------------------------------------------------------------- figures
def fig_reliability(pooled: pd.DataFrame, mts: pd.DataFrame, out: Path) -> None:
    fams = ["attacking", "progression", "defending"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True)
    for ax, fam in zip(axes, fams):
        for metric, g in pooled[pooled["family"] == fam].groupby("metric"):
            g = g.sort_values("matches")
            ax.plot(g["matches"], g["reliability"], marker="o", ms=3, lw=1.4, label=g["label"].iloc[0])
        ax.axhline(0.7, color="black", lw=0.8, ls="--")
        ax.text(N_GRID[-1], 0.71, "r = 0.7", ha="right", va="bottom", fontsize=8)
        ax.set_title(fam.capitalize(), fontsize=10)
        ax.set_xlabel("Matches in each half-sample")
        ax.set_xticks(N_GRID)
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, frameon=False, loc="lower right")
    axes[0].set_ylabel("Split-half reliability (Spearman r)")
    fig.suptitle("How many matches until a per-90 metric is repeatable?", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out, dpi=220)
    plt.close(fig)


def fig_honest_pizza(detail: dict, ps: pd.DataFrame, out: Path, target_minutes=(900, 1300)) -> None:
    """Pick a real player with a modest sample and draw the usual percentile chart next to
    the same chart with 90% bootstrap intervals."""
    cands = [(k, v) for k, v in detail.items() if target_minutes[0] <= v[0]["minutes"] <= target_minutes[1] and len(v[1]) == len(METRICS)]
    if not cands:
        cands = [(k, v) for k, v in detail.items() if len(v[1]) == len(METRICS)]
    if not cands:
        return
    # choose the player with the widest average interval among the candidates: the most instructive case
    k, (r, pct) = max(cands, key=lambda kv: np.mean([hi - lo for _, lo, hi in kv[1][1].values()]))
    short = {"npxg_p90": "npxG", "shots_p90": "Shots", "xa_p90": "xA", "key_passes_p90": "Key passes", "touches_box_p90": "Box touches",
             "progressive_passes_p90": "Prog. passes", "progressive_carries_p90": "Prog. carries", "dribbles_completed_p90": "Dribbles",
             "pass_completion_pct": "Pass %", "pressures_p90": "Pressures", "tackles_interceptions_p90": "Tkl + Int",
             "ball_recoveries_p90": "Recoveries", "aerials_won_p90": "Aerials won"}
    labels = [short[col] for col, _, _, _ in METRICS]
    point = np.array([pct[col][0] for col, _, _, _ in METRICS])
    lo = np.array([pct[col][1] for col, _, _, _ in METRICS])
    hi = np.array([pct[col][2] for col, _, _, _ in METRICS])
    colors = [{"attacking": "#c05621", "progression": "#1f3a5f", "defending": "#4a7c59"}[FAMILY[col]] for col, _, _, _ in METRICS]
    theta = np.linspace(0, 2 * np.pi, len(METRICS), endpoint=False)
    width = 2 * np.pi / len(METRICS) * 0.9

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 6.8), subplot_kw={"projection": "polar"})
    for ax, title, with_ci in zip(axes, ["As usually shown", "With 90% bootstrap intervals"], [False, True]):
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        ax.bar(theta, point, width=width, color=colors, alpha=0.9 if not with_ci else 0.45, edgecolor="white", linewidth=1)
        if with_ci:
            for t, l, h, c in zip(theta, lo, hi, colors):
                ax.plot([t, t], [l, h], color="black", lw=1.6)
                ax.plot([t - width / 4, t + width / 4], [l, l], color="black", lw=1.2)
                ax.plot([t - width / 4, t + width / 4], [h, h], color="black", lw=1.2)
        for t, p, lab in zip(theta, point, labels):
            ax.text(t, 121, f"{lab}\n{p:.0f}", ha="center", va="center", fontsize=7)
        ax.set_ylim(0, 100)
        ax.set_yticks([25, 50, 75, 100])
        ax.set_yticklabels(["25", "50", "75", ""], fontsize=6, color="grey")
        ax.set_xticks([])
        ax.grid(alpha=0.35)
        ax.set_title(title, fontsize=10, pad=54)
    fig.suptitle(f"{r['pos_group']}, {int(r['minutes'])} minutes, {int(r['matches'])} matches — positional percentiles (same data, two charts)", fontsize=10.5, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out, dpi=220)
    plt.close(fig)


# ----------------------------------------------------------------------------- main
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--derived", default="data/derived")
    parser.add_argument("--out", default="results_uncertainty")
    parser.add_argument("--min-matches-per-competition", type=int, default=300, help="keep only complete seasons")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    pm = add_derived(pd.read_parquet(Path(args.derived) / "player_match.parquet"))
    counts = pm.groupby(["competition", "season"])["match_id"].nunique()
    complete = counts[counts >= args.min_matches_per_competition].index
    pm = pm[pm.set_index(["competition", "season"]).index.isin(complete)]
    pm = pm[pm["minutes"] >= MIN_APPEARANCE]
    print("Competitions used:", ", ".join(f"{c} {s} ({counts[(c, s)]} matches)" for c, s in complete))
    print(f"Appearances >= {MIN_APPEARANCE} min: {len(pm):,}")

    ps = player_seasons(pm)
    print(f"Player-seasons: {len(ps):,} | >= {MIN_SEASON_MINUTES} min: {(ps['minutes'] >= MIN_SEASON_MINUTES).sum():,}")

    print("\n[A] reliability curves...")
    curves = reliability_curves(pm)
    minutes_per_appearance = float(pm["minutes"].mean())
    print(f"Mean minutes per appearance: {minutes_per_appearance:.1f}")
    pooled, mts = matches_to_stability(curves, minutes_per_appearance)
    curves.to_csv(out / "reliability_curves.csv", index=False)
    pooled.to_csv(out / "reliability_curves_pooled.csv", index=False)
    mts.to_csv(out / "matches_to_stability.csv", index=False)
    print(mts.round(2).to_string(index=False))

    print("\n[B] percentile intervals (bootstrap)...")
    pdet, psum, detail = percentile_intervals(pm, ps)
    pdet.to_csv(out / "percentile_intervals_detail.csv", index=False)
    psum.to_csv(out / "percentile_intervals.csv", index=False)
    print("Median width of the 90% interval on the positional percentile, by minutes played:")
    print(psum.to_string(index=False))
    share_wide = (pdet[pdet["minutes_bucket"] == "900-1500"]["width"] >= 30).mean()
    print(f"Share of percentile estimates at 900-1500 minutes whose 90% interval spans 30+ points: {100 * share_wide:.0f}%")

    print("\n[C] ranking stability...")
    rdet, rsum = rank_stability(pm, ps)
    rdet.to_csv(out / "rank_stability_detail.csv", index=False)
    rsum.to_csv(out / "rank_stability.csv", index=False)
    print("Share of the season-end top-10 already in the top-10 after k matches:")
    print(rsum.to_string(index=False))
    print(f"Overall mean overlap at 10 matches: {rdet[rdet['matches'] == 10]['top10_overlap'].mean():.2f}")

    fig_reliability(pooled, mts, out / "fig1_reliability_curves.png")
    fig_honest_pizza(detail, ps, out / "fig2_honest_pizza.png")
    print(f"\nTables and figures written to {out}/")


if __name__ == "__main__":
    main()
