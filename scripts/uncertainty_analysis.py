"""
Sampling uncertainty of per-90 scouting metrics.

Input:  data/derived/player_match.parquet (from build_player_match.py)
Output: <out>/
  reliability_curves.csv        split-half reliability of each metric as a function of matches, by position
  reliability_curves_pooled.csv the same, pooled across positions (player-weighted)
  matches_to_stability.csv      matches (and minutes) needed for reliability >= 0.7 per metric
  stabilization_constants.csv   k per metric from r(n) = n / (n + k), with matches for r = 0.5 / 0.7 / 0.8
  stabilization_by_position.csv k per metric and position group
  percentile_intervals.csv      bootstrap 90% interval width of positional percentiles, by minutes played
  percentile_shrinkage.csv      how far percentiles move when shrunk towards the positional mean
  rank_stability.csv            share of the top-10 after k matches still in the top-10 at season end
  mirage_rate.csv               share of top-decile players at 900 minutes who are not top decile at season end
  fig1_reliability_curves.png
  fig2_honest_pizza.png         the usual percentile chart next to the same chart with intervals
  fig3_shrinkage.png            share of the observed value that is signal, m / (m + k), by minutes played
  k_vs_frequency.csv            stabilization constant against the mean number of events per appearance
  fig4_k_vs_frequency.png

Usage:
  python scripts/uncertainty_analysis.py --derived data/derived --out results_uncertainty
  python scripts/uncertainty_analysis.py --derived data/derived_women2023 --out results_women2023 \
      --min-matches-per-competition 100 --min-players 15
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

warnings.filterwarnings("ignore")
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
MIN_PLAYERS = 20           # minimum players per cell (metric x position x n); overridable with --min-players
SNAPSHOT_MINUTES = 900     # the "typical scouting sample" used for the mirage rate


# ----------------------------------------------------------------------------- helpers
COUNT_COLS = sorted({num for _, _, num, _ in METRICS} | {den for _, _, _, den in METRICS})
NUM_IDX = np.array([COUNT_COLS.index(num) for _, _, num, _ in METRICS])
DEN_IDX = np.array([COUNT_COLS.index(den) for _, _, _, den in METRICS])
SCALE = np.array([90.0 if den == "minutes" else 100.0 for _, _, _, den in METRICS])
METRIC_COLS = [col for col, _, _, _ in METRICS]


def configure_metrics(pm: pd.DataFrame, pos_groups=None) -> None:
    """Keep only the metrics the data set supports (columns present and not all NaN) and,
    optionally, replace the position groups. Rebuilds the module-level index arrays."""
    global METRICS, COUNT_COLS, NUM_IDX, DEN_IDX, SCALE, METRIC_COLS, POS_GROUPS
    keep = []
    for m in METRICS:
        col, label, num, den = m
        ok = num in pm.columns and den in pm.columns and pm[num].notna().any() and pm[den].notna().any()
        if ok:
            keep.append(m)
        else:
            print(f"  metric not available in this data set, skipped: {label}")
    METRICS = keep
    COUNT_COLS = sorted({num for _, _, num, _ in METRICS} | {den for _, _, _, den in METRICS})
    NUM_IDX = np.array([COUNT_COLS.index(num) for _, _, num, _ in METRICS])
    DEN_IDX = np.array([COUNT_COLS.index(den) for _, _, _, den in METRICS])
    SCALE = np.array([90.0 if den == "minutes" else 100.0 for _, _, _, den in METRICS])
    METRIC_COLS = [col for col, _, _, _ in METRICS]
    if pos_groups:
        POS_GROUPS = pos_groups


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
            if len(eligible) < MIN_PLAYERS:
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


# ----------------------------------------------------------------------------- D. stabilization constant
def fit_k(n: np.ndarray, r: np.ndarray) -> tuple[float, float]:
    """Least-squares fit of r(n) = n / (n + k). Returns k and the RMSE of the fit."""
    mask = np.isfinite(r)
    n, r = n[mask], r[mask]
    if len(n) < 2:
        return np.nan, np.nan
    def loss(k):
        return float(np.mean((r - n / (n + k)) ** 2))
    # coarse log-spaced grid, then golden-section refinement around the best cell (no scipy needed)
    grid = np.logspace(np.log10(0.05), np.log10(500), 400)
    best = int(np.argmin([loss(k) for k in grid]))
    lo, hi = grid[max(best - 1, 0)], grid[min(best + 1, len(grid) - 1)]
    phi = (np.sqrt(5) - 1) / 2
    a, b = hi - phi * (hi - lo), lo + phi * (hi - lo)
    for _ in range(60):
        if loss(a) < loss(b):
            hi, b = b, a
            a = hi - phi * (hi - lo)
        else:
            lo, a = a, b
            b = lo + phi * (hi - lo)
    k = (lo + hi) / 2
    return float(k), float(np.sqrt(loss(k)))


def stabilization_constants(pooled: pd.DataFrame, curves: pd.DataFrame, minutes_per_appearance: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """k per metric (pooled curve) and per metric x position. Matches needed for a target
    reliability r* follow from the model: n* = k * r* / (1 - r*)."""
    rows = []
    for (metric, label, family), g in pooled.groupby(["metric", "label", "family"]):
        k, rmse = fit_k(g["matches"].values.astype(float), g["reliability"].values.astype(float))
        rows.append({"metric": metric, "label": label, "family": family, "k_matches": k, "k_minutes": k * minutes_per_appearance,
                     "fit_rmse": rmse, "matches_for_r50": k, "matches_for_r70": k * 0.7 / 0.3, "matches_for_r80": k * 0.8 / 0.2,
                     "minutes_for_r70": k * 0.7 / 0.3 * minutes_per_appearance})
    by_metric = pd.DataFrame(rows).sort_values("k_matches")
    rows = []
    for (metric, label, pos), g in curves.groupby(["metric", "label", "pos_group"]):
        k, rmse = fit_k(g["matches"].values.astype(float), g["reliability"].values.astype(float))
        rows.append({"metric": metric, "label": label, "pos_group": pos, "k_matches": k, "fit_rmse": rmse, "players_min": int(g["players"].min())})
    by_pos = pd.DataFrame(rows)
    return by_metric, by_pos.pivot(index="label", columns="pos_group", values="k_matches").round(1).reset_index()


# ----------------------------------------------------------------------------- E. shrinkage
def percentile_shrinkage(ps: pd.DataFrame, kmet: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Empirical-Bayes shrinkage: a player's rate after m appearances is pulled towards the
    positional mean with weight 1 - m / (m + k). Players are then re-ranked on their shrunk
    rates, so the shrunk percentile is a percentile among shrunk estimates: what changes is the
    ordering, with small-sample players moving towards the middle relative to well-observed ones."""
    ref = ps[ps["minutes"] >= MIN_SEASON_MINUTES]
    kmap = kmet.set_index("metric")["k_matches"].to_dict()
    rows = []
    for pos in POS_GROUPS:
        sub = ref[ref["pos_group"] == pos]
        if len(sub) < MIN_PLAYERS:
            continue
        for col, label, _, _ in METRICS:
            k = kmap.get(col, np.nan)
            vals = sub[col].values
            ok = np.isfinite(vals)
            if not np.isfinite(k) or ok.sum() < MIN_PLAYERS:
                continue
            x, m = vals[ok], sub["matches"].values[ok]
            w = m / (m + k)
            shrunk = w * x + (1 - w) * x.mean()
            raw_ref, shr_ref = np.sort(x), np.sort(shrunk)
            raw_pct = 100.0 * np.searchsorted(raw_ref, x, side="right") / len(raw_ref)
            shr_pct = 100.0 * np.searchsorted(shr_ref, shrunk, side="right") / len(shr_ref)
            for pid, mins, mm, ww, rp, sp in zip(sub["player_id"].values[ok], sub["minutes"].values[ok], m, w, raw_pct, shr_pct):
                rows.append({"player_id": pid, "pos_group": pos, "minutes": mins, "matches": mm, "metric": col, "label": label,
                             "family": FAMILY[col], "reliability": ww, "raw_percentile": rp, "shrunk_percentile": sp, "shift": sp - rp})
    df = pd.DataFrame(rows)
    df["minutes_bucket"] = pd.cut(df["minutes"], [899, 1500, 2500, 10000], labels=["900-1500", "1500-2500", "2500+"])
    df["abs_shift"] = df["shift"].abs()
    summary = (df.groupby(["family", "label", "minutes_bucket"], observed=True)["abs_shift"].mean()
               .unstack("minutes_bucket").round(1).reset_index())
    # how often a raw top-decile player is no longer top decile once everyone is shrunk and re-ranked
    top_raw = df[df["raw_percentile"] >= 90]
    drop = (top_raw.groupby(["family", "label", "minutes_bucket"], observed=True)
            .apply(lambda g: (g["shrunk_percentile"] < 90).mean(), include_groups=False)
            .unstack("minutes_bucket").round(2).reset_index())
    drop.columns = ["family", "label"] + [f"top10_lost_{c}" for c in drop.columns[2:]]
    summary = summary.merge(drop, on=["family", "label"], how="left")
    return df, summary


# ----------------------------------------------------------------------------- F. mirage rate
def mirage_rate(pm: pd.DataFrame, ps: pd.DataFrame, snapshot_minutes: int = SNAPSHOT_MINUTES) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Of the players in the top decile of a metric when their season reaches `snapshot_minutes`,
    what share is no longer in the top decile at season end? Same player set on both sides."""
    keys = ["player_id", "competition", "season"]
    logs = count_matrices(pm, keys, sort_by_date=True)
    mins = {k: g.sort_values("match_date")["minutes"].to_numpy() for k, g in pm.groupby(keys)}
    # the snapshot must be a genuine early look: at least ~7 more appearances after it
    full = ps[ps["minutes"] >= snapshot_minutes + 600]
    rows = []
    for (comp, season), rs in full.groupby(["competition", "season"]):
        for pos in POS_GROUPS:
            sub = rs[rs["pos_group"] == pos]
            if len(sub) < MIN_PLAYERS:
                continue
            snap = {}
            for pid in sub["player_id"]:
                k = (pid, comp, season)
                cut = int(np.searchsorted(np.cumsum(mins[k]), snapshot_minutes) + 1)
                snap[pid] = rates_from_counts(logs[k][:cut])
            snap = pd.DataFrame(snap, index=METRIC_COLS).T
            final = sub.set_index("player_id")[METRIC_COLS]
            for col, label, _, _ in METRICS:
                q_snap, q_final = snap[col].quantile(0.9), final[col].quantile(0.9)
                top_snap = set(snap.index[snap[col] >= q_snap])
                top_final = set(final.index[final[col] >= q_final])
                if not top_snap:
                    continue
                rows.append({"competition": comp, "season": season, "pos_group": pos, "metric": col, "label": label, "family": FAMILY[col],
                             "players": len(sub), "top_decile_at_snapshot": len(top_snap),
                             "mirage_rate": 1 - len(top_snap & top_final) / len(top_snap)})
    df = pd.DataFrame(rows)
    if df.empty:
        return df, pd.DataFrame(columns=["family", "label", "mirage_rate", "cells"])
    summary = (df.assign(w=df["top_decile_at_snapshot"]).groupby(["family", "label"])
               .apply(lambda g: pd.Series({"mirage_rate": np.average(g["mirage_rate"], weights=g["w"]), "cells": len(g)}), include_groups=False)
               .reset_index().sort_values("mirage_rate", ascending=False))
    return df, summary


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

def place_labels(ax, texts) -> None:
    """Spread point labels apart when adjustText is installed; otherwise leave them in place."""
    try:
        from adjustText import adjust_text
        adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle="-", color="grey", lw=0.5), expand=(1.2, 1.4))
    except ImportError:
        for t in texts:
            t.set_position((t.get_position()[0] * 1.04, t.get_position()[1] * 1.03))


# ----------------------------------------------------------------------------- G. k versus event frequency
def k_vs_frequency(pm: pd.DataFrame, kmet: pd.DataFrame) -> pd.DataFrame:
    """Does a metric stabilise faster simply because its events are more frequent? Mean events
    per appearance (the numerator count; for ratios, the denominator count) against k."""
    # xG and xA are sums of probabilities, so their underlying events are shots and key passes
    event_col = {"npxg_p90": "shots", "xa_p90": "key_passes"}
    rows = []
    for col, label, num, den in METRICS:
        count_col = event_col.get(col, num if den == "minutes" else den)
        rows.append({"metric": col, "label": label, "family": FAMILY[col],
                     "events_per_appearance": float(pm[count_col].mean()), "count_column": count_col})
    df = pd.DataFrame(rows).merge(kmet[["metric", "k_matches"]], on="metric")
    rho = df["events_per_appearance"].corr(df["k_matches"], method="spearman")
    df["spearman_k_vs_frequency"] = rho
    return df.sort_values("events_per_appearance")


def fig_k_vs_frequency(df: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    colors = {"attacking": "#c05621", "progression": "#1f3a5f", "defending": "#4a7c59"}
    texts = []
    for _, r in df.iterrows():
        ax.scatter(r["events_per_appearance"], r["k_matches"], color=colors[r["family"]], s=40, zorder=3)
        texts.append(ax.text(r["events_per_appearance"], r["k_matches"], r["label"].replace(" /90", ""), fontsize=7))
    ax.set_xscale("log")
    ax.set_yscale("log")
    from matplotlib.ticker import ScalarFormatter, NullFormatter
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_formatter(ScalarFormatter())
        axis.set_minor_formatter(NullFormatter())
    ax.set_yticks([2, 3, 5, 10, 20, 30])
    place_labels(ax, texts)
    ax.set_xlabel("Mean events per appearance (log scale)")
    ax.set_ylabel("Stabilization constant k (matches, log scale)")
    rho = df["spearman_k_vs_frequency"].iloc[0]
    ax.set_title(f"Rarer events, slower stabilisation (Spearman {rho:.2f})", fontsize=10)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(out, dpi=220)
    plt.close(fig)


def fig_shrinkage(kmet: pd.DataFrame, minutes_per_appearance: float, out: Path) -> None:
    """Share of the observed per-90 value that is signal, m / (m + k), as a function of minutes
    played, for every metric. The complement is the weight given to the positional mean."""
    fams = ["attacking", "progression", "defending"]
    minutes = np.linspace(200, 3400, 200)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True)
    for ax, fam in zip(axes, fams):
        for _, r in kmet[kmet["family"] == fam].sort_values("k_matches").iterrows():
            m = minutes / minutes_per_appearance
            ax.plot(minutes, m / (m + r["k_matches"]), lw=1.5, label=f"{r['label']} (k = {r['k_matches']:.1f})")
        ax.axvline(900, color="black", lw=0.8, ls="--")
        ax.text(930, 0.04, "900 min", fontsize=7.5)
        ax.set_title(fam.capitalize(), fontsize=10)
        ax.set_xlabel("Minutes played")
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, frameon=False, loc="lower right")
    axes[0].set_ylabel("Signal share  m / (m + k)")
    fig.suptitle("How much of a per-90 value to keep, and how much to pull towards the positional mean", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out, dpi=220)
    plt.close(fig)


# ----------------------------------------------------------------------------- main
def main() -> None:
    global MIN_PLAYERS
    parser = argparse.ArgumentParser()
    parser.add_argument("--derived", default="data/derived")
    parser.add_argument("--out", default="results_uncertainty")
    parser.add_argument("--min-matches-per-competition", type=int, default=300, help="keep only complete seasons")
    parser.add_argument("--min-players", type=int, default=MIN_PLAYERS, help="minimum players per cell")
    parser.add_argument("--max-n", type=int, default=None, help="largest half-sample size (about half the season length)")
    parser.add_argument("--pos-groups", default=None, help="comma-separated position groups to use instead of the default seven")
    args = parser.parse_args()
    MIN_PLAYERS = args.min_players
    global N_GRID
    if args.max_n:
        N_GRID = [n for n in N_GRID if n <= args.max_n]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    pm = add_derived(pd.read_parquet(Path(args.derived) / "player_match.parquet"))
    configure_metrics(pm, args.pos_groups.split(",") if args.pos_groups else None)
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

    print("\n[D] stabilization constants: r(n) = n / (n + k)...")
    kmet, kpos = stabilization_constants(pooled, curves, minutes_per_appearance)
    kmet.to_csv(out / "stabilization_constants.csv", index=False)
    kpos.to_csv(out / "stabilization_by_position.csv", index=False)
    print(kmet.round(2).to_string(index=False))
    print("\nk by position group (matches):")
    print(kpos.to_string(index=False))

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

    print("\n[E] shrinkage towards the positional mean...")
    shr, shr_sum = percentile_shrinkage(ps, kmet)
    shr.to_csv(out / "percentile_shrinkage_detail.csv", index=False)
    shr_sum.to_csv(out / "percentile_shrinkage.csv", index=False)
    print("Mean |shift| of the percentile after shrinkage, and share of raw top-decile players that drop out:")
    print(shr_sum.to_string(index=False))

    print(f"\n[F] mirage rate at {SNAPSHOT_MINUTES} minutes (players with >= {SNAPSHOT_MINUTES + 600} minutes)...")
    mir, mir_sum = mirage_rate(pm, ps)
    mir.to_csv(out / "mirage_rate_detail.csv", index=False)
    mir_sum.to_csv(out / "mirage_rate.csv", index=False)
    print("Share of top-decile players at the snapshot who are not top decile at season end:")
    print(mir_sum.round(2).to_string(index=False))

    fig_reliability(pooled, mts, out / "fig1_reliability_curves.png")
    fig_honest_pizza(detail, ps, out / "fig2_honest_pizza.png")
    fig_shrinkage(kmet, minutes_per_appearance, out / "fig3_shrinkage.png")

    kf = k_vs_frequency(pm, kmet)
    kf.to_csv(out / "k_vs_frequency.csv", index=False)
    print(f"\n[G] k versus event frequency (Spearman across metrics: {kf['spearman_k_vs_frequency'].iloc[0]:.2f})")
    print(kf[["label", "events_per_appearance", "k_matches"]].round(2).to_string(index=False))
    fig_k_vs_frequency(kf, out / "fig4_k_vs_frequency.png")
    print(f"\nTables and figures written to {out}/")


if __name__ == "__main__":
    main()
