"""
Season-to-season validity of per-90 metrics.

For every pair of consecutive seasons in a data set, players with at least 900 minutes in
both seasons are kept, their positional percentiles are computed within each season, and the
Spearman correlation across players between the two seasons is reported per metric. The
within-season split-half reliability (Spearman-Brown corrected to a full season) is reported
next to it as the ceiling that sampling noise alone would allow.

Input:  data/derived_wsl/player_match.parquet (several seasons of one competition)
Output: <out>/season_validity.csv, <out>/season_validity_detail.csv, <out>/fig_season_validity.png

Usage:
  python scripts/season_validity.py --derived data/derived_wsl --out results_wsl_validity --min-players 12
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import uncertainty_analysis as ua  # noqa: E402

warnings.filterwarnings("ignore")


def positional_percentiles(ps: pd.DataFrame, min_players: int) -> pd.DataFrame:
    """Percentile of each metric within position group and season (reference: >= 900 minutes)."""
    ref = ps[ps["minutes"] >= ua.MIN_SEASON_MINUTES].copy()
    for col in ua.METRIC_COLS:
        ref[f"{col}_pct"] = np.nan
    for (season, pos), g in ref.groupby(["season", "pos_group"]):
        if len(g) < min_players:
            continue
        for col in ua.METRIC_COLS:
            vals = g[col].values
            ok = np.isfinite(vals)
            srt = np.sort(vals[ok])
            pct = np.full(len(g), np.nan)
            pct[ok] = 100.0 * np.searchsorted(srt, vals[ok], side="right") / len(srt)
            ref.loc[g.index, f"{col}_pct"] = pct
    return ref


def season_pairs(seasons: list) -> list:
    """Consecutive pairs by start year, e.g. 2018/2019 -> 2019/2020."""
    def start(s):
        return int(str(s)[:4])
    ordered = sorted(seasons, key=start)
    return [(a, b) for a, b in zip(ordered[:-1], ordered[1:]) if start(b) == start(a) + 1]


def detail_matches(ref: pd.DataFrame) -> float:
    """Mean appearances of the player-seasons that take part in at least one consecutive pair."""
    ids = set()
    for a, b in season_pairs(ref["season"].unique().tolist()):
        both = set(ref.loc[ref["season"] == a, "player_id"]) & set(ref.loc[ref["season"] == b, "player_id"])
        ids |= {(pid, a) for pid in both} | {(pid, b) for pid in both}
    sub = ref[[(pid, se) in ids for pid, se in zip(ref["player_id"], ref["season"])]]
    return sub["matches"].mean()


def validity(ref: pd.DataFrame, min_players: int) -> pd.DataFrame:
    rows = []
    for a, b in season_pairs(ref["season"].unique().tolist()):
        both = ref[ref["season"] == a].merge(ref[ref["season"] == b], on="player_id", suffixes=("_a", "_b"))
        if len(both) < min_players:
            continue
        for col, label, _, _ in ua.METRICS:
            x, y = both[f"{col}_pct_a"], both[f"{col}_pct_b"]
            ok = x.notna() & y.notna()
            if ok.sum() < min_players:
                continue
            rows.append({"season_a": a, "season_b": b, "metric": col, "label": label, "family": ua.FAMILY[col],
                         "players": int(ok.sum()), "spearman": float(x[ok].corr(y[ok], method="spearman"))})
    return pd.DataFrame(rows)


def reliability_ceiling(k_table: Path, mean_matches: float) -> pd.DataFrame:
    """Full-season reliability implied by the stabilization constant of each metric,
    r = N / (N + k), where N is the mean number of appearances of the players in the pairs.
    k comes from a larger data set (e.g. results_women2023/stabilization_constants.csv)."""
    k = pd.read_csv(k_table)[["metric", "k_matches"]]
    k["ceiling_matches"] = mean_matches
    k["full_season_reliability"] = mean_matches / (mean_matches + k["k_matches"])
    return k


def fig_validity(summary: pd.DataFrame, out: Path) -> None:
    df = summary.sort_values("spearman")
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    y = np.arange(len(df))
    colors = [{"attacking": "#c05621", "progression": "#1f3a5f", "defending": "#4a7c59"}[f] for f in df["family"]]
    ax.barh(y, df["spearman"], color=colors, height=0.6, label="Season-to-season correlation")
    ax.scatter(df["full_season_reliability"], y, marker="|", s=220, color="black", zorder=3, label="Reliability ceiling, N / (N + k)")
    ax.set_yticks(y)
    ax.set_yticklabels(df["label"], fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Spearman correlation across players")
    ax.set_title("Does this season's number predict next season's?", fontsize=10)
    ax.grid(axis="x", alpha=0.3)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(out, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--derived", default="data/derived_wsl")
    parser.add_argument("--out", default="results_wsl_validity")
    parser.add_argument("--min-matches-per-competition", type=int, default=80)
    parser.add_argument("--min-players", type=int, default=12)
    parser.add_argument("--k-from", default="results_women2023/stabilization_constants.csv",
                        help="stabilization constants used for the reliability ceiling")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    pm = ua.add_derived(pd.read_parquet(Path(args.derived) / "player_match.parquet"))
    counts = pm.groupby(["competition", "season"])["match_id"].nunique()
    complete = counts[counts >= args.min_matches_per_competition].index
    pm = pm[pm.set_index(["competition", "season"]).index.isin(complete)]
    pm = pm[pm["minutes"] >= ua.MIN_APPEARANCE]
    print("Seasons used:", ", ".join(f"{c} {s} ({counts[(c, s)]} matches)" for c, s in complete))

    ps = ua.player_seasons(pm)
    ref = positional_percentiles(ps, args.min_players)
    print(f"Player-seasons >= {ua.MIN_SEASON_MINUTES} min: {len(ref):,}")

    detail = validity(ref, args.min_players)
    detail.to_csv(out / "season_validity_detail.csv", index=False)
    summary = (detail.assign(w=detail["players"]).groupby(["metric", "label", "family"])
               .apply(lambda g: pd.Series({"spearman": np.average(g["spearman"], weights=g["w"]), "players": g["players"].sum(),
                                           "pairs": len(g)}), include_groups=False).reset_index())
    mean_matches = float(detail_matches(ref))
    ceiling = reliability_ceiling(Path(args.k_from), mean_matches)
    print(f"Mean appearances per player-season in the pairs: {mean_matches:.1f}")
    summary = summary.merge(ceiling, on="metric", how="left").sort_values("spearman", ascending=False)
    summary.to_csv(out / "season_validity.csv", index=False)
    print("\nSeason-to-season Spearman correlation of positional percentiles (players with >= 900 min in both seasons):")
    print(summary.round(2).to_string(index=False))
    fig_validity(summary, out / "fig_season_validity.png")
    print(f"\nTables and figure written to {out}/")


if __name__ == "__main__":
    main()
