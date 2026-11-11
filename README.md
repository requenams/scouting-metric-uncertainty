# How Many Minutes Until a Number Means Something?

Code and data for the paper *How Many Minutes Until a Number Means Something? Sampling Uncertainty
in Data-Driven Scouting*, submitted to the MIT Sloan Sports Analytics Conference 2027 research paper
competition (paper ID 175). The manuscript is in `paper/`.

## What this is about

Scouting profiles show a player's per-90 numbers as percentiles against his position, usually as a
radar or pizza chart, and usually without any indication of how reliable those percentiles are at the
sample sizes scouts work with. We measured it: how many matches each of thirteen common metrics
needs before it becomes repeatable, how wide the uncertainty around a positional percentile is, how
much a percentile should be discounted when the sample is small, and how often an early-season
ranking still holds at the end of the season.

In short: pass completion, pressures and progressive actions settle after six to eight matches;
shots, chances created and expected assists are still noisy after sixteen. A single constant per
metric (k, in r(n) = n / (n + k)) describes each reliability curve, and k is mostly a matter of how
often the underlying event happens. Between 900 and 1,500 minutes the typical 90% interval on a
percentile is 31 to 57 points wide, and a fifth to two fifths of the players in the top decile of a
metric at 900 minutes are not there at season end. The same picture appears in five women's leagues,
in the Indian Super League and in Wyscout data from a different season and provider.

## Data

Everything runs on open data:

- StatsBomb open data, 2015/16 season of La Liga, Premier League, Serie A and Ligue 1 (main set,
  1,517 matches).
- StatsBomb open data for the replications: FA WSL, Frauen-Bundesliga, Liga F, Serie A Women and
  NWSL in 2023/24 (771 matches), the Indian Super League 2021/22 (115), and the FA WSL seasons
  2018/19 to 2020/21 (325) for the season-to-season check.
- The Wyscout public data set (Pappalardo et al., 2019), 2017/18 season of the five big leagues
  (1,826 matches), used to see whether the results survive a change of provider.

The raw event files are not in the repository (they are large and belong to their providers); the
download scripts fetch them in a few minutes. The derived player-match tables and all results are
included.

## Layout

```
scripts/
  download_statsbomb.py          StatsBomb events by data set (--set men2015 | women2023 | wsl-history | isl2021)
  download_wyscout.py            Wyscout public data set from figshare
  build_player_match.py          StatsBomb events -> one row per player and match
  build_player_match_wyscout.py  same table from the Wyscout event model (8 of the 13 metrics)
  uncertainty_analysis.py        reliability curves, k, percentile intervals, shrinkage, ranking stability, mirage rate
  season_validity.py             season-to-season correlations (FA WSL 2018-2021)
  replication_figures.py         comparison of curves and k across data sets
data/derived*/                   player-match tables for each data set
results_*/                       tables and figures for each data set
paper/                           the manuscript
abstract/                        the abstract from the first round
```

## Running it

```bash
pip install -r requirements.txt
```

Main set:

```bash
python scripts/download_statsbomb.py --set men2015 --out data/statsbomb
python scripts/build_player_match.py --data data/statsbomb --out data/derived
python scripts/uncertainty_analysis.py --derived data/derived --out results_uncertainty
```

Women's leagues, Indian Super League and the WSL seasons (shorter seasons, so the grid of sample
sizes stops earlier and the minimum players per cell is lower):

```bash
python scripts/download_statsbomb.py --set women2023 --out data/statsbomb_women2023
python scripts/build_player_match.py --data data/statsbomb_women2023 --out data/derived_women2023
python scripts/uncertainty_analysis.py --derived data/derived_women2023 --out results_women2023 --min-matches-per-competition 100 --min-players 15 --max-n 11

python scripts/download_statsbomb.py --set isl2021 --out data/statsbomb_isl
python scripts/build_player_match.py --data data/statsbomb_isl --out data/derived_isl
python scripts/uncertainty_analysis.py --derived data/derived_isl --out results_isl --min-matches-per-competition 100 --min-players 12 --max-n 10

python scripts/download_statsbomb.py --set wsl-history --out data/statsbomb_wsl
python scripts/build_player_match.py --data data/statsbomb_wsl --out data/derived_wsl
python scripts/season_validity.py --derived data/derived_wsl --out results_wsl_validity --min-players 12 --k-from results_women2023/stabilization_constants.csv
```

Wyscout (three position groups only, because that is all the data set provides):

```bash
python scripts/download_wyscout.py --out data/wyscout
python scripts/build_player_match_wyscout.py --data data/wyscout --out data/derived_wyscout
python scripts/uncertainty_analysis.py --derived data/derived_wyscout --out results_wyscout --pos-groups DF,MD,FW
```

Figures that put the data sets side by side:

```bash
python scripts/replication_figures.py --out results_replication \
    --run "Men 2015/16 (4 leagues)=results_uncertainty" \
    --run "Women 2023/24 (5 leagues)=results_women2023" \
    --run "Indian Super League 2021/22=results_isl" \
    --run "Men 2017/18 (Wyscout)=results_wyscout"
```

Each analysis takes a few minutes on a normal laptop. Random seeds are fixed, so the numbers in the
paper come out exactly.

## A few notes on the method

Rates are ratios of sums (total events over total minutes, times 90), not averages of per-match
rates. Reliability is a split-half correlation: for each metric, position and sample size n, players
with at least 2n appearances are split at random into two disjoint samples of n matches, the rate is
computed on each half, and the Spearman correlation across players is averaged over 40 splits. The
constant k comes from a least-squares fit of r(n) = n / (n + k) to the pooled curve. Percentile
intervals come from resampling a player's matches 400 times. Shrinkage pulls each rate towards the
positional mean with weight 1 − m / (m + k) and re-ranks everyone on the shrunk values. The mirage
rate is the share of players in the top decile of a metric at 900 minutes who are not in the top
decile at season end, among players who went on to play at least 1,500 minutes.

The Wyscout event model has no pressures, no carries and no expected goals, and its key pass tag is
stricter than StatsBomb's while its duel categories are broader. The mapping we used is documented at
the top of `scripts/build_player_match_wyscout.py`; the paper discusses what the differences mean.

## Licences

StatsBomb event data are used under the StatsBomb open data licence (non-commercial). The Wyscout
data set is released under CC BY 4.0; please cite Pappalardo et al. (2019), *A public data set of
spatio-temporal match events in soccer competitions*, Scientific Data 6, 236. The code in this
repository is under the MIT licence.
