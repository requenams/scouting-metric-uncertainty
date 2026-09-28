# How Many Minutes Until a Number Means Something? Sampling Uncertainty in Data-Driven Scouting

Companion repository for the abstract submitted to the MIT Sloan Sports Analytics Conference 2027
Research Paper Competition.

**Question.** How much of a per-90 positional percentile is signal, how many matches does each
metric need before it repeats itself, and how often is the ranking a scout sees in October the one
he would see in May?

**Data.** 1,517 matches of the 2015/16 La Liga, Premier League, Serie A and Ligue 1 (StatsBomb open
data): 36,052 player appearances of at least 30 minutes, 2,093 player-seasons, 13 per-90 metrics,
seven position groups.

**Headline results.**
- Pass completion is repeatable (split-half r ≥ 0.7) after 6 matches; pressures, progressive passes
  and carries after 7–8; aerial duels after 9; box touches after 10; tackles + interceptions after 14.
  Shots, ball recoveries, dribbles, npxG, key passes and xA never reach 0.7 within 16 matches.
- Between 900 and 1,500 minutes, the median 90% interval on a positional percentile spans 31–57
  points; 72% of percentile estimates in that range have an interval of 30 points or more.
- After five matches only 54–73% of the eventual top-10 is in place; 75% after ten; 82–93% after twenty.

## Repository layout

```
scripts/
  download_statsbomb.py     # StatsBomb open-data events (resumable)
  build_player_match.py     # events -> one row per player and match (minutes + raw counts)
  uncertainty_analysis.py   # reliability curves, percentile intervals, ranking stability, figures
data/derived/player_match.parquet   # derived table (raw events are regenerated, not redistributed)
results_uncertainty/                 # tables (CSV) and figures
abstract/                            # submitted abstract (PDF and LaTeX source)
```

## Reproduce

```bash
pip install -r requirements.txt
python scripts/download_statsbomb.py --out data/statsbomb
python scripts/build_player_match.py --data data/statsbomb --out data/derived
python scripts/uncertainty_analysis.py --derived data/derived --out results_uncertainty
```

The analysis runs in a couple of minutes on a laptop. Only complete seasons (≥ 300 matches in the
open data) are used; appearances shorter than 30 minutes are excluded.

## Method notes

- **Rates** are ratios of sums (total count / total minutes × 90), never means of per-match ratios.
- **Reliability.** For every metric, position group and sample size n ∈ {3, 5, 8, 10, 13, 16, 19},
  players with at least 2n appearances are split at random into two disjoint samples of n matches;
  the Spearman correlation across players between the two samples is averaged over 40 splits and
  pooled across position groups weighted by the number of players. *Minutes-to-stability* is the
  interpolated sample at which reliability reaches 0.7, converted with the mean minutes per appearance.
- **Percentile intervals.** For every player-season with ≥ 900 minutes, matches are resampled with
  replacement 400 times; each resample gives a per-90 value and its percentile against the ≥ 900-minute
  population of the same position group. The 5th–95th percentiles of that distribution form the
  90% interval.
- **Ranking stability.** For each league, position group and metric, the overlap between the top-10
  after the first k matches and the top-10 at season end, restricted in both cases to players with at
  least k appearances.

## Data and licences

Event data: [StatsBomb open data](https://github.com/statsbomb/open-data), used under its
non-commercial licence; please cite StatsBomb if you reuse the derived tables. Code is released
under the MIT licence.
