"""
Download StatsBomb open-data events for the study.

Three data sets, selected with --set (each goes to its own folder):
  men2015       2015/16 La Liga, Premier League, Serie A, Ligue 1 (+ partial 1. Bundesliga)
  women2023     2023/24 FA Women's Super League, Frauen Bundesliga, Liga F, Serie A Women, NWSL 2023
  wsl-history   FA Women's Super League 2018/19, 2019/20, 2020/21 (season-to-season validity)
  isl2021       Indian Super League 2021/22 (secondary replication)

Usage:
  pip install statsbombpy pandas pyarrow tqdm
  python scripts/download_statsbomb.py --set men2015 --out data/statsbomb
  python scripts/download_statsbomb.py --set women2023 --out data/statsbomb_women2023
  python scripts/download_statsbomb.py --set wsl-history --out data/statsbomb_wsl
  python scripts/download_statsbomb.py --set isl2021 --out data/statsbomb_isl

The download is resumable: matches already on disk are skipped.
"""

import argparse
import json
import time
from pathlib import Path

import pandas as pd
from statsbombpy import sb
from tqdm import tqdm

SETS = {
    "men2015": {
        "2015/2016": ["La Liga", "Premier League", "Serie A", "1. Bundesliga", "Ligue 1"],
        "2023/2024": ["1. Bundesliga"],
    },
    "women2023": {
        "2023/2024": ["FA Women's Super League", "Frauen Bundesliga", "Liga F", "Serie A Women"],
        "2023": ["NWSL"],
    },
    "wsl-history": {
        "2018/2019": ["FA Women's Super League"],
        "2019/2020": ["FA Women's Super League"],
        "2020/2021": ["FA Women's Super League"],
    },
    "isl2021": {
        "2021/2022": ["Indian Super league"],
    },
}
RETRIES = 3


def jsonify_complex_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Make a DataFrame safe for Parquet.

    - Nested lists/dicts are stored as JSON strings.
    - Object columns with mixed types (e.g. numeric IDs next to
      "1001017, 226" when a match had two managers) are stored as text.
      Nulls are kept as nulls.
    """
    df = df.copy()
    for col in df.columns:
        if df[col].dtype != object and str(df[col].dtype) != "str":
            continue
        values = df[col]
        if values.map(lambda v: isinstance(v, (list, dict))).any():
            values = values.map(lambda v: json.dumps(v) if isinstance(v, (list, dict)) else v)
        types = {type(v) for v in values if v is not None and not (isinstance(v, float) and pd.isna(v))}
        if len(types) > 1:
            values = values.map(lambda v: None if v is None or (isinstance(v, float) and pd.isna(v)) else str(v))
        df[col] = values
    return df


def fetch_with_retry(fn, *args, **kwargs):
    for attempt in range(1, RETRIES + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001
            if attempt == RETRIES:
                raise
            wait = 5 * attempt
            print(f"  error ({exc}); retry {attempt}/{RETRIES} in {wait}s")
            time.sleep(wait)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--set", default="men2015", choices=sorted(SETS), help="which data set to download")
    parser.add_argument("--out", default="data/statsbomb", help="output folder")
    args = parser.parse_args()
    targets = SETS[args.set]
    out = Path(args.out)
    (out / "events").mkdir(parents=True, exist_ok=True)

    comps = sb.competitions()
    selected = comps[
        comps.apply(lambda r: r["competition_name"] in targets.get(r["season_name"], []), axis=1)
    ]
    if selected.empty:
        raise SystemExit("Target competitions not found; check SETS.")
    print("Selected competitions:")
    print(selected[["competition_id", "season_id", "competition_name", "season_name"]].to_string(index=False))

    all_matches = []
    for _, comp in selected.iterrows():
        matches = fetch_with_retry(sb.matches, competition_id=comp["competition_id"], season_id=comp["season_id"])
        matches["competition_name"] = comp["competition_name"]
        matches["season_name"] = comp["season_name"]
        all_matches.append(matches)
    matches = pd.concat(all_matches, ignore_index=True)
    jsonify_complex_columns(matches).to_parquet(out / "matches.parquet", index=False)
    print(f"\n{len(matches)} matches in total. Downloading events...\n")

    done, failed = 0, []
    for mid in tqdm(matches["match_id"].tolist(), unit="match"):
        target = out / "events" / f"{mid}.parquet"
        if target.exists():
            continue
        try:
            events = fetch_with_retry(sb.events, match_id=mid)
            jsonify_complex_columns(events).to_parquet(target, index=False)
            done += 1
            time.sleep(0.2)
        except Exception as exc:  # noqa: BLE001
            failed.append((mid, str(exc)))
            print(f"  FAILED match {mid}: {exc}")

    print(f"\nDownloaded now: {done}. Failed: {len(failed)}.")
    if failed:
        pd.DataFrame(failed, columns=["match_id", "error"]).to_csv(out / "failed.csv", index=False)
        print("Failures listed in failed.csv; re-run to retry them.")


if __name__ == "__main__":
    main()
