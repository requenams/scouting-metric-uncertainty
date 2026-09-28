"""
Build the player-match table from StatsBomb open-data events.

Input:  data/statsbomb/matches.parquet and data/statsbomb/events/<match_id>.parquet
Output: data/derived/player_match.parquet  (one row per player and match)

Each row holds minutes played, position and raw counts for ~29 metrics.
Per-90 values and percentiles are computed later, at aggregation time.

Usage:
  python scripts/build_player_match.py --data data/statsbomb --out data/derived
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

# StatsBomb pitch: 120 x 80 yards, attacking towards x = 120.
BOX_X, BOX_Y_MIN, BOX_Y_MAX = 102, 18, 62
PROGRESSIVE_YARDS = 10
FINAL_THIRD_X = 80

RED_CARDS = {"Red Card", "Second Yellow"}
TACKLE_WON = {"Won", "Success", "Success In Play", "Success Out"}
INTERCEPTION_WON = {"Won", "Success", "Success In Play", "Success Out"}
IN_BOX_EVENT_TYPES = {"Pass", "Shot", "Carry", "Dribble", "Ball Receipt*"}

# StatsBomb positions grouped into 8 families for the percentiles.
POSITION_GROUPS = {
    "Goalkeeper": "GK",
    "Right Back": "FB", "Left Back": "FB", "Right Wing Back": "FB", "Left Wing Back": "FB",
    "Center Back": "CB", "Right Center Back": "CB", "Left Center Back": "CB",
    "Center Defensive Midfield": "DM", "Right Defensive Midfield": "DM", "Left Defensive Midfield": "DM",
    "Center Midfield": "CM", "Right Center Midfield": "CM", "Left Center Midfield": "CM",
    "Right Midfield": "CM", "Left Midfield": "CM",
    "Center Attacking Midfield": "AM", "Right Attacking Midfield": "AM", "Left Attacking Midfield": "AM",
    "Right Wing": "W", "Left Wing": "W",
    "Center Forward": "FW", "Right Center Forward": "FW", "Left Center Forward": "FW", "Secondary Striker": "FW",
}


def parse_loc(value):
    """location / *_end_location are stored as JSON strings after download."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if isinstance(value, str):
        return json.loads(value)
    return list(value)


def col(df: pd.DataFrame, name: str) -> pd.Series:
    """Optional column: returns NaN if it is absent in this match."""
    return df[name] if name in df.columns else pd.Series(np.nan, index=df.index, dtype=object)


def minutes_played(ev: pd.DataFrame) -> pd.DataFrame:
    """Minutes played and starting position per player, from line-ups,
    substitutions, red cards and the final minute of the match."""
    end_minute = float(ev.loc[ev["period"] <= 2, "minute"].max())
    rows = {}

    for _, e in ev[ev["type"] == "Starting XI"].iterrows():
        tactics = e["tactics"]
        tactics = json.loads(tactics) if isinstance(tactics, str) else tactics
        for entry in tactics["lineup"]:
            pid = entry["player"]["id"]
            rows[pid] = {
                "player_id": pid, "player": entry["player"]["name"], "team": e["team"],
                "position": entry["position"]["name"], "start": 0.0, "end": end_minute, "starter": True,
            }

    subs = ev[ev["type"] == "Substitution"]
    for _, e in subs.iterrows():
        minute = float(e["minute"])
        if e["player_id"] in rows:
            rows[e["player_id"]]["end"] = minute
        rid = e.get("substitution_replacement_id")
        if pd.notna(rid):
            rid = int(rid)
            rows[rid] = {
                "player_id": rid, "player": e["substitution_replacement"], "team": e["team"],
                "position": e.get("position", None), "start": minute, "end": end_minute, "starter": False,
            }

    for card_col in ("foul_committed_card", "bad_behaviour_card"):
        if card_col in ev.columns:
            reds = ev[ev[card_col].isin(RED_CARDS)]
            for _, e in reds.iterrows():
                if e["player_id"] in rows:
                    rows[e["player_id"]]["end"] = min(rows[e["player_id"]]["end"], float(e["minute"]))

    out = pd.DataFrame(rows.values())
    out["minutes"] = (out["end"] - out["start"]).clip(lower=0)
    return out.drop(columns=["start", "end"])


def player_positions(ev: pd.DataFrame) -> pd.Series:
    """Most frequent position of each player across their events (more robust
    than the line-up position when there are tactical changes)."""
    p = ev.dropna(subset=["player_id", "position"])
    return p.groupby("player_id")["position"].agg(lambda s: s.mode().iloc[0])


def metrics(ev: pd.DataFrame) -> pd.DataFrame:
    ev = ev[ev["player_id"].notna()].copy()
    ev["player_id"] = ev["player_id"].astype(int)
    loc = ev["location"].map(parse_loc)
    ev["x"] = loc.map(lambda l: l[0] if l else np.nan)
    ev["y"] = loc.map(lambda l: l[1] if l else np.nan)

    t = ev["type"]
    is_pass, is_shot, is_carry, is_dribble = t == "Pass", t == "Shot", t == "Carry", t == "Dribble"

    pass_end = col(ev, "pass_end_location").map(parse_loc)
    pass_end_x = pass_end.map(lambda l: l[0] if l else np.nan)
    pass_completed = is_pass & col(ev, "pass_outcome").isna()
    pass_open_play = is_pass & ~col(ev, "pass_type").isin(["Corner", "Free Kick", "Throw-in", "Goal Kick", "Kick Off"])

    carry_end = col(ev, "carry_end_location").map(parse_loc)
    carry_end_x = carry_end.map(lambda l: l[0] if l else np.nan)

    shot_xg = pd.to_numeric(col(ev, "shot_statsbomb_xg"), errors="coerce").fillna(0.0)
    is_pen = col(ev, "shot_type") == "Penalty"
    is_goal = col(ev, "shot_outcome") == "Goal"

    in_box = (ev["x"] >= BOX_X) & ev["y"].between(BOX_Y_MIN, BOX_Y_MAX)
    att_third = ev["x"] >= FINAL_THIRD_X

    # xA: the xG of the shot credited to the passer of the key pass.
    key_pass_xg = ev.loc[is_shot & col(ev, "shot_key_pass_id").notna(), ["shot_key_pass_id", "shot_statsbomb_xg"]]
    xa_by_pass = key_pass_xg.set_index("shot_key_pass_id")["shot_statsbomb_xg"] if len(key_pass_xg) else pd.Series(dtype=float)
    ev["xa"] = ev["id"].map(xa_by_pass).fillna(0.0)

    duel_type = col(ev, "duel_type")
    duel_outcome = col(ev, "duel_outcome")
    is_tackle = (t == "Duel") & (duel_type == "Tackle")

    m = pd.DataFrame({
        "player_id": ev["player_id"],
        # Attacking
        "shots": (is_shot & ~is_pen).astype(int),
        "npxg": shot_xg.where(is_shot & ~is_pen, 0.0),
        "goals": (is_shot & is_goal & ~is_pen).astype(int),
        "key_passes": (is_pass & col(ev, "pass_shot_assist").fillna(False).astype(bool)).astype(int),
        "assists": (is_pass & col(ev, "pass_goal_assist").fillna(False).astype(bool)).astype(int),
        "xa": ev["xa"],
        "touches_box": (in_box & t.isin(IN_BOX_EVENT_TYPES)).astype(int),
        # Passing and progression
        "passes": is_pass.astype(int),
        "passes_completed": pass_completed.astype(int),
        "progressive_passes": (pass_completed & pass_open_play & ((pass_end_x - ev["x"]) >= PROGRESSIVE_YARDS) & (pass_end_x >= 60)).astype(int),
        "passes_final_third": (pass_completed & (pass_end_x >= FINAL_THIRD_X) & (ev["x"] < FINAL_THIRD_X)).astype(int),
        "crosses": (is_pass & col(ev, "pass_cross").fillna(False).astype(bool)).astype(int),
        "carries": is_carry.astype(int),
        "progressive_carries": (is_carry & ((carry_end_x - ev["x"]) >= PROGRESSIVE_YARDS) & (carry_end_x >= 60)).astype(int),
        "dribbles": is_dribble.astype(int),
        "dribbles_completed": (is_dribble & (col(ev, "dribble_outcome") == "Complete")).astype(int),
        "dispossessed": (t == "Dispossessed").astype(int),
        # Defending
        "pressures": (t == "Pressure").astype(int),
        "pressures_att_third": ((t == "Pressure") & att_third).astype(int),
        "tackles": is_tackle.astype(int),
        "tackles_won": (is_tackle & duel_outcome.isin(TACKLE_WON)).astype(int),
        "interceptions": ((t == "Interception") & col(ev, "interception_outcome").isin(INTERCEPTION_WON)).astype(int),
        "blocks": (t == "Block").astype(int),
        "clearances": (t == "Clearance").astype(int),
        "ball_recoveries": ((t == "Ball Recovery") & ~col(ev, "ball_recovery_recovery_failure").fillna(False).astype(bool)).astype(int),
        "dribbled_past": (t == "Dribbled Past").astype(int),
        "fouls_committed": (t == "Foul Committed").astype(int),
        "fouls_won": (t == "Foul Won").astype(int),
        "aerials_won": (col(ev, "pass_aerial_won").fillna(False).astype(bool) | col(ev, "clearance_aerial_won").fillna(False).astype(bool) | col(ev, "shot_aerial_won").fillna(False).astype(bool)).astype(int),
    })
    return m.groupby("player_id").sum()


def process_match(path: Path, match_meta: dict) -> pd.DataFrame:
    ev = pd.read_parquet(path)
    base = minutes_played(ev)
    pos = player_positions(ev)
    base["position"] = base["player_id"].map(pos).fillna(base["position"])
    base["pos_group"] = base["position"].map(POSITION_GROUPS).fillna("OTHER")
    m = metrics(ev)
    df = base.merge(m, left_on="player_id", right_index=True, how="left").fillna(0)
    for k, v in match_meta.items():
        df[k] = v
    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/statsbomb")
    parser.add_argument("--out", default="data/derived")
    args = parser.parse_args()
    data, out = Path(args.data), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    matches = pd.read_parquet(data / "matches.parquet")
    frames, failed = [], []
    for _, mt in tqdm(matches.iterrows(), total=len(matches), unit="match"):
        path = data / "events" / f"{mt['match_id']}.parquet"
        if not path.exists():
            failed.append((mt["match_id"], "no events file"))
            continue
        meta = {
            "match_id": mt["match_id"], "competition": mt["competition_name"], "season": mt["season_name"],
            "match_date": mt["match_date"], "home_team": mt["home_team"], "away_team": mt["away_team"],
        }
        try:
            frames.append(process_match(path, meta))
        except Exception as exc:  # noqa: BLE001
            failed.append((mt["match_id"], repr(exc)))

    pm = pd.concat(frames, ignore_index=True)
    pm = pm[pm["minutes"] > 0]
    pm.to_parquet(out / "player_match.parquet", index=False)

    print(f"\nPlayer-match rows: {len(pm):,}  |  players: {pm['player_id'].nunique():,}  |  matches: {pm['match_id'].nunique():,}")
    print(f"Failures: {len(failed)}")
    if failed:
        pd.DataFrame(failed, columns=["match_id", "error"]).to_csv(out / "build_failed.csv", index=False)
    print("\nMean minutes per team and match by competition (sanity check: ~990 = 11 players x 90):")
    print((pm.groupby(["competition", "season"])["minutes"].sum() / pm.groupby(["competition", "season"])["match_id"].nunique() / 2).round(0))
    print("\nPlayers per position group:")
    print(pm.groupby("pos_group")["player_id"].nunique())


if __name__ == "__main__":
    main()
