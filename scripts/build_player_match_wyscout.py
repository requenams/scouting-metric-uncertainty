"""
Build the player-match table from the Wyscout public data set (Pappalardo et al., 2019):
full event data for the 2017/18 season of the five big European leagues.

The Wyscout event model differs from StatsBomb's. It has no pressures, no carries and no
expected-goals model, and player positions are only given as coarse roles (DF / MD / FW),
so this table supports eight of the thirteen metrics and three position groups:

  shots                 Shot events (penalties are separate Free Kick events)
  key_passes            passes tagged keyPass (302)
  touches_box           passes, shots, duels and touches starting inside the opposition box
  passes / completed    Pass events / tagged accurate (1801)
  progressive_passes    accurate passes gaining >= 10 yards towards goal, ending in the opposition half
  dribbles_completed    ground attacking duels tagged won (703)
  tackles_interceptions ground defending duels + events tagged interception (1401)
  aerials_won           air duels tagged won (703)

Input:  the files of the figshare collection (events_<league>.json, matches_<league>.json, players.json)
Output: <out>/player_match.parquet with the same columns as build_player_match.py (missing
        metrics are NaN so that uncertainty_analysis.py skips them).

Usage:
  python scripts/build_player_match_wyscout.py --data data/wyscout --out data/derived_wyscout
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

LEAGUES = {"England": "Premier League", "Spain": "La Liga", "Italy": "Serie A", "Germany": "Bundesliga", "France": "Ligue 1"}
ROLE_TO_GROUP = {"DF": "DF", "MD": "MD", "FW": "FW", "GK": "GK"}
PROGRESSIVE_UNITS = 100 * 9.144 / 105   # 10 yards on a 105 m pitch, in Wyscout's 0-100 x units (~8.7)
BOX_X, BOX_Y_MIN, BOX_Y_MAX = 84.3, 20.4, 79.6   # 16.5 m deep, 40.3 m wide on a 105 x 68 m pitch
TAG_ACCURATE, TAG_KEYPASS, TAG_WON, TAG_INTERCEPTION, TAG_RED, TAG_SECOND_YELLOW = 1801, 302, 703, 1401, 1701, 1703
ALL_COUNT_COLS = ["shots", "npxg", "goals", "key_passes", "assists", "xa", "touches_box", "passes", "passes_completed",
                  "progressive_passes", "passes_final_third", "crosses", "carries", "progressive_carries", "dribbles",
                  "dribbles_completed", "dispossessed", "pressures", "pressures_att_third", "tackles", "tackles_won",
                  "interceptions", "blocks", "clearances", "ball_recoveries", "dribbled_past", "fouls_committed",
                  "fouls_won", "aerials_won"]


def match_minutes(match: dict, red_cards: dict) -> dict:
    """Minutes played per player from line-ups, substitutions and red cards (events)."""
    end = red_cards.get(("__end__", match["wyId"]), 90.0)
    rows = {}
    for team in match["teamsData"].values():
        f = team.get("formation") or {}
        for p in f.get("lineup", []):
            rows[p["playerId"]] = {"start": 0.0, "end": end, "team": team["teamId"]}
        for s in f.get("substitutions", []) if isinstance(f.get("substitutions"), list) else []:
            if s["playerOut"] in rows:
                rows[s["playerOut"]]["end"] = float(s["minute"])
            rows[s["playerIn"]] = {"start": float(s["minute"]), "end": end, "team": team["teamId"]}
    for pid, minute in red_cards.items():
        if pid in rows and isinstance(pid, int):
            rows[pid]["end"] = min(rows[pid]["end"], minute)
    return {pid: max(0.0, r["end"] - r["start"]) for pid, r in rows.items()}, {pid: r["team"] for pid, r in rows.items()}


def process_league(data: Path, league: str, players: dict) -> pd.DataFrame:
    matches = json.load(open(data / f"matches_{league}.json", encoding="utf-8"))
    events = json.load(open(data / f"events_{league}.json", encoding="utf-8"))
    by_match = {}
    for e in events:
        by_match.setdefault(e["matchId"], []).append(e)

    out = []
    for m in tqdm(matches, desc=league, unit="match"):
        evs = by_match.get(m["wyId"], [])
        if not evs:
            continue
        # match length and red cards from the events
        end_minute = max((45.0 if e["matchPeriod"] == "2H" else 0.0) + e["eventSec"] / 60.0 for e in evs if e["matchPeriod"] in ("1H", "2H"))
        reds = {("__end__", m["wyId"]): end_minute}
        for e in evs:
            tags = {t["id"] for t in e["tags"]}
            if tags & {TAG_RED, TAG_SECOND_YELLOW}:
                reds[e["playerId"]] = (45.0 if e["matchPeriod"] == "2H" else 0.0) + e["eventSec"] / 60.0
        minutes, team_of = match_minutes(m, reds)

        counts = {}
        for e in evs:
            pid = e["playerId"]
            if pid == 0 or pid not in minutes:
                continue
            c = counts.setdefault(pid, {k: 0.0 for k in ["shots", "key_passes", "touches_box", "passes", "passes_completed",
                                                          "progressive_passes", "dribbles", "dribbles_completed", "tackles",
                                                          "interceptions", "aerials_won", "aerials"]})
            tags = {t["id"] for t in e["tags"]}
            name, sub = e["eventName"], e.get("subEventName", "")
            pos = e["positions"]
            x0, y0 = (pos[0]["x"], pos[0]["y"]) if pos else (np.nan, np.nan)
            if name in ("Pass", "Shot", "Duel", "Others on the ball") and x0 >= BOX_X and BOX_Y_MIN <= y0 <= BOX_Y_MAX:
                c["touches_box"] += 1
            if name == "Shot":
                c["shots"] += 1
            elif name == "Pass":
                c["passes"] += 1
                acc = TAG_ACCURATE in tags
                c["passes_completed"] += acc
                c["key_passes"] += TAG_KEYPASS in tags
                if acc and len(pos) > 1:
                    x1 = pos[1]["x"]
                    if x1 - x0 >= PROGRESSIVE_UNITS and x1 >= 50:
                        c["progressive_passes"] += 1
            elif name == "Duel":
                if sub == "Ground attacking duel":
                    c["dribbles"] += 1
                    c["dribbles_completed"] += TAG_WON in tags
                elif sub == "Ground defending duel":
                    c["tackles"] += 1
                elif sub == "Air duel":
                    c["aerials"] += 1
                    c["aerials_won"] += TAG_WON in tags
            if TAG_INTERCEPTION in tags:
                c["interceptions"] += 1

        for pid, mins in minutes.items():
            if mins <= 0:
                continue
            pl = players.get(pid, {})
            row = {"match_id": m["wyId"], "competition": LEAGUES[league], "season": "2017/2018", "match_date": m["dateutc"][:10],
                   "player_id": pid, "player": pl.get("shortName", str(pid)), "team": team_of[pid],
                   "position": pl.get("role", "?"), "pos_group": ROLE_TO_GROUP.get(pl.get("role", "?"), "OTHER"),
                   "minutes": mins, "starter": True}
            row.update({k: np.nan for k in ALL_COUNT_COLS})
            row.update(counts.get(pid, {}))
            row["tackles_won"] = np.nan
            out.append(row)
    return pd.DataFrame(out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/wyscout")
    parser.add_argument("--out", default="data/derived_wyscout")
    args = parser.parse_args()
    data, out = Path(args.data), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    players = {p["wyId"]: {"shortName": p["shortName"], "role": p["role"]["code2"]} for p in json.load(open(data / "players.json", encoding="utf-8"))}
    frames = [process_league(data, lg, players) for lg in LEAGUES]
    pm = pd.concat(frames, ignore_index=True)
    # metrics the Wyscout model cannot provide stay NaN; the ones it can are zero when absent
    for col in ["shots", "key_passes", "touches_box", "passes", "passes_completed", "progressive_passes", "dribbles",
                "dribbles_completed", "tackles", "interceptions", "aerials_won"]:
        pm[col] = pm[col].fillna(0.0)
    pm.to_parquet(out / "player_match.parquet", index=False)
    print(f"\nPlayer-match rows: {len(pm):,}  |  players: {pm['player_id'].nunique():,}  |  matches: {pm['match_id'].nunique():,}")
    print("\nMean minutes per team and match by competition (sanity check: ~990 = 11 players x 90):")
    print((pm.groupby("competition")["minutes"].sum() / pm.groupby("competition")["match_id"].nunique() / 2).round(0))
    print("\nPlayers per position group:")
    print(pm.groupby("pos_group")["player_id"].nunique())


if __name__ == "__main__":
    main()
