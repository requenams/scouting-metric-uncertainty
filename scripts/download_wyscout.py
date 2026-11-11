"""
Download the Wyscout public data set (Pappalardo et al., 2019) from figshare:
events, matches, players, teams and competitions for the 2017/18 season of the five big leagues.

Usage:
  python scripts/download_wyscout.py --out data/wyscout
"""

import argparse
import urllib.request
import zipfile
from pathlib import Path

FILES = {
    "events.zip": "https://ndownloader.figshare.com/files/14464685",
    "matches.zip": "https://ndownloader.figshare.com/files/14464622",
    "players.json": "https://ndownloader.figshare.com/files/15073721",
    "teams.json": "https://ndownloader.figshare.com/files/15073697",
    "competitions.json": "https://ndownloader.figshare.com/files/15073685",
    "eventid2name.csv": "https://ndownloader.figshare.com/files/21385245",
    "tags2name.csv": "https://ndownloader.figshare.com/files/21385239",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/wyscout")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, url in FILES.items():
        target = out / name
        if target.exists():
            print(f"{name}: already present")
            continue
        print(f"downloading {name} ...")
        urllib.request.urlretrieve(url, target)
        if name.endswith(".zip"):
            with zipfile.ZipFile(target) as z:
                z.extractall(out)
    print("done:", ", ".join(sorted(p.name for p in out.iterdir())))


if __name__ == "__main__":
    main()
