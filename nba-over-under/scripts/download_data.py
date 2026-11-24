#!/usr/bin/env python
"""Re-download the raw data from the original project repository.

The CSVs needed to run the experiments are already bundled in ``data/`` (about
24 MB).  Use this script only if ``data/`` is missing or you want a fresh copy.

Source: https://github.com/abucquet/bank_is_open  (Stanford CS229 project
"The Bank is Open: AI in Sports Gambling" by A. Bucquet and V. Sarukkai).  The
underlying numbers come from Basketball Reference (box scores) and Sports Book
Review Online (odds).
"""
import os
import sys
import urllib.request

BASE = "https://raw.githubusercontent.com/abucquet/bank_is_open/master/data"
GAME_SEASONS = ["2007-2008", "2008-2009", "2009-2010", "2010-2011", "2012-2013",
                "2013-2014", "2014-2015", "2015-2016", "2016-2017", "2017-2018"]
ODDS_SEASONS = GAME_SEASONS[:4] + ["2011-2012"] + GAME_SEASONS[4:] + ["2018-2019"]


def fetch(rel_path: str, dest_root: str) -> None:
    dest = os.path.join(dest_root, rel_path)
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        print(f"  have  {rel_path}")
        return
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    print(f"  fetch {rel_path}")
    urllib.request.urlretrieve(f"{BASE}/{rel_path}", dest)


def main() -> None:
    root = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
    for s in GAME_SEASONS:
        fetch(f"final_game_data/{s}_games_final.csv", root)
    for s in ODDS_SEASONS:
        fetch(f"odds_data_processed/{s}.csv", root)
    print("done ->", os.path.abspath(root))


if __name__ == "__main__":
    main()
