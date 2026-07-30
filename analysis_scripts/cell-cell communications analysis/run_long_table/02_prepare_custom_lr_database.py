#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create the two-column custom LR database required by NICHES."
    )
    parser.add_argument(
        "--input-lr-db",
        default="/sdd/bgi/wangzilu/00.motorcortex_stduio/motorcortex_proj/45.DT-CCC/lr_db_human_with_pathway.csv",
        help="Full LR database with at least from/to columns.",
    )
    parser.add_argument(
        "--output-lr-db",
        default="/sdd/bgi/wangzilu/00.motorcortex_stduio/motorcortex_proj/45.DT-CCC/lr_db_human_niches_custom.csv",
        help="Output CSV used as custom_LR_database by NICHES.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_lr_db)
    output_path = Path(args.output_lr_db)

    lr_db = pd.read_csv(input_path)
    if {"from", "to"}.issubset(lr_db.columns):
        custom = lr_db.loc[:, ["from", "to"]]
    else:
        custom = lr_db.iloc[:, :2].copy()
        custom.columns = ["from", "to"]

    custom = custom.dropna().drop_duplicates()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    custom.to_csv(output_path, index=False)

    print(f"input={input_path}")
    print(f"output={output_path}")
    print(f"pairs={len(custom)}")


if __name__ == "__main__":
    main()
