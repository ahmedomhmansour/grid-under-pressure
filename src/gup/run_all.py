"""One command for the whole real-data run:

    python -m gup.run_all              # fetch -> clean -> model -> figures
    python -m gup.run_all --skip-fetch # re-run the model on data you already downloaded
"""
from __future__ import annotations

import argparse
import sys

from . import clean, fetch_eia, fetch_weather, figures, pipeline


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-fetch", action="store_true")
    ap.add_argument("--draws", type=int, default=None)
    args = ap.parse_args()
    steps = []
    if not args.skip_fetch:
        steps += [("Downloading EIA-930 load", fetch_eia.main),
                  ("Downloading weather", fetch_weather.main)]
    steps += [("Cleaning", clean.main)]
    for title, fn in steps:
        print(f"\n=== {title} ===")
        sys.argv = [sys.argv[0]]  # sub-steps use their own defaults
        fn()
    print("\n=== Modeling ===")
    sys.argv = [sys.argv[0]] + (["--draws", str(args.draws)] if args.draws else [])
    pipeline.main()
    print("\n=== Figures ===")
    figures.main()
    print("\nDone. Next: `streamlit run app/streamlit_app.py`")


if __name__ == "__main__":
    main()
