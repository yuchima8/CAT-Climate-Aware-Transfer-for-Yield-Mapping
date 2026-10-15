import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cat.harmonic import SETTINGS, extract


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--region", required=True, choices=["us", "argentina"])
    p.add_argument("--crop", required=True, choices=["soybean", "corn", "winter_wheat"])
    p.add_argument("--input", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    s = SETTINGS[(a.region, a.crop)]
    df = extract(a.input, a.region, a.crop)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    df.to_csv(a.out, index=False)
    print(a.region, a.crop, len(df), "county-years,", min(s["years"]), "-", max(s["years"]))


if __name__ == "__main__":
    main()
