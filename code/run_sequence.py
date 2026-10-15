import argparse
import os
import sys

import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cat import sequence
from cat.sequence import ARCHITECTURES, CONFIGS, RUNNERS, load_argentina, load_us, metrics


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    p = argparse.ArgumentParser()
    p.add_argument("--arch", required=True, choices=list(ARCHITECTURES))
    p.add_argument("--models", nargs="+", default=["local", "direct", "CAT"], choices=list(RUNNERS))
    p.add_argument("--seeds", nargs="+", type=int, default=None)
    p.add_argument("--data", default=os.path.join(here, "..", "data"))
    p.add_argument("--out", default=os.path.join(here, "..", "outputs"))
    p.add_argument("--device", default=None)
    a = p.parse_args()
    if a.device:
        sequence.DEVICE = torch.device(a.device)
    us, arg = load_us(a.data), load_argentina(a.data)
    names = arg["yld"].drop_duplicates("id").set_index("id")[["Province", "District"]]
    runs = []
    for model in a.models:
        cfg = CONFIGS[a.arch][model]
        for seed in a.seeds or [cfg["seed"]]:
            r = RUNNERS[model](a.arch, us, arg, seed, cfg)
            r = r.join(names, on="unit")
            runs.append(pd.DataFrame(dict(crop="soybean", model=f"{model}_{a.arch}", seed=seed, year=r["year"].values,
                                          province=r["Province"].values, district=r["District"].values,
                                          observed=r["observed"].values, predicted=r["predicted"].values)))
            print(a.arch, model, seed, flush=True)
    pred = pd.concat(runs, ignore_index=True)
    os.makedirs(a.out, exist_ok=True)
    pred.to_csv(os.path.join(a.out, f"predictions_{a.arch}_soybean.csv"), index=False)
    metrics(pred).to_csv(os.path.join(a.out, f"metrics_{a.arch}_soybean.csv"), index=False)


if __name__ == "__main__":
    main()
