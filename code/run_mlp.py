import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cat.climate import cat_weights
from cat.data import FEATURES, LOADERS, YEARS
from cat.mlp import YieldMLP, loader, set_seed

WEIGHT_CLIP = {"soybean": None, "corn": None, "winter_wheat": 20}


def column(a):
    return np.expand_dims(np.asarray(a, dtype=float), axis=1)


def fit_predict(x_src, y_src, x_tar, y_tar, seed, batch_size, weights=None):
    set_seed(seed)
    if weights is None:
        model = YieldMLP(x_src.shape[1], weighted=False)
        src = loader([x_src, y_src], batch_size)
    else:
        model = YieldMLP(x_src.shape[1], weighted=True)
        src = loader([x_src, y_src, column(weights)], batch_size)
    monitor = loader([x_src, y_src], len(x_src), shuffle=False)
    return model.fit(src, loader([x_tar, y_tar], batch_size), monitor).predict(x_tar)


def run_cat(us, arg, years, seed, clip):
    x_us, y_us = us[FEATURES].to_numpy(), column(us["yield"])
    out = []
    for year in years:
        a = arg[arg["year"] == year]
        x_a, y_a = a[FEATURES].to_numpy(), column(a["yield"])
        scaler = StandardScaler().fit(np.concatenate((x_us, x_a), axis=0))
        w = cat_weights(us, a, clip)
        pred = fit_predict(scaler.transform(x_us), y_us, scaler.transform(x_a), y_a, seed, 512, w)
        out.append(a.assign(predicted=pred))
    return pd.concat(out)


def run_direct(us, arg, seed):
    x_us, y_us = us[FEATURES].to_numpy(), column(us["yield"])
    scaler = StandardScaler().fit(x_us)
    x_a = scaler.transform(arg[FEATURES].to_numpy())
    pred = fit_predict(scaler.transform(x_us), y_us, x_a, column(arg["yield"]), seed, 512)
    return arg.assign(predicted=pred)


def run_local(arg, years, seed):
    out = []
    for year in years:
        train, test = arg[arg["year"] != year], arg[arg["year"] == year]
        scaler = StandardScaler().fit(train[FEATURES].to_numpy())
        pred = fit_predict(scaler.transform(train[FEATURES].to_numpy()), column(train["yield"]),
                           scaler.transform(test[FEATURES].to_numpy()), column(test["yield"]), seed, 512)
        out.append(test.assign(predicted=pred))
    return pd.concat(out)


def metrics(df):
    rows = []
    for (model, seed), g in df.groupby(["model", "seed"]):
        for year, h in list(g.groupby("year")) + [("all", g)]:
            rows.append(dict(model=model, seed=seed, year=year, R2=r2_score(h["observed"], h["predicted"]),
                             RMSE=np.sqrt(mean_squared_error(h["observed"], h["predicted"]))))
    return pd.DataFrame(rows)


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    p = argparse.ArgumentParser()
    p.add_argument("--crop", required=True, choices=list(LOADERS))
    p.add_argument("--models", nargs="+", default=["local_MLP", "direct_MLP", "CAT_MLP"])
    p.add_argument("--seeds", nargs="+", type=int, default=[940, 941, 942, 943, 944])
    p.add_argument("--data", default=os.path.join(here, "..", "data"))
    p.add_argument("--out", default=os.path.join(here, "..", "outputs"))
    a = p.parse_args()
    us, arg = LOADERS[a.crop](a.data)
    years = YEARS[a.crop]
    runs = []
    for seed in a.seeds:
        for model in a.models:
            if model == "CAT_MLP":
                r = run_cat(us, arg, years, seed, WEIGHT_CLIP[a.crop])
            elif model == "direct_MLP":
                r = run_direct(us, arg, seed)
            else:
                r = run_local(arg, years, seed)
            runs.append(pd.DataFrame(dict(crop=a.crop, model=model, seed=seed, year=r["year"].values,
                                          province=r["Province"].values, district=r["District"].values,
                                          observed=r["yield"].values, predicted=r["predicted"].values)))
            print(a.crop, model, seed, flush=True)
    pred = pd.concat(runs, ignore_index=True)
    os.makedirs(a.out, exist_ok=True)
    pred.to_csv(os.path.join(a.out, f"predictions_MLP_{a.crop}.csv"), index=False)
    metrics(pred).to_csv(os.path.join(a.out, f"metrics_MLP_{a.crop}.csv"), index=False)


if __name__ == "__main__":
    main()
