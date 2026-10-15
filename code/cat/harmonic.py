import glob
import os

import numpy as np
import pandas as pd
import statsmodels.api as sm

FEATURE = "GCVI"

SETTINGS = {
    ("argentina", "soybean"): dict(years=range(2019, 2025), keys=["name_1", "name_2"], gcvi=(0, 20), ndvi=None,
                            doy_max=None, same_year=False, n_refit=0, peak=(100, 250)),
    ("argentina", "corn"): dict(years=range(2019, 2025), keys=["name_1", "name_2"], gcvi=(0, 20), ndvi=None,
                            doy_max=None, same_year=False, n_refit=0, peak=(100, 250)),
    ("argentina", "winter_wheat"): dict(years=range(2018, 2024), keys=["name_1", "name_2"], gcvi=(-1, 6), ndvi=(0, 1),
                            doy_max=None, same_year=False, n_refit=0, peak=(90, None)),
    ("us", "soybean"): dict(years=range(2016, 2025), keys=["GEOID"], gcvi=(0, 20), ndvi=None,
                            doy_max=None, same_year=True, n_refit=5, peak=None),
    ("us", "corn"): dict(years=range(2017, 2025), keys=["GEOID"], gcvi=(0, 20), ndvi=None,
                            doy_max=None, same_year=True, n_refit=5, peak=None),
    ("us", "winter_wheat"): dict(years=range(2016, 2025), keys=["GEOID"], gcvi=(0, 20), ndvi=(None, 1),
                            doy_max=213, same_year=True, n_refit=5, peak=None, read_all=True),
}


def design(doy, period):
    t = 2 * np.pi * doy / period
    return np.column_stack([np.ones_like(doy), np.cos(t), np.sin(t), np.cos(2 * t), np.sin(2 * t),
                            np.cos(3 * t), np.sin(3 * t)])


def fit_harmonics(doy, y, n_refit):
    doy = np.asarray(doy, dtype=float)
    y = np.asarray(y, dtype=float)
    X = design(doy, max(doy))
    model = sm.OLS(y, X).fit()
    observed = y
    for _ in range(n_refit):
        y = np.maximum(model.predict(X), observed)
        model = sm.OLS(y, X).fit()
    return model


def phenology(model, doy, peak_window):
    x = np.arange(0, max(doy) + 31)
    curve = model.predict(design(x, max(doy)))
    i = int(np.argmax(curve))
    if peak_window is not None:
        lo, hi = peak_window
        if i < lo or (hi is not None and i > hi):
            window = curve[lo:] if hi is None else curve[lo:hi + 1]
            i = int(np.argmax(window)) + lo
    a30 = min(i + 30, len(curve) - 1)
    return dict(inter=np.sum(curve[i:i + 31]), inter_b30=np.sum(curve[i - 31:i]), peak=curve[i],
                b30=curve[i - 30], a30=curve[a30])


def clean(df, year, s):
    if s["same_year"]:
        df = df[df["year"] == year]
    if s["doy_max"] is not None:
        df = df[df["DOY"] < s["doy_max"]]
    df = df[(df[FEATURE] > s["gcvi"][0]) & (df[FEATURE] < s["gcvi"][1])].copy()
    if s["ndvi"] is not None:
        lo, hi = s["ndvi"]
        if lo is not None:
            df = df[df["NDVI"] > lo]
        df = df[df["NDVI"] < hi]
    return df


def read_csvs(folder, pattern):
    return pd.concat([pd.read_csv(f) for f in sorted(glob.glob(os.path.join(folder, pattern)))], ignore_index=True)


def extract(folder, region, crop):
    s = SETTINGS[(region, crop)]
    rows = []
    everything = read_csvs(folder, "*.csv") if s.get("read_all") else None
    for year in s["years"]:
        raw = everything if everything is not None else read_csvs(folder, f"*HLS_S2_{year}*.csv")
        df = clean(raw, year, s)
        for key, g in df.groupby(s["keys"], sort=False):
            g = g[s["keys"] + ["DOY", FEATURE, "year"]].dropna()
            if len(g) < 10:
                continue
            doy = np.array(g["DOY"] + (g["year"] - year) * 365)
            doy = doy - np.min(doy)
            model = fit_harmonics(doy, g[FEATURE].to_numpy(), s["n_refit"])
            c = model.params
            key = key if isinstance(key, tuple) else (key,)
            row = dict(zip(s["keys"], key))
            row.update({"year": year, "num": len(g),
                        f"{FEATURE}_const": c[0], f"{FEATURE}_sin1": c[2], f"{FEATURE}_cos1": c[1],
                        f"{FEATURE}_sin2": c[4], f"{FEATURE}_cos2": c[3], f"{FEATURE}_sin3": c[6],
                        f"{FEATURE}_cos3": c[5]})
            row.update({f"{FEATURE}_{k}": v for k, v in phenology(model, doy, s["peak"]).items()})
            rows.append(row)
    out = pd.DataFrame(rows)
    names = {"name_1": "Province", "name_2": "District", "GEOID": "FIPS"}
    return out.rename(columns=names)
