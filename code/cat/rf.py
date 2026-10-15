import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

FEATURES = ["GCVI_peak", "GCVI_b30", "GCVI_a30", "GCVI_inter", "GCVI_inter_b30"]
KEYS = ["year", "District", "Province"]


@dataclass(frozen=True)
class CropSetup:
    seeds: tuple
    n_estimators: int
    dtype: type
    pooled_scaler: bool
    local_model: bool
    distance_clip: float


SETUPS = {
    "soybean": CropSetup((941, 942, 943, 944, 945), 500, np.float32, True, False, np.inf),
    "corn": CropSetup((930, 931, 932, 933, 934), 500, np.float32, False, True, np.inf),
    "winter_wheat": CropSetup((940, 941, 942, 943, 944), 200, np.float64, False, True, 20.0),
}


def drop_incomplete(df):
    required = FEATURES + ["yield_t_ha", "year", "GDD", "PPT"]
    return df.replace([np.inf, -np.inf], np.nan).dropna(subset=required).copy()


def load_soybean(data_dir):
    us = pd.read_csv(os.path.join(data_dir, "us", "soybean", "US_county_Soy_16_24_5.csv"))
    us = us[us["GCVI_peak"] < 20].copy()
    us = us[us["state"] != 5].copy()
    us["GDD"] = us[["GDD_May", "GDD_Jun", "GDD_Jul", "GDD_Aug", "GDD_Sep"]].sum(axis=1)
    us["PPT"] = us[["PPT_May", "PPT_Jun", "PPT_Jul", "PPT_Aug", "PPT_Sep"]].sum(axis=1)
    us["yield_t_ha"] = pd.to_numeric(us["yield"], errors="coerce")

    folder = os.path.join(data_dir, "argentina", "soybean")
    yields = pd.read_csv(os.path.join(folder, "Argentina_yield_19_24_1st.csv"), float_precision="round_trip")
    gcvi = pd.read_csv(os.path.join(folder, "ARG_HLS_S2_Harm_3rd_GCVI_2019_2024_0.csv"))
    weather = pd.read_csv(os.path.join(folder, "ERA5_ARG_soybean_19_24.csv")).drop_duplicates(subset=KEYS)
    yields = yields[["Province", "District", "year", "yield", "harvested_area"]].copy()
    yields["yield_t_ha"] = pd.to_numeric(yields["yield"], errors="coerce")
    yields["harvested_area"] = pd.to_numeric(yields["harvested_area"], errors="coerce")
    yields = yields.dropna(subset=["yield_t_ha", "harvested_area"])
    yields = yields[(yields["yield_t_ha"] > 1) & (yields["harvested_area"] > 1000)].copy()
    arg = yields.merge(gcvi, on=KEYS, how="inner").merge(weather, on=KEYS, how="left")
    arg["GDD"] = arg[["GDD_12", "GDD_1", "GDD_2", "GDD_3", "GDD_4"]].sum(axis=1)
    arg["PPT"] = arg[["ppt_12", "ppt_1", "ppt_2", "ppt_3", "ppt_4"]].sum(axis=1)
    return drop_incomplete(us), drop_incomplete(arg)


def load_argentina(folder, yield_file, gcvi_file, weather_file, months):
    yields = pd.read_csv(os.path.join(folder, yield_file), float_precision="round_trip")
    gcvi = pd.read_csv(os.path.join(folder, gcvi_file))
    weather = pd.read_csv(os.path.join(folder, weather_file)).drop_duplicates(subset=KEYS)
    gcvi = gcvi[gcvi["GCVI_peak"] < 10].copy()
    yields = yields[["Province", "District", "year", "yield", "harvested_area"]].copy()
    yields["yield"] = pd.to_numeric(yields["yield"], errors="coerce")
    yields = yields.dropna(subset=["yield"])
    yields = yields[(yields["yield"] > 0) & (yields["yield"] < 10) & (yields["harvested_area"] > 1000)]
    yields = yields.drop_duplicates(subset=KEYS)
    arg = gcvi.merge(weather, on=KEYS, how="inner").merge(yields, on=KEYS, how="inner").dropna().copy()
    arg["GDD"] = sum(arg[f"GDD_{m}"] for m in months)
    arg["PPT"] = sum(arg[f"ppt_{m}"] for m in months)
    arg["yield_t_ha"] = arg["yield"]
    return arg


def load_us(folder, gcvi_file, weather_file, yield_file):
    us = pd.read_csv(os.path.join(folder, gcvi_file))
    us = us[us["GCVI_peak"] < 20].copy()
    us = us.merge(pd.read_csv(os.path.join(folder, weather_file)), on=["FIPS", "year"], how="inner")
    return us.merge(pd.read_csv(os.path.join(folder, yield_file), float_precision="round_trip"), on=["FIPS", "year"], how="inner")


def load_corn(data_dir):
    us = load_us(
        os.path.join(data_dir, "us", "corn"),
        "US_HLS_S2_Harm_3rd_GCVI_2016_2024_5.csv",
        "ERA5_US_corn_17_24.csv",
        "corn_US_yield_1999_2024.csv",
    )
    us["GDD"] = sum(us[f"GDD_{m}"] for m in range(5, 10))
    us["PPT"] = sum(us[f"ppt_{m}"] for m in range(5, 10))
    us["yield_t_ha"] = pd.to_numeric(us["yield"], errors="coerce")
    arg = load_argentina(
        os.path.join(data_dir, "argentina", "corn"),
        "ARG_corn_yield_19_24.csv",
        "ARG_corn_HLS_Harm_3rd_GCVI_2019_2024_0.csv",
        "ERA5_ARG_corn_19_24.csv",
        [12, 1, 2, 3, 4],
    )
    return drop_incomplete(us), drop_incomplete(arg)


def load_winter_wheat(data_dir):
    us = load_us(
        os.path.join(data_dir, "us", "winter_wheat"),
        "US_HLS_S2_Harm_3rd_GCVI_2016_2024_5.csv",
        "ERA5_US_winter_wheat_16_23_wET.csv",
        "winter_wheat_US_yield_2000_2024.csv",
    )
    us["yield_t_ha"] = us["yield"]
    arg = load_argentina(
        os.path.join(data_dir, "argentina", "winter_wheat"),
        "ARG_wheat_yield_18_24.csv",
        "ARG_HLS_S2_Harm_3rd_GCVI_2018_2023.csv",
        "ERA5_ARG_winter_wheat_18_23.csv",
        [9, 10, 11, 12],
    )
    return us, arg


LOADERS = {"soybean": load_soybean, "corn": load_corn, "winter_wheat": load_winter_wheat}


def cat_weights(source, target, distance_clip, dtype):
    distances = []
    for column in ["PPT", "GDD"]:
        spread = max(float(target[column].std(ddof=0)), 1e-6)
        distance = np.abs(source[column].to_numpy(dtype=float) - float(target[column].mean())) / spread
        distances.append(np.minimum(distance, distance_clip))
    weights = 1.0 / np.sqrt(np.exp(distances[0]) ** 2 + np.exp(distances[1]) ** 2)
    return (weights / np.max(weights)).astype(dtype)


def fit_predict(X_train, y_train, X_test, seed, n_estimators, sample_weight=None):
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=None,
        min_samples_leaf=1,
        max_features="sqrt",
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(X_train, y_train, sample_weight=sample_weight)
    return model.predict(X_test)


def scaled(X_fit, *arrays, dtype):
    scaler = StandardScaler().fit(X_fit)
    return [scaler.transform(X).astype(dtype) for X in arrays]


def run_seed(us, arg, setup, seed):
    dtype = setup.dtype
    X_us_raw = us[FEATURES].to_numpy(dtype=dtype)
    y_us = us["yield_t_ha"].to_numpy(dtype=dtype)
    frames = []
    for year in sorted(arg["year"].unique()):
        target = arg[arg["year"] == year]
        X_target_raw = target[FEATURES].to_numpy(dtype=dtype)
        scaler_fit = np.vstack([X_us_raw, X_target_raw]) if setup.pooled_scaler else X_us_raw
        X_us, X_target = scaled(scaler_fit, X_us_raw, X_target_raw, dtype=dtype)
        predictions = {
            "direct_RF": fit_predict(X_us, y_us, X_target, seed, setup.n_estimators),
            "CAT_RF": fit_predict(
                X_us, y_us, X_target, seed, setup.n_estimators,
                sample_weight=cat_weights(us, target, setup.distance_clip, dtype),
            ),
        }
        if setup.local_model:
            train = arg[arg["year"] != year]
            X_train_raw = train[FEATURES].to_numpy(dtype=dtype)
            X_train, X_test = scaled(X_train_raw, X_train_raw, X_target_raw, dtype=dtype)
            y_train = train["yield_t_ha"].to_numpy(dtype=dtype)
            predictions["local_RF"] = fit_predict(X_train, y_train, X_test, seed, setup.n_estimators)
        for model, predicted in predictions.items():
            frames.append(pd.DataFrame({
                "model": model,
                "seed": seed,
                "year": int(year),
                "province": target["Province"].to_numpy(),
                "district": target["District"].to_numpy(),
                "observed": target["yield_t_ha"].to_numpy(dtype=float),
                "predicted": predicted.astype(float),
            }))
    return pd.concat(frames, ignore_index=True)


def score(observed, predicted):
    return {
        "R2": r2_score(observed, predicted),
        "RMSE": float(np.sqrt(mean_squared_error(observed, predicted))),
    }


def summarize(predictions):
    rows = []
    for (model, seed), group in predictions.groupby(["model", "seed"], sort=False):
        for year, subset in group.groupby("year"):
            rows.append({"model": model, "seed": seed, "year": str(year), **score(subset["observed"], subset["predicted"])})
        rows.append({"model": model, "seed": seed, "year": "all", **score(group["observed"], group["predicted"])})
    return pd.DataFrame(rows)


LOCAL_SOYBEAN_SEEDS = (940, 941, 942, 943, 944)


def run_local_soybean(data_dir, seed):
    arg = pd.read_csv(os.path.join(data_dir, "argentina", "soybean", "ARG_soybean_inputs_MLP.csv"))
    arg = arg.sort_values(["Province", "District", "year"], kind="stable")
    frames = []
    for year in sorted(arg["year"].unique()):
        train, test = arg[arg["year"] != year], arg[arg["year"] == year]
        model = RandomForestRegressor(n_estimators=100, random_state=seed)
        model.fit(train[FEATURES].to_numpy(), train["yield"].to_numpy())
        frames.append(pd.DataFrame({
            "model": "local_RF",
            "seed": seed,
            "year": int(year),
            "province": test["Province"].to_numpy(),
            "district": test["District"].to_numpy(),
            "observed": test["yield"].to_numpy(dtype=float),
            "predicted": model.predict(test[FEATURES].to_numpy()),
        }))
    return pd.concat(frames, ignore_index=True)


def run(crop, data_dir, seeds=None):
    setup = SETUPS[crop]
    us, arg = LOADERS[crop](data_dir)
    frames = [run_seed(us, arg, setup, seed) for seed in (seeds or setup.seeds)]
    if crop == "soybean":
        frames += [run_local_soybean(data_dir, seed) for seed in (seeds or LOCAL_SOYBEAN_SEEDS)]
    predictions = pd.concat(frames, ignore_index=True)
    predictions.insert(0, "crop", crop)
    return predictions, summarize(predictions)
