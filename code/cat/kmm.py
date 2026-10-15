import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.optimize import minimize
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics.pairwise import pairwise_distances, rbf_kernel
from sklearn.preprocessing import StandardScaler
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, TensorDataset

LANDSAT_FEATURES = ["GCVI_peak", "GCVI_b30", "GCVI_a30", "GCVI_inter", "GCVI_inter_b30"]
CLIMATE_FEATURES = ["GDD", "PPT"]
MODEL_FEATURES = LANDSAT_FEATURES
WEIGHT_FEATURES = LANDSAT_FEATURES + CLIMATE_FEATURES
KEYS = ["year", "District", "Province"]
WEIGHT_CLIP = (0.05, 20.0)
KMM_BOUND = 20.0
KMM_SUM_EPS = 0.05

CROPS = {
    "soybean": dict(
        seeds=[941, 942, 943, 944, 945],
        target_years=[2019, 2020, 2021, 2022, 2023, 2024],
        kmm_max_samples=600,
        kmm_maxiter=200,
        gamma_max_samples=1200,
        rf_min_samples_leaf=2,
    ),
    "corn": dict(
        seeds=[930, 931, 932, 933, 934],
        target_years=[2019, 2020, 2021, 2022, 2023, 2024],
        kmm_max_samples=500,
        kmm_maxiter=120,
        gamma_max_samples=1000,
        rf_min_samples_leaf=1,
    ),
    "winter_wheat": dict(
        seeds=[940, 941, 942, 943, 944],
        target_years=[2018, 2019, 2020, 2021, 2022, 2023],
        kmm_max_samples=300,
        kmm_maxiter=120,
        gamma_max_samples=1000,
        rf_min_samples_leaf=1,
    ),
}

MLP_PARAMS = dict(
    num_shared_feature=64,
    learning_rate=0.001,
    weight_decay=0.01,
    dropout_ratio=0.3,
    epochs=500,
    batch_size=512,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def finalize(df):
    df = df.replace([np.inf, -np.inf], np.nan)
    return df.dropna(subset=MODEL_FEATURES + WEIGHT_FEATURES + ["yield_t_ha", "year"]).copy()


def load_us_soybean(data_dir):
    df = pd.read_csv(os.path.join(data_dir, "us", "soybean", "US_county_Soy_16_24_5.csv"))
    df = df[df["GCVI_peak"] < 20].copy()
    df = df[df["state"] != 5].copy()
    df["GDD"] = df[["GDD_May", "GDD_Jun", "GDD_Jul", "GDD_Aug", "GDD_Sep"]].sum(axis=1)
    df["PPT"] = df[["PPT_May", "PPT_Jun", "PPT_Jul", "PPT_Aug", "PPT_Sep"]].sum(axis=1)
    df["yield_t_ha"] = df["yield"].astype(float)
    return df.dropna(subset=MODEL_FEATURES + WEIGHT_FEATURES + ["yield_t_ha"]).copy()


def load_arg_soybean(data_dir):
    folder = os.path.join(data_dir, "argentina", "soybean")
    yld = pd.read_csv(os.path.join(folder, "Argentina_yield_19_24_1st.csv"), float_precision="round_trip")
    landsat = pd.read_csv(os.path.join(folder, "ARG_HLS_S2_Harm_3rd_GCVI_2019_2024_0.csv"))
    era5 = pd.read_csv(os.path.join(folder, "ERA5_ARG_soybean_19_24.csv"))
    era5 = era5.drop_duplicates(subset=KEYS)
    yld = yld[["Province", "District", "year", "yield", "harvested_area"]].copy()
    yld["yield"] = pd.to_numeric(yld["yield"], errors="coerce")
    yld = yld.dropna(subset=["yield"])
    yld = yld[(yld["yield"] > 1) & (yld["harvested_area"] > 1000)].copy()
    df = yld.merge(landsat, on=KEYS)
    df = df.merge(era5, on=KEYS, how="left")
    df["GDD"] = df[["GDD_12", "GDD_1", "GDD_2", "GDD_3", "GDD_4"]].sum(axis=1)
    df["PPT"] = df[["ppt_12", "ppt_1", "ppt_2", "ppt_3", "ppt_4"]].sum(axis=1)
    df = df.rename(columns={"yield": "yield_t_ha"})
    return df.dropna(subset=MODEL_FEATURES + WEIGHT_FEATURES + ["yield_t_ha"]).copy()


def load_us_hls(data_dir, crop, era5_file, yield_file, gdd_cols, ppt_cols):
    folder = os.path.join(data_dir, "us", crop)
    df = pd.read_csv(os.path.join(folder, "US_HLS_S2_Harm_3rd_GCVI_2016_2024_5.csv"))
    df = df[df["GCVI_peak"] < 20].copy()
    era5 = pd.read_csv(os.path.join(folder, era5_file))
    yld = pd.read_csv(os.path.join(folder, yield_file), float_precision="round_trip")
    df = df.merge(era5, on=["FIPS", "year"], how="inner")
    df = df.merge(yld, on=["FIPS", "year"], how="inner")
    df["GDD"] = df[gdd_cols].sum(axis=1)
    df["PPT"] = df[ppt_cols].sum(axis=1)
    df["yield_t_ha"] = pd.to_numeric(df["yield"], errors="coerce")
    return finalize(df)


def load_arg_hls(data_dir, crop, yield_file, landsat_file, era5_file, gdd_cols, ppt_cols):
    folder = os.path.join(data_dir, "argentina", crop)
    yld = pd.read_csv(os.path.join(folder, yield_file), float_precision="round_trip")
    landsat = pd.read_csv(os.path.join(folder, landsat_file))
    era5 = pd.read_csv(os.path.join(folder, era5_file))
    era5 = era5.drop_duplicates(subset=KEYS)
    landsat = landsat[landsat["GCVI_peak"] < 10].copy()
    yld = yld[["Province", "District", "year", "yield", "harvested_area"]].copy()
    yld["yield"] = pd.to_numeric(yld["yield"], errors="coerce")
    yld = yld.dropna(subset=["yield"])
    yld = yld[(yld["yield"] > 0) & (yld["yield"] < 10) & (yld["harvested_area"] > 1000)].copy()
    yld = yld.drop_duplicates(subset=KEYS)
    df = landsat.merge(era5, on=KEYS, how="inner")
    df = df.merge(yld, on=KEYS, how="inner")
    df["GDD"] = df[gdd_cols].sum(axis=1)
    df["PPT"] = df[ppt_cols].sum(axis=1)
    df["yield_t_ha"] = df["yield"]
    return finalize(df)


def load_data(crop, data_dir):
    if crop == "soybean":
        return load_us_soybean(data_dir), load_arg_soybean(data_dir)
    if crop == "corn":
        us = load_us_hls(
            data_dir, "corn", "ERA5_US_corn_17_24.csv", "corn_US_yield_1999_2024.csv",
            ["GDD_5", "GDD_6", "GDD_7", "GDD_8", "GDD_9"], ["ppt_5", "ppt_6", "ppt_7", "ppt_8", "ppt_9"],
        )
        arg = load_arg_hls(
            data_dir, "corn", "ARG_corn_yield_19_24.csv", "ARG_corn_HLS_Harm_3rd_GCVI_2019_2024_0.csv",
            "ERA5_ARG_corn_19_24.csv",
            ["GDD_12", "GDD_1", "GDD_2", "GDD_3", "GDD_4"], ["ppt_12", "ppt_1", "ppt_2", "ppt_3", "ppt_4"],
        )
        return us, arg
    us = load_us_hls(
        data_dir, "winter_wheat", "ERA5_US_winter_wheat_16_23_wET.csv", "winter_wheat_US_yield_2000_2024.csv",
        ["GDD_3", "GDD_4", "GDD_5", "GDD_6"], ["PPT_3", "PPT_4", "PPT_5", "PPT_6"],
    )
    arg = load_arg_hls(
        data_dir, "winter_wheat", "ARG_wheat_yield_18_24.csv", "ARG_HLS_S2_Harm_3rd_GCVI_2018_2023.csv",
        "ERA5_ARG_winter_wheat_18_23.csv",
        ["GDD_9", "GDD_10", "GDD_11", "GDD_12"], ["ppt_9", "ppt_10", "ppt_11", "ppt_12"],
    )
    return us, arg


def normalize_weights(w, clip=WEIGHT_CLIP):
    w = np.asarray(w, dtype=float).reshape(-1)
    w = np.nan_to_num(w, nan=1.0, posinf=clip[1], neginf=clip[0])
    w = np.clip(w, clip[0], clip[1])
    mean_w = np.mean(w)
    if not np.isfinite(mean_w) or mean_w <= 0:
        return np.ones_like(w)
    return w / mean_w


def median_gamma(X, Y, max_samples, random_state):
    rng = np.random.default_rng(random_state)
    Z = np.vstack([X, Y])
    if len(Z) > max_samples:
        Z = Z[rng.choice(len(Z), size=max_samples, replace=False)]
    d2 = pairwise_distances(Z, metric="sqeuclidean")
    vals = d2[np.triu_indices_from(d2, k=1)]
    vals = vals[vals > 0]
    if len(vals) == 0:
        return 1.0
    return 1.0 / (2.0 * np.median(vals) + 1e-12)


def kmm_weights(X_source, X_target, max_samples, maxiter, gamma_max_samples, random_state):
    rng = np.random.default_rng(random_state)
    X_source = np.asarray(X_source, dtype=float)
    X_target = np.asarray(X_target, dtype=float)
    gamma = median_gamma(X_source, X_target, gamma_max_samples, random_state)

    n_source = min(max_samples, len(X_source))
    n_target = min(max_samples, len(X_target))
    Xs = X_source[rng.choice(len(X_source), size=n_source, replace=False)]
    Xt = X_target[rng.choice(len(X_target), size=n_target, replace=False)]

    K = rbf_kernel(Xs, Xs, gamma=gamma) + 1e-6 * np.eye(n_source)
    kappa = (n_source / n_target) * rbf_kernel(Xs, Xt, gamma=gamma).sum(axis=1)

    lower_sum = n_source * (1.0 - KMM_SUM_EPS)
    upper_sum = n_source * (1.0 + KMM_SUM_EPS)
    constraints = [
        {"type": "ineq", "fun": lambda b: b.sum() - lower_sum, "jac": lambda b: np.ones_like(b)},
        {"type": "ineq", "fun": lambda b: upper_sum - b.sum(), "jac": lambda b: -np.ones_like(b)},
    ]
    result = minimize(
        lambda b: 0.5 * b @ K @ b - kappa @ b,
        np.ones(n_source),
        method="SLSQP",
        jac=lambda b: K @ b - kappa,
        bounds=[(0.0, KMM_BOUND)] * n_source,
        constraints=constraints,
        options={"maxiter": maxiter, "ftol": 1e-6, "disp": False},
    )
    beta = normalize_weights(result.x)
    K_all = rbf_kernel(X_source, Xs, gamma=gamma)
    raw_weights = (K_all @ beta) / np.clip(K_all.sum(axis=1), 1e-12, None)
    return normalize_weights(raw_weights)


class CATEncoder(nn.Module):
    def __init__(self, num_predictors, num_shared_feature, dropout_ratio):
        super().__init__()
        self.dp = nn.Dropout(p=dropout_ratio)
        self.ac = nn.ReLU()
        self.fc_1 = nn.Linear(num_predictors, 128)
        self.fc_2 = nn.Linear(128, 128)
        self.fc_3 = nn.Linear(128, 64)
        self.fc_4 = nn.Linear(64, 64)
        self.fc_5 = nn.Linear(64, num_shared_feature)

    def forward(self, x):
        x = self.ac(self.dp(self.fc_1(x)))
        x = self.ac(self.dp(self.fc_2(x)))
        x = self.ac(self.dp(self.fc_3(x)))
        x = self.ac(self.dp(self.fc_4(x)))
        return self.fc_5(x)


class CATPredictor(nn.Module):
    def __init__(self, num_shared_feature, dropout_ratio):
        super().__init__()
        self.dp = nn.Dropout(p=dropout_ratio)
        self.ac = nn.ReLU()
        self.r_fc1 = nn.Linear(num_shared_feature, 64)
        self.r_fc2 = nn.Linear(64, 64)
        self.r_fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = self.ac(self.dp(self.r_fc1(x)))
        x = self.ac(self.dp(self.r_fc2(x)))
        return self.r_fc3(x)


def fit_predict_mlp(Xs, ys, Xt, weights, seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    w = normalize_weights(weights).astype(np.float32).reshape(-1, 1)
    dataset = TensorDataset(
        torch.tensor(np.asarray(Xs, dtype=np.float32)),
        torch.tensor(np.asarray(ys, dtype=np.float32).reshape(-1, 1)),
        torch.tensor(w),
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    loader = DataLoader(dataset, batch_size=MLP_PARAMS["batch_size"], shuffle=True, generator=generator)

    encoder = CATEncoder(Xs.shape[1], MLP_PARAMS["num_shared_feature"], MLP_PARAMS["dropout_ratio"]).to(DEVICE)
    predictor = CATPredictor(MLP_PARAMS["num_shared_feature"], MLP_PARAMS["dropout_ratio"]).to(DEVICE)
    criterion = nn.MSELoss(reduction="none")
    params = list(encoder.parameters()) + list(predictor.parameters())
    optimizer = optim.Adam(params, lr=MLP_PARAMS["learning_rate"], weight_decay=MLP_PARAMS["weight_decay"])
    scheduler = StepLR(optimizer, 50, 0.5)

    for _ in range(MLP_PARAMS["epochs"]):
        encoder.train()
        predictor.train()
        for xb, yb, wb in loader:
            xb, yb, wb = xb.to(DEVICE), yb.to(DEVICE), wb.to(DEVICE)
            optimizer.zero_grad()
            loss = torch.mean(criterion(predictor(encoder(xb)), yb) * wb)
            loss.backward()
            optimizer.step()
        scheduler.step()

    encoder.eval()
    predictor.eval()
    test_loader = DataLoader(TensorDataset(torch.tensor(np.asarray(Xt, dtype=np.float32))), batch_size=4096)
    preds = []
    with torch.no_grad():
        for (xb,) in test_loader:
            preds.append(predictor(encoder(xb.to(DEVICE))).cpu().numpy().reshape(-1))
    return np.concatenate(preds)


def fit_predict_rf(Xs, ys, Xt, weights, seed, min_samples_leaf):
    model = RandomForestRegressor(
        n_estimators=300,
        min_samples_leaf=min_samples_leaf,
        max_features="sqrt",
        n_jobs=-1,
        random_state=seed,
    )
    model.fit(Xs, ys, sample_weight=weights)
    return model.predict(Xt)


def prepare_year(us_df, arg_df, year):
    target = arg_df[arg_df["year"] == year].copy()
    Xs_model = us_df[MODEL_FEATURES].to_numpy(dtype=float)
    Xt_model = target[MODEL_FEATURES].to_numpy(dtype=float)
    Xs_weight = us_df[WEIGHT_FEATURES].to_numpy(dtype=float)
    Xt_weight = target[WEIGHT_FEATURES].to_numpy(dtype=float)
    model_scaler = StandardScaler().fit(np.vstack([Xs_model, Xt_model]))
    weight_scaler = StandardScaler().fit(np.vstack([Xs_weight, Xt_weight]))
    return (
        target,
        model_scaler.transform(Xs_model),
        model_scaler.transform(Xt_model),
        weight_scaler.transform(Xs_weight),
        weight_scaler.transform(Xt_weight),
    )


def metrics_row(model, seed, year, observed, predicted):
    return {
        "model": model,
        "seed": seed,
        "year": year,
        "R2": r2_score(observed, predicted) if len(observed) > 1 else np.nan,
        "RMSE": np.sqrt(mean_squared_error(observed, predicted)),
    }


def run(crop, data_dir, seeds=None):
    cfg = CROPS[crop]
    seeds = cfg["seeds"] if seeds is None else seeds
    us_df, arg_df = load_data(crop, data_dir)
    y_source = us_df["yield_t_ha"].to_numpy(dtype=float)

    frames = []
    for seed in seeds:
        for year in cfg["target_years"]:
            target, Xs_model, Xt_model, Xs_weight, Xt_weight = prepare_year(us_df, arg_df, year)
            if target.empty:
                continue
            run_seed = seed + int(year)
            weights = kmm_weights(
                Xs_weight, Xt_weight, cfg["kmm_max_samples"], cfg["kmm_maxiter"],
                cfg["gamma_max_samples"], run_seed,
            )
            predictions = {
                "KMM_RF": fit_predict_rf(Xs_model, y_source, Xt_model, weights, run_seed, cfg["rf_min_samples_leaf"]),
                "KMM_MLP": fit_predict_mlp(Xs_model, y_source, Xt_model, weights, run_seed),
            }
            for model, pred in predictions.items():
                frames.append(pd.DataFrame({
                    "crop": crop,
                    "model": model,
                    "seed": seed,
                    "year": target["year"].to_numpy(),
                    "province": target["Province"].to_numpy(),
                    "district": target["District"].to_numpy(),
                    "observed": target["yield_t_ha"].to_numpy(dtype=float),
                    "predicted": pred,
                }))

    predictions = pd.concat(frames, ignore_index=True)
    rows = []
    for (model, seed), group in predictions.groupby(["model", "seed"], sort=True):
        for year, sub in group.groupby("year", sort=True):
            rows.append(metrics_row(model, seed, str(year), sub["observed"], sub["predicted"]))
        rows.append(metrics_row(model, seed, "all", group["observed"], group["predicted"]))
    return predictions, pd.DataFrame(rows)
