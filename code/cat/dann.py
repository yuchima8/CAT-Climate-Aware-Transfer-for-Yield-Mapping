import os
import random
from itertools import cycle

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.autograd import Function
from torch.utils.data import DataLoader, TensorDataset

FEATURES = ["GCVI_peak", "GCVI_b30", "GCVI_a30", "GCVI_inter", "GCVI_inter_b30"]
KEYS = ["year", "District", "Province"]

FILES = {
    "soybean": {
        "us": ["US_county_Soy_16_24_5.csv"],
        "argentina": ["ARG_HLS_S2_Harm_3rd_GCVI_2019_2024_0.csv", "Argentina_yield_19_24_1st.csv"],
    },
    "corn": {
        "us": ["US_HLS_S2_Harm_3rd_GCVI_2016_2024_5.csv", "ERA5_US_corn_17_24.csv", "corn_US_yield_1999_2024.csv"],
        "argentina": ["ARG_corn_HLS_Harm_3rd_GCVI_2019_2024_0.csv", "ERA5_ARG_corn_19_24.csv", "ARG_corn_yield_19_24.csv"],
    },
    "winter_wheat": {
        "us": ["US_HLS_S2_Harm_3rd_GCVI_2016_2024_5.csv", "ERA5_US_winter_wheat_16_23_wET.csv",
               "winter_wheat_US_yield_2000_2024.csv"],
        "argentina": ["ARG_HLS_S2_Harm_3rd_GCVI_2018_2023.csv", "ERA5_ARG_winter_wheat_18_23.csv",
                      "ARG_wheat_yield_18_24.csv"],
    },
}


def read(data_dir, region, crop):
    return [pd.read_csv(os.path.join(data_dir, region, crop, name), float_precision="round_trip" if "yield" in name else None)
            for name in FILES[crop][region]]


def argentina_yield(df):
    df = df[["Province", "District", "year", "yield", "harvested_area"]].copy()
    df["yield_t_ha"] = pd.to_numeric(df["yield"], errors="coerce")
    return df.dropna(subset=["yield_t_ha"])


def drop_incomplete(df):
    return df.replace([np.inf, -np.inf], np.nan).dropna(subset=FEATURES + ["yield_t_ha", "year"])


def load_soybean(data_dir):
    (us,) = read(data_dir, "us", "soybean")
    us = us[us["GCVI_peak"] < 20]
    us = us[us["state"] != 5].copy()
    us["yield_t_ha"] = pd.to_numeric(us["yield"], errors="coerce")
    hls, reported = read(data_dir, "argentina", "soybean")
    reported = argentina_yield(reported)
    reported = reported[(reported["yield_t_ha"] > 1) & (reported["harvested_area"] > 1000)]
    arg = reported.merge(hls, on=KEYS, how="inner")
    return drop_incomplete(us), drop_incomplete(arg)


def load_with_climate(data_dir, crop):
    hls, era5, reported = read(data_dir, "us", crop)
    us = hls[hls["GCVI_peak"] < 20].merge(era5, on=["FIPS", "year"]).merge(reported, on=["FIPS", "year"])
    us["yield_t_ha"] = us["yield"]
    us = us.dropna(subset=["yield_t_ha"])
    hls, era5, reported = read(data_dir, "argentina", crop)
    era5 = era5.drop_duplicates(subset=KEYS)
    reported = argentina_yield(reported).drop(columns="yield")
    reported = reported[(reported["yield_t_ha"] > 0) & (reported["yield_t_ha"] < 10)]
    reported = reported[reported["harvested_area"] > 1000].drop_duplicates(subset=KEYS)
    arg = hls[hls["GCVI_peak"] < 10].merge(era5, on=KEYS).merge(reported, on=KEYS).dropna()
    return us, arg


def load_data(crop, data_dir):
    return load_soybean(data_dir) if crop == "soybean" else load_with_climate(data_dir, crop)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class GradientReversal(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None


def encoder(input_dim, dims, dropout):
    layers = []
    for i, (n_in, n_out) in enumerate(zip([input_dim, *dims[:-1]], dims)):
        layers += [nn.Linear(n_in, n_out), nn.ReLU()]
        if i < len(dims) - 1:
            layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


def head(input_dim, dims, output_dim, dropout):
    layers = [nn.Linear(input_dim, dims[0]), nn.ReLU(), nn.Dropout(dropout)]
    for n_in, n_out in zip(dims[:-1], dims[1:]):
        layers += [nn.Linear(n_in, n_out), nn.ReLU()]
    layers.append(nn.Linear(dims[-1], output_dim))
    return nn.Sequential(*layers)


class DANN(nn.Module):
    def __init__(self, input_dim, encoder_dims, head_dims, dropout):
        super().__init__()
        self.encoder = encoder(input_dim, encoder_dims, dropout)
        self.regressor = head(encoder_dims[-1], head_dims, 1, dropout)
        self.domain_classifier = head(encoder_dims[-1], head_dims, 2, dropout)

    def forward(self, x, alpha):
        z = self.encoder(x)
        return self.regressor(z), self.domain_classifier(GradientReversal.apply(z, alpha))

    def predict(self, x):
        return self.regressor(self.encoder(x))

    def domain_logits(self, x, alpha):
        return self.domain_classifier(GradientReversal.apply(self.encoder(x), alpha))


class ReshuffledStream:
    def __init__(self, loader):
        self.loader = loader
        self.iterator = iter(loader)

    def __next__(self):
        try:
            return next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.loader)
            return next(self.iterator)


def epoch_iterators(src_loader, tar_loader, reshuffle_target_each_step):
    if reshuffle_target_each_step:
        src_iter = iter(src_loader)
        return src_iter, ReshuffledStream(tar_loader)
    tar_iter = cycle(tar_loader)
    return iter(src_loader), tar_iter


def make_loader(arrays, batch_size, shuffle, drop_last=False):
    tensors = [torch.tensor(a, dtype=torch.float32) for a in arrays]
    return DataLoader(TensorDataset(*tensors), batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)


def grl_coefficient(progress):
    return float(2.0 / (1.0 + np.exp(-10 * progress)) - 1.0)


def predict(model, loader):
    model.eval()
    with torch.no_grad():
        return np.concatenate([model.predict(x).numpy().reshape(-1) for (x,) in loader])


def train_and_predict(x_src, y_src, x_tar, seed, cfg):
    set_seed(seed)
    batch_size, epochs = cfg["batch_size"], cfg["epochs"]
    validate = cfg["val_fraction"] > 0
    if validate:
        idx_train, idx_val = train_test_split(np.arange(len(x_src)), test_size=cfg["val_fraction"],
                                              random_state=seed, shuffle=True)
        val_loader = make_loader([x_src[idx_val]], batch_size, False)
        y_val = y_src[idx_val].reshape(-1)
    else:
        idx_train = np.arange(len(x_src))
    src_loader = make_loader([x_src[idx_train], y_src[idx_train]], batch_size, True,
                             cfg["drop_last"] and len(idx_train) >= batch_size)
    tar_loader = make_loader([x_tar], batch_size, True, cfg["drop_last"] and len(x_tar) >= batch_size)
    tar_eval_loader = make_loader([x_tar], batch_size, False)

    model = DANN(x_src.shape[1], cfg["encoder_dims"], cfg["head_dims"], cfg["dropout"])
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg["lr_step"], gamma=0.5)
    total_steps = epochs * len(src_loader)
    step = 0
    best_rmse, best_pred = float("inf"), None

    for epoch in range(epochs):
        model.train()
        src_iter, tar_iter = epoch_iterators(src_loader, tar_loader, cfg["reshuffle_target_each_step"])
        for _ in range(len(src_loader)):
            x_s, y_s = next(src_iter)
            (x_t,) = next(tar_iter)
            alpha = grl_coefficient(step / total_steps if cfg["alpha_per_step"] else epoch / max(epochs - 1, 1))
            optimizer.zero_grad()
            if cfg["joint_forward"]:
                y_hat, domain_s = model(x_s, alpha)
                _, domain_t = model(x_t, alpha)
            else:
                y_hat = model.predict(x_s)
                domain_s = model.domain_logits(x_s, alpha)
                domain_t = model.domain_logits(x_t, alpha)
            label_s = torch.zeros(x_s.shape[0], dtype=torch.long)
            label_t = torch.ones(x_t.shape[0], dtype=torch.long)
            domain_loss = 0.5 * (F.cross_entropy(domain_s, label_s) + F.cross_entropy(domain_t, label_t))
            loss = F.mse_loss(y_hat, y_s) + cfg["lambda_domain"] * domain_loss
            loss.backward()
            optimizer.step()
            step += 1
        if validate:
            val_pred = predict(model, val_loader)
            tar_pred = predict(model, tar_eval_loader)
        scheduler.step()
        if validate:
            rmse = float(np.sqrt(mean_squared_error(y_val, val_pred)))
            if rmse < best_rmse:
                best_rmse, best_pred = rmse, tar_pred

    return best_pred if validate else predict(model, tar_eval_loader)


def run(crop, data_dir, seeds, cfg):
    us, arg = load_data(crop, data_dir)
    dtype = cfg["scaling_dtype"]
    x_us = us[FEATURES].to_numpy(dtype=dtype)
    y_us = us["yield_t_ha"].to_numpy(dtype=np.float32).reshape(-1, 1)
    rows = []
    for seed in seeds:
        for year in sorted(arg["year"].unique()):
            tar = arg[arg["year"] == year]
            x_tar = tar[FEATURES].to_numpy(dtype=dtype)
            scaler = StandardScaler().fit(np.vstack([x_us, x_tar]))
            pred = train_and_predict(scaler.transform(x_us), y_us, scaler.transform(x_tar), seed, cfg)
            rows.append(pd.DataFrame({
                "crop": crop, "model": "DANN", "seed": seed, "year": int(year),
                "province": tar["Province"].to_numpy(), "district": tar["District"].to_numpy(),
                "observed": tar["yield_t_ha"].to_numpy(), "predicted": pred.astype(float),
            }))
            print(crop, seed, year, flush=True)
    predictions = pd.concat(rows, ignore_index=True)
    return predictions, metrics(predictions)


def metrics(predictions):
    rows = []
    for (model, seed), group in predictions.groupby(["model", "seed"]):
        for year, part in list(group.groupby("year")) + [("all", group)]:
            rows.append({
                "model": model, "seed": seed, "year": year,
                "R2": r2_score(part["observed"], part["predicted"]),
                "RMSE": float(np.sqrt(mean_squared_error(part["observed"], part["predicted"]))),
            })
    return pd.DataFrame(rows)
