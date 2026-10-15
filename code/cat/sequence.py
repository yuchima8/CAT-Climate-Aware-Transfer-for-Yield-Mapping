import copy
import glob
import math
import os
import random
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_squared_error, r2_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split

BANDS = ["BLUE", "GREEN", "RED", "NIR", "SWIR1", "SWIR2"]
DOY_RANGE = (91, 273)
DAY_GAP = 10
US_YEARS = ["2017", "2018", "2019", "2020", "2021", "2022", "2023", "2024"]
ARG_YEARS = ["2019", "2020", "2021", "2022", "2023", "2024"]
DROPOUT = 0.2

CONFIGS = {
    "LSTM": {
        "local": dict(lr=1e-3, val_ratio=0.2, batch_size=128, epochs=200, seed=42),
        "direct": dict(lr=1e-3, val_ratio=0.2, batch_size=128, epochs=200, seed=41),
        "CAT": dict(lr=1e-3, val_ratio=0.1, batch_size=256, epochs=200, seed=41),
    },
    "Transformer": {
        "local": dict(lr=1e-3, val_ratio=0.2, batch_size=128, epochs=200, seed=42),
        "direct": dict(lr=1e-3, val_ratio=0.2, batch_size=128, epochs=200, seed=43),
        "CAT": dict(lr=1e-4, val_ratio=0.2, batch_size=128, epochs=500, seed=42),
    },
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def generator(seed):
    return torch.Generator().manual_seed(seed)


def read_timeseries(folder):
    frames = [pd.read_csv(f) for f in sorted(glob.glob(os.path.join(folder, "*.csv")))]
    ts = pd.concat([f for f in frames if not f.empty], ignore_index=True)
    ts["date"] = pd.to_datetime(ts["date"], format="mixed")
    return ts


def load_us(data):
    d = os.path.join(data, "us", "soybean")
    fea = read_timeseries(os.path.join(d, "hls_timeseries"))
    year = fea["date"].dt.year
    fea["year"] = year
    fea["DOY"] = (fea["date"] - pd.to_datetime({"year": year, "month": 1, "day": 1})).dt.days + 1
    yld = pd.read_csv(os.path.join(d, "soy_US_yield_1999_2024.csv"), low_memory=False, float_precision="round_trip")
    clm = pd.read_csv(os.path.join(d, "ERA5_US_soybean_16_24.csv"))
    return dict(fea=fea, yld=yld, clm=clm, fea_key=["year", "GEOID"], yld_key=["year", "FIPS"], clm_key=["year", "GEOID"])


def load_argentina(data):
    d = os.path.join(data, "argentina", "soybean")
    fea = read_timeseries(os.path.join(d, "hls_timeseries"))
    year = fea["date"].dt.year
    start_this = pd.to_datetime({"year": year, "month": 7, "day": 1})
    start_prev = pd.to_datetime({"year": year - 1, "month": 7, "day": 1})
    after = fea["date"] >= start_this
    fea["year"] = np.where(after, year + 1, year)
    fea["DOY"] = (fea["date"] - start_this.where(after, start_prev)).dt.days + 1
    yld = pd.read_csv(os.path.join(d, "Argentina_yield_19_24_1st.csv"), float_precision="round_trip")
    yld = yld[(yld["yield"] > 0) & (yld["harvested_area"] > 1000)].copy()
    clm = pd.read_csv(os.path.join(d, "ERA5_ARG_soybean_19_24.csv"))
    fea["id"] = fea["name_1"].astype(str) + fea["name_2"].astype(str)
    yld["id"] = yld["Province"].astype(str) + yld["District"].astype(str)
    clm["id"] = clm["Province"].astype(str) + clm["District"].astype(str)
    return dict(fea=fea, yld=yld, clm=clm, fea_key=["year", "id"], yld_key=["year", "id"], clm_key=["year", "id"])


def composite(raw):
    raw = raw[(raw[:, 0] >= DOY_RANGE[0]) & (raw[:, 0] <= DOY_RANGE[1])]
    if raw.size == 0:
        raise ValueError("no observations")
    doy, feat = raw[:, 0], raw[:, 1:]
    bins = np.arange(DOY_RANGE[0], DOY_RANGE[1] + DAY_GAP, DAY_GAP)
    groups = np.digitize(doy, bins)
    medians = [np.median(feat[groups == g], axis=0) if np.any(groups == g) else np.nan for g in range(1, len(bins))]
    filled = []
    for band in range(feat.shape[1]):
        values = [m[band] if isinstance(m, np.ndarray) else np.nan for m in medians]
        valid = np.where(~np.isnan(values))[0]
        if len(valid) == 0:
            raise ValueError("no valid observations")
        filled.append(np.interp(np.arange(len(values)), valid, [values[i] for i in valid]))
    return filled


def make_dataset(domain, years, with_climate):
    years = {str(y) for y in years}
    features = defaultdict(dict)
    for (year, unit), group in domain["fea"].groupby(domain["fea_key"], dropna=False, sort=True):
        year, unit = str(year), str(unit)
        if year not in years:
            continue
        try:
            features[year][unit] = composite(group[["DOY"] + BANDS].to_numpy(copy=True))
        except Exception:
            continue
    yields = defaultdict(dict)
    for (year, unit), group in domain["yld"].groupby(domain["yld_key"], dropna=False, sort=True):
        year, unit = str(year), str(unit)
        if year in years:
            yields[year][unit] = group[["yield"]].to_numpy(copy=True)
    climate = defaultdict(dict)
    if with_climate:
        clm = domain["clm"]
        gdd = [c for c in clm.columns if c.startswith("GDD")]
        ppt = [c for c in clm.columns if c.startswith("ppt")]
        for (year, unit), group in clm.groupby(domain["clm_key"], dropna=False, sort=True):
            climate[str(year)][str(unit)] = (float(group[gdd].sum(axis=1).sum()), float(group[ppt].sum(axis=1).sum()))
    x, y, c, keys = [], [], [], []
    for year in sorted(years):
        for unit in features[year].keys():
            try:
                fea = np.array(features[year][unit])
                yld = next(iter(yields[year][unit]))
                clm = np.array(climate[year][unit]) if with_climate else None
            except Exception:
                continue
            x.append(fea)
            y.extend(yld)
            if with_climate:
                c.append(clm)
            keys.append((int(year), unit))
    tensors = [torch.tensor(np.stack(x, axis=0), dtype=torch.float32).unsqueeze(1),
               torch.tensor(np.array(y), dtype=torch.float32)]
    c = np.stack(c, axis=0) if with_climate else None
    if with_climate:
        tensors.append(torch.tensor(c, dtype=torch.float32))
    return TensorDataset(*tensors), keys, c


class LSTMRegressor(nn.Module):
    def __init__(self, n_bands=6, hidden=128, layers=2, dropout=DROPOUT):
        super().__init__()
        self.lstm = nn.LSTM(n_bands, hidden, num_layers=layers, batch_first=True, dropout=dropout)
        self.head = nn.Sequential(nn.LayerNorm(hidden), nn.Dropout(dropout), nn.Linear(hidden, 1))

    def forward(self, x):
        _, (h, _) = self.lstm(x.squeeze(1).permute(0, 2, 1))
        return self.head(h[-1]).squeeze(1)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model

    def forward(self, x):
        position = torch.arange(x.size(1), device=x.device, dtype=x.dtype).unsqueeze(1)
        div = torch.exp(torch.arange(0, self.d_model, 2, device=x.device, dtype=x.dtype) * (-math.log(10000.0) / self.d_model))
        pe = torch.zeros(1, x.size(1), self.d_model, device=x.device, dtype=x.dtype)
        pe[0, :, 0::2] = torch.sin(position * div)
        pe[0, :, 1::2] = torch.cos(position * div[: pe[0, :, 1::2].shape[-1]])
        return x + pe


class TransformerRegressor(nn.Module):
    def __init__(self, n_bands=6, d_model=128, heads=8, layers=3, feedforward=256, dropout=DROPOUT):
        super().__init__()
        self.input_proj = nn.Linear(n_bands, d_model)
        self.positional_encoding = PositionalEncoding(d_model)
        layer = nn.TransformerEncoderLayer(d_model, heads, feedforward, dropout, batch_first=True, norm_first=False)
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Dropout(dropout), nn.Linear(d_model, 1))

    def forward(self, x):
        z = self.encoder(self.positional_encoding(self.input_proj(x.squeeze(1).permute(0, 2, 1))))
        return self.head(z.mean(dim=1)).squeeze(1)


ARCHITECTURES = {"LSTM": LSTMRegressor, "Transformer": TransformerRegressor}


def climate_weighted_loss(out, true, clm, clm_mean, clm_std):
    d = (clm - clm_mean) / (clm_std + 1e-8)
    w = 1.0 / torch.sqrt(torch.exp(d[:, 0]) ** 2 + torch.exp(d[:, 1]) ** 2).clamp(1e-5, 1e5)
    return (w * nn.functional.smooth_l1_loss(out.view(-1), true.view(-1), beta=1.0, reduction="none")).mean()


@torch.inference_mode()
def evaluate(model, loader):
    model.eval()
    true, pred = [], []
    for batch in loader:
        pred.extend(model(batch[0].to(DEVICE)).cpu().numpy())
        true.extend(batch[1].numpy())
    return np.array(true), np.array(pred)


def train_best(arch, dataset, cfg, seed, offset, clm_stats=None):
    val_size = int(cfg["val_ratio"] * len(dataset))
    train, val = random_split(dataset, [len(dataset) - val_size, val_size], generator=generator(seed + offset))
    train_loader = DataLoader(train, batch_size=cfg["batch_size"], shuffle=True, generator=generator(seed + offset + 1000))
    val_loader = DataLoader(val, batch_size=cfg["batch_size"], shuffle=False)
    seed_everything(seed + offset)
    model = ARCHITECTURES[arch]().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=0)
    criterion = nn.SmoothL1Loss(beta=1.0)
    best_r2, best_state = -np.inf, None
    for _ in range(cfg["epochs"]):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            batch = [t.to(DEVICE) for t in batch]
            out = model(batch[0])
            loss = criterion(out, batch[1]) if clm_stats is None else climate_weighted_loss(out, batch[1], batch[2], *clm_stats)
            loss.backward()
            optimizer.step()
        r2 = r2_score(*evaluate(model, val_loader))
        if r2 > best_r2:
            best_r2, best_state = r2, copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    return model


def predict_years(model, target, years, cfg, with_climate):
    rows = []
    for year in years:
        test, keys, _ = make_dataset(target, [year], with_climate)
        true, pred = evaluate(model, DataLoader(test, batch_size=cfg["batch_size"], shuffle=False))
        rows.append(pd.DataFrame({"year": [k[0] for k in keys], "unit": [k[1] for k in keys],
                                  "observed": true, "predicted": pred}))
    return pd.concat(rows, ignore_index=True)


def run_local(arch, us, arg, seed, cfg):
    rows = []
    offset = DOY_RANGE[0] + DOY_RANGE[1]
    for year in ARG_YEARS:
        train, _, _ = make_dataset(arg, [y for y in ARG_YEARS if y != year], False)
        model = train_best(arch, train, cfg, seed, offset + int(year))
        rows.append(predict_years(model, arg, [year], cfg, False))
    return pd.concat(rows, ignore_index=True)


def run_direct(arch, us, arg, seed, cfg):
    train, _, _ = make_dataset(us, US_YEARS, False)
    model = train_best(arch, train, cfg, seed, DOY_RANGE[0] + DOY_RANGE[1])
    return predict_years(model, arg, ARG_YEARS, cfg, False)


def run_cat(arch, us, arg, seed, cfg):
    train, _, _ = make_dataset(us, US_YEARS, True)
    _, _, clm = make_dataset(arg, ARG_YEARS, True)
    stats = (torch.tensor(np.nanmean(clm, axis=0), dtype=torch.float32).to(DEVICE),
             torch.tensor(np.nanstd(clm, axis=0, ddof=0), dtype=torch.float32).to(DEVICE))
    model = train_best(arch, train, cfg, seed, DOY_RANGE[0] + DOY_RANGE[1], stats)
    return predict_years(model, arg, ARG_YEARS, cfg, True)


RUNNERS = {"local": run_local, "direct": run_direct, "CAT": run_cat}


def metrics(pred):
    rows = []
    for (model, seed), g in pred.groupby(["model", "seed"]):
        for year, h in list(g.groupby("year")) + [("all", g)]:
            rows.append(dict(model=model, seed=seed, year=year, R2=r2_score(h["observed"], h["predicted"]),
                             RMSE=float(np.sqrt(mean_squared_error(h["observed"], h["predicted"])))))
    return pd.DataFrame(rows)
