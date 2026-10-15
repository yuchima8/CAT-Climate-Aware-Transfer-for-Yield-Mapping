import os

import pandas as pd

FEATURES = ["GCVI_peak", "GCVI_b30", "GCVI_a30", "GCVI_inter", "GCVI_inter_b30"]
KEYS = ["year", "District", "Province"]
YEARS = {"soybean": [2019, 2020, 2021, 2022, 2023, 2024], "corn": [2019, 2020, 2021, 2022, 2023, 2024],
         "winter_wheat": [2018, 2019, 2020, 2021, 2022, 2023]}


def _argentina_yield(path):
    y = pd.read_csv(path, float_precision="round_trip")[["Province", "District", "year", "yield", "harvested_area"]].copy()
    y["yield"] = pd.to_numeric(y["yield"], errors="coerce")
    y = y.dropna(subset=["yield"])
    y = y[(y["yield"] > 0) & (y["yield"] < 10) & (y["harvested_area"] > 1000)]
    return y.drop_duplicates(subset=KEYS)


def _sum(df, cols):
    out = df[cols[0]].copy()
    for c in cols[1:]:
        out = out + df[c]
    return out


def load_soybean(data):
    us = pd.read_csv(os.path.join(data, "us", "soybean", "US_county_Soy_16_24_5.csv"))
    us = us[(us["GCVI_peak"] < 20) & (us["state"] != 5)].copy()
    us["GDD"] = _sum(us, ["GDD_May", "GDD_Jun", "GDD_Jul", "GDD_Aug", "GDD_Sep"])
    us["PPT"] = _sum(us, ["PPT_May", "PPT_Jun", "PPT_Jul", "PPT_Aug", "PPT_Sep"])
    arg = pd.read_csv(os.path.join(data, "argentina", "soybean", "ARG_soybean_inputs_MLP.csv"))
    return us, arg


def load_corn(data):
    d = os.path.join(data, "us", "corn")
    us = pd.read_csv(os.path.join(d, "US_HLS_S2_Harm_3rd_GCVI_2016_2024_5.csv"))
    us = us[us["GCVI_peak"] < 20]
    us = us.merge(pd.read_csv(os.path.join(d, "ERA5_US_corn_17_24.csv")), on=["FIPS", "year"])
    us = us.merge(pd.read_csv(os.path.join(d, "corn_US_yield_1999_2024.csv"), float_precision="round_trip"), on=["FIPS", "year"])
    us["GDD"] = us[["GDD_5", "GDD_6", "GDD_7", "GDD_8", "GDD_9"]].sum(axis=1)
    us["PPT"] = us[["ppt_5", "ppt_6", "ppt_7", "ppt_8", "ppt_9"]].sum(axis=1)
    d = os.path.join(data, "argentina", "corn")
    hls = pd.read_csv(os.path.join(d, "ARG_corn_HLS_Harm_3rd_GCVI_2019_2024_0.csv"))
    hls = hls[hls["GCVI_peak"] < 10]
    era = pd.read_csv(os.path.join(d, "ERA5_ARG_corn_19_24.csv")).drop_duplicates(subset=KEYS)
    arg = hls.merge(era, on=KEYS).merge(_argentina_yield(os.path.join(d, "ARG_corn_yield_19_24.csv")), on=KEYS).dropna()
    arg["GDD"] = arg[["GDD_12", "GDD_1", "GDD_2", "GDD_3", "GDD_4"]].sum(axis=1)
    arg["PPT"] = arg[["ppt_12", "ppt_1", "ppt_2", "ppt_3", "ppt_4"]].sum(axis=1)
    return us, arg


def load_winter_wheat(data):
    d = os.path.join(data, "us", "winter_wheat")
    us = pd.read_csv(os.path.join(d, "US_HLS_S2_Harm_3rd_GCVI_2016_2024_5.csv"))
    us = us[us["GCVI_peak"] < 20]
    us = us.merge(pd.read_csv(os.path.join(d, "ERA5_US_winter_wheat_16_23_wET.csv")), on=["FIPS", "year"])
    us = us.merge(pd.read_csv(os.path.join(d, "winter_wheat_US_yield_2000_2024.csv"), float_precision="round_trip"), on=["FIPS", "year"])
    us["GDD"] = _sum(us, ["GDD_3", "GDD_4", "GDD_5", "GDD_6"])
    us["PPT"] = _sum(us, ["PPT_3", "PPT_4", "PPT_5", "PPT_6"])
    d = os.path.join(data, "argentina", "winter_wheat")
    hls = pd.read_csv(os.path.join(d, "ARG_HLS_S2_Harm_3rd_GCVI_2018_2023.csv"))
    hls = hls[hls["GCVI_peak"] < 10]
    era = pd.read_csv(os.path.join(d, "ERA5_ARG_winter_wheat_18_23.csv")).drop_duplicates(subset=KEYS)
    arg = hls.merge(era, on=KEYS).merge(_argentina_yield(os.path.join(d, "ARG_wheat_yield_18_24.csv")), on=KEYS).dropna()
    arg["GDD"] = _sum(arg, ["GDD_9", "GDD_10", "GDD_11", "GDD_12"])
    arg["PPT"] = _sum(arg, ["ppt_9", "ppt_10", "ppt_11", "ppt_12"])
    return us, arg


LOADERS = {"soybean": load_soybean, "corn": load_corn, "winter_wheat": load_winter_wheat}
