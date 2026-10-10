# CAT: Climate-Aware Transfer for Inter-Continental Yield Mapping

This repository contains the code for the Argentina experiments in:

> Ma, Y., Tong, X.-Y., Sadeh, Y., Baber, S., Oliinyk, O., Sawahn, R., Wang, S., Becker-Reshef, I., Lobell, D.B. CAT: Climate-Aware Transfer for Inter-Continental Yield Mapping. *ISPRS Journal of Photogrammetry and Remote Sensing* (under review).

The experiments transfer county-level yield models from the United States (source domain) to Argentina (target domain) for soybean, corn and winter wheat, without using any Argentine yield data during training. CAT weights each U.S. county-year by the similarity of its growing-season climate to the target region and year, using growing degree days (GDD) and precipitation (PPT).

## Contents

| Path | Description |
|---|---|
| `run_mlp.py` | Locally trained MLP (leave-one-year-out), direct-transfer MLP and CAT-based MLP |
| `run_rf.py` | Direct-transfer and CAT-weighted random forests, and locally trained random forests for corn and winter wheat |
| `run_dann.py` | Domain-adversarial neural network (DANN) baseline |
| `run_kmm.py` | Kernel mean matching (KMM) source-instance weighting baselines (KMM-MLP, KMM-RF) |
| `run_sequence.py` | Locally trained, direct-transfer and CAT-based LSTM and Transformer models (soybean) |
| `cat/` | Shared modules: data loading, climate weights, network and training routines (`cat/sequence.py` for the LSTM and Transformer) |
| `analysis/summarize.py` | Per-year and pooled R² and RMSE (mean and SD over runs) |
| `analysis/bootstrap.py` | Paired bootstrap over spatial blocks of counties and test years |
| `analysis/robustness.py` | Year-level robustness metrics (worst-year accuracy, regret, spread) |
| `environment.yml` | Software environment |

The CropNet model is available at https://github.com/x-ytong/CropNet

## Installation

```bash
conda env create -f environment.yml
conda activate cat
```

The experiments run on a CPU.

## Data

The scripts expect the data folder next to this repository (`../data`). Pass `--data` (and `--predictions` for the analysis scripts) to use another location.

```
data/
  us/{soybean,corn,winter_wheat}/          U.S. county features, climate and yields (source domain)
  argentina/{soybean,corn,winter_wheat}/   Argentine department features, climate and yields (target domain)
  {us,argentina}/soybean/hls_timeseries/   HLS surface reflectance time series used by the LSTM and Transformer
  argentina/county_blocks.csv              Spatial blocks used in the bootstrap
  predictions/                             County-level predictions of all models
  gee/HLS_GCVI_export.js                   Google Earth Engine script: HLS (Landsat 8/9 and Sentinel-2) reflectance and GCVI time series
  gee/ERA5_GDD_PPT_export.js               Google Earth Engine script: ERA5-Land monthly GDD and precipitation
```

The two Google Earth Engine scripts download the raw inputs for the United States and Argentina. Run them in the Earth Engine Code Editor (https://code.earthengine.google.com/). Set `REGION` (`US` or `Argentina`), `CROP` (`soybean`, `corn` or `winter_wheat`) and `YEARS` at the top of each script. One CSV per year is exported to Google Drive.

- `HLS_GCVI_export.js` uses the HLS v2.0 L30 (Landsat 8/9) and S30 (Sentinel-2) surface reflectance collections. It masks clouds, cloud shadows, snow and water with the Fmask band and keeps only cropland of the selected crop. Each export holds county (department) means of the six surface reflectance bands (blue, green, red, NIR, SWIR1, SWIR2), GCVI and NDVI for every image date in the growing season. Crop masks come from the USDA Cropland Data Layer for the United States. For Argentina, upload the crop type maps (de Abelleyra et al., 2020, 2024) as Earth Engine assets and set `ARG_CROP_MAP` and `ARG_CROP_CLASS`.
- `ERA5_GDD_PPT_export.js` uses the ERA5-Land daily aggregates. It computes monthly growing degree days (columns `GDD_<month>`) and total precipitation in metres (columns `ppt_<month>`) for each county (department). Daily GDD is computed from hourly temperatures interpolated between the daily minimum and maximum. The base and upper temperatures are 8 and 30 °C for soybean and corn, and 0 and 26 °C for winter wheat.
- U.S. counties come from the TIGER/2018 county boundaries in Earth Engine. For Argentina, upload the department boundaries with the attributes `name_1` (province) and `name_2` (department), and set `ARG_BOUNDARIES`.

All yields in the data folder are in t/ha. U.S. yields from USDA NASS (bu/acre) were converted with a factor of 0.0673, and Argentine yields (kg/ha) were divided by 1000. The harmonic GCVI features in `us/` and `argentina/` are fitted to these GCVI time series. Yields are not downloaded by the scripts; see Data availability for their sources.

## Usage

Train the models and write county-level predictions and metrics to `../outputs`:

```bash
python run_mlp.py  --crop soybean        # also: corn, winter_wheat
python run_rf.py   --crop soybean
python run_dann.py --crop soybean
python run_kmm.py  --crop soybean
python run_sequence.py --arch LSTM         # also: Transformer
```

Summarize accuracy, run the bootstrap and compute the robustness metrics from the predictions in `../data/predictions`:

```bash
python analysis/summarize.py  --crop soybean
python analysis/bootstrap.py  --crop soybean --schemes block:block_kmeans,year_exact,loyo --B 2000
python analysis/bootstrap.py  --crop soybean --schemes hier:block_kmeans --B 1000
python analysis/robustness.py --crop soybean
```

Each experiment uses five random seeds; the default seeds are those used in the paper.

## Data availability

The US and Argentina data used in this study are publicly available. Please download the data and put them in an accompany data folder. 

**Input data.** The accompanying data folder provides the county-level (Argentine department-level) inputs used in the experiments:

- **Argentina:** harmonic GCVI phenology features derived from Harmonized Landsat Sentinel-2 (HLS v2.0) surface reflectance, and growing-season growing degree days and precipitation derived from ERA5-Land. Department-level yields of soybean, corn and wheat are from the Agricultural Estimates of the Ministry of Agriculture, Livestock and Fisheries of Argentina (https://datosestimaciones.magyp.gob.ar/).
- **United States (source domain):** county-level features derived from the same satellite and climate products. County-level yields are from USDA NASS Quick Stats (https://quickstats.nass.usda.gov/).
- **Predictions:** county-level predictions of all models reported in the paper for the Argentina experiments.

**Source data.** The HLS and ERA5-Land source data are publicly available, for example through Google Earth Engine (https://earthengine.google.com/). Crop masks were derived from the USDA Cropland Data Layer and published crop type maps of Argentina (de Abelleyra et al., 2020, 2024).

**Ukraine.** The Ukraine experiments are not included. The county-level yield data for Ukraine were obtained from the State Statistics Service of Ukraine under an agreement that does not permit redistribution. Interested users may request these data from the State Statistics Service of Ukraine.

## Citation

If you use this code, please cite the paper above.

## Contact

Yuchi Ma, Stanford University (yuchima@stanford.edu)
