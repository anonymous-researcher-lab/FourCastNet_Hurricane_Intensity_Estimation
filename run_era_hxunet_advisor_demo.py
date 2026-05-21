#!/usr/bin/env python3
"""
run_era_hxunet_advisor_demo.py
ERA-only advisor demo: HURDAT vs raw FCN-ERA vs HxCNN-ERA (CNN bias-correction checkpoint).

Outputs -> outputs_advisor_demo/
  hie_era_hxunet_predictions.csv
  hie_era_hxunet_metrics.csv
  hie_era_hxunet_wind_mslp_timeseries.png
  hie_era_hxunet_delta24_ri.png
  README_results_summary.md
"""

import sys
import os
import glob

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(REPO, 'example'))

from src import (
    BiasCorrectionModel,
    get_means_stds_from_hurdat_tracks,
    run_inference,
    calculate_mean_differences,
)

OUTPUT_DIR = os.path.join(REPO, 'outputs_advisor_demo')
os.makedirs(OUTPUT_DIR, exist_ok=True)

HURRICANE_PREFIX = '2024C5Beryl_2024063006'
ERA_SCALING = os.path.join(REPO, 'ERA_scaling')
CKPT_ERA    = os.path.join(REPO, 'bias_correction_checkpoints_ERA')
FILES_DIR   = os.path.join(REPO, 'files')

# ── File collections ─────────────────────────────────────────────────────────
era_nc_files = sorted(glob.glob(os.path.join(FILES_DIR, 'ERA*.nc')))
hurdat_csvs  = sorted(glob.glob(os.path.join(FILES_DIR, '*HURDAT.csv')))
era_csvs     = sorted(glob.glob(os.path.join(FILES_DIR, '*ERA.csv')))

# raw FCN-ERA predictions are stored in the ERA CSV
files_era = {
    'cropped_data_files': era_nc_files,
    'hurdat_track_files': hurdat_csvs,
    'merra_track_files':  era_csvs,
}

means_track_files = get_means_stds_from_hurdat_tracks(track_data_files=hurdat_csvs)

# ── Load ERA CNN checkpoint ───────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

model = BiasCorrectionModel(in_channels=20).to(device)
ckpt_path = os.path.join(CKPT_ERA, 'CNN_checkpoints.pth')

print(f"Running ERA HxCNN inference  [{ckpt_path}] ...")
result = run_inference(
    model=model,
    model_weights=ckpt_path,
    files=files_era,
    hurricane_prefix=HURRICANE_PREFIX,
    means_track_files=means_track_files,
    cropped_data_meanstd_folder=ERA_SCALING,
)

hurdat  = result['hurdat']   # HURDAT best-track
fcn_era = result['merra']    # raw FCN-ERA (mslp already /100 inside run_inference)
hxcnn   = result['model']    # HxCNN bias-corrected

n_samples = len(hurdat)
fhours    = np.arange(n_samples) * 6
diffs     = calculate_mean_differences(result)

print(f"Timesteps loaded: {n_samples}  ({fhours[0]}–{fhours[-1]} h)")

# ── 1. Predictions CSV ────────────────────────────────────────────────────────
hurdat_times = pd.read_csv(hurdat_csvs[0])['time'].values

df_pred = pd.DataFrame({
    'forecast_hour':      fhours,
    'time_utc':           hurdat_times,
    'hurdat_vmax_ms':     hurdat['vmax'].values,
    'hurdat_mslp_hPa':    hurdat['mslp'].values,
    'fcn_era_ws_ms':      fcn_era['ws'].values,
    'fcn_era_mslp_hPa':   fcn_era['mslp'].values,
    'hxcnn_era_ws_ms':    hxcnn['ws'].values,
    'hxcnn_era_mslp_hPa': hxcnn['mslp'].values,
})
pred_csv = os.path.join(OUTPUT_DIR, 'hie_era_hxunet_predictions.csv')
df_pred.to_csv(pred_csv, index=False)
print(f"Saved {pred_csv}")

# ── 2. Wind speed & MSLP time series ─────────────────────────────────────────
fig, axs = plt.subplots(1, 2, figsize=(14, 5))

axs[0].plot(fhours, hurdat['vmax'].values,  'k-o', lw=2,  label='HURDAT')
axs[0].plot(fhours, fcn_era['ws'].values,   'b-o',        label=f'Raw FCN-ERA  bias {diffs["merra_ws"]:+.1f} m/s')
axs[0].plot(fhours, hxcnn['ws'].values,     'r-o',        label=f'HxCNN-ERA   bias {diffs["model_ws"]:+.1f} m/s')
axs[0].set_ylabel('Wind Speed (m/s)', fontsize=12)
axs[0].set_xlabel('Forecast Hour', fontsize=12)
axs[0].set_title('(a) Wind Speed: HURDAT vs FCN vs HxCNN', fontsize=12)
axs[0].legend(fontsize=10, frameon=False)
axs[0].set_xticks(np.arange(0, fhours[-1] + 1, 12))

axs[1].plot(fhours, hurdat['mslp'].values,  'k-o', lw=2,  label='HURDAT')
axs[1].plot(fhours, fcn_era['mslp'].values, 'b-o',        label=f'Raw FCN-ERA  bias {diffs["merra_mslp"]:+.1f} hPa')
axs[1].plot(fhours, hxcnn['mslp'].values,   'r-o',        label=f'HxCNN-ERA   bias {diffs["model_mslp"]:+.1f} hPa')
axs[1].set_ylabel('MSLP (hPa)', fontsize=12)
axs[1].set_xlabel('Forecast Hour', fontsize=12)
axs[1].set_title('(b) MSLP: HURDAT vs FCN vs HxCNN', fontsize=12)
axs[1].legend(fontsize=10, frameon=False)
axs[1].set_xticks(np.arange(0, fhours[-1] + 1, 12))

plt.suptitle(
    f'Hurricane Beryl (C5)  |  ERA HxCNN Bias Correction  |  IC: {HURRICANE_PREFIX}',
    fontweight='bold',
)
plt.tight_layout()
ts_png = os.path.join(OUTPUT_DIR, 'hie_era_hxunet_wind_mslp_timeseries.png')
plt.savefig(ts_png, dpi=150)
plt.close()
print(f"Saved {ts_png}")

# ── 3. 24-hour delta wind (RI evaluation) ────────────────────────────────────
def delta24(arr):
    """Forward 24-h change (4 × 6-h steps)."""
    a = np.asarray(arr, dtype=float)
    d = np.full_like(a, np.nan)
    d[:-4] = a[4:] - a[:-4]
    return d

d_hurdat = delta24(hurdat['vmax'].values)
d_fcn    = delta24(fcn_era['ws'].values)
d_hxcnn  = delta24(hxcnn['ws'].values)
valid    = ~np.isnan(d_hurdat)

RI_THRESH = 15.4  # ~30 kt / 24 h (NHC RI criterion)

fig, ax = plt.subplots(figsize=(11, 5))
ax.plot(fhours[valid], d_hurdat[valid], 'k-o', lw=2, label='HURDAT Δ24h wind')
ax.plot(fhours[valid], d_fcn[valid],    'b-o',       label='Raw FCN-ERA Δ24h wind')
ax.plot(fhours[valid], d_hxcnn[valid],  'r-o',       label='HxCNN-ERA Δ24h wind')
ax.axhline(0,         color='gray',   ls='--', lw=0.8)
ax.axhline(RI_THRESH, color='purple', ls=':',  lw=1.5,
           label=f'RI threshold ({RI_THRESH} m/s / 24h ≈ 30 kt)')
ax.set_xlabel('Forecast Hour', fontsize=12)
ax.set_ylabel('Δ Wind Speed (m/s / 24h)', fontsize=12)
ax.set_title('24-Hour Wind Change — RI Evaluation  |  Hurricane Beryl ERA HxCNN', fontsize=12)
ax.legend(fontsize=10, frameon=False)
ax.set_xticks(np.arange(0, fhours[-1] + 1, 12))
plt.tight_layout()
ri_png = os.path.join(OUTPUT_DIR, 'hie_era_hxunet_delta24_ri.png')
plt.savefig(ri_png, dpi=150)
plt.close()
print(f"Saved {ri_png}")

# ── 4. Metrics ────────────────────────────────────────────────────────────────
def metrics(pred, obs, name):
    mae  = mean_absolute_error(obs, pred)
    rmse = np.sqrt(mean_squared_error(obs, pred))
    bias = float(np.mean(pred - obs))
    corr, _ = pearsonr(pred, obs)
    return {'model': name, 'mae': mae, 'rmse': rmse, 'bias': bias, 'corr': corr}

hw = hurdat['vmax'].values.astype(float)
hm = hurdat['mslp'].values.astype(float)

metric_rows = [
    metrics(fcn_era['ws'].values.astype(float),   hw, 'FCN_ERA_WS'),
    metrics(hxcnn['ws'].values.astype(float),      hw, 'HxCNN_ERA_WS'),
    metrics(fcn_era['mslp'].values.astype(float),  hm, 'FCN_ERA_MSLP'),
    metrics(hxcnn['mslp'].values.astype(float),    hm, 'HxCNN_ERA_MSLP'),
]
metrics_df = pd.DataFrame(metric_rows)
metrics_csv = os.path.join(OUTPUT_DIR, 'hie_era_hxunet_metrics.csv')
metrics_df.to_csv(metrics_csv, index=False)
print(f"Saved {metrics_csv}")

# ── 5. RI window analysis ─────────────────────────────────────────────────────
ri_idx = np.where(d_hurdat[valid] >= RI_THRESH)[0]
d_fcn_v   = d_fcn[valid]
d_hxcnn_v = d_hxcnn[valid]
d_hurdat_v = d_hurdat[valid]

if len(ri_idx) > 0:
    fcn_ri_mae   = float(np.mean(np.abs(d_fcn_v[ri_idx]   - d_hurdat_v[ri_idx])))
    hxcnn_ri_mae = float(np.mean(np.abs(d_hxcnn_v[ri_idx] - d_hurdat_v[ri_idx])))
    hxcnn_improves_ri = hxcnn_ri_mae < fcn_ri_mae
else:
    fcn_ri_mae = hxcnn_ri_mae = None
    hxcnn_improves_ri = None

# ── 6. README summary ─────────────────────────────────────────────────────────
def _mr(name):
    return next(r for r in metric_rows if r['model'] == name)

fcn_ws   = _mr('FCN_ERA_WS')
hxcnn_ws = _mr('HxCNN_ERA_WS')
fcn_mslp   = _mr('FCN_ERA_MSLP')
hxcnn_mslp = _mr('HxCNN_ERA_MSLP')

ws_improv_pct   = 100 * (fcn_ws['rmse']   - hxcnn_ws['rmse'])   / fcn_ws['rmse']
mslp_improv_pct = 100 * (fcn_mslp['rmse'] - hxcnn_mslp['rmse']) / fcn_mslp['rmse']

ri_lines = ""
if hxcnn_improves_ri is not None:
    ri_lines = f"""
### RI-like window (Δ24h wind ≥ {RI_THRESH} m/s)
| Model | MAE (m/s) |
|---|---|
| Raw FCN-ERA | {fcn_ri_mae:.2f} |
| HxCNN-ERA | {hxcnn_ri_mae:.2f} |

**HxCNN improves 24-h delta wind during RI-like windows: {hxcnn_improves_ri}**
"""
else:
    ri_lines = "\n*No RI-like windows (Δ24h wind ≥ 15.4 m/s) detected in this sample.*\n"

readme_text = f"""# ERA HxCNN Advisor Demo Results
**Storm:** Hurricane Beryl (C5)  |  **IC:** {HURRICANE_PREFIX}  |  **Timesteps:** {n_samples} (6-h intervals, {fhours[0]}–{fhours[-1]} h)

## Wind Speed vs HURDAT
| Model | MAE (m/s) | RMSE (m/s) | Bias (m/s) | Corr |
|---|---|---|---|---|
| Raw FCN-ERA | {fcn_ws['mae']:.2f} | {fcn_ws['rmse']:.2f} | {fcn_ws['bias']:+.2f} | {fcn_ws['corr']:.3f} |
| HxCNN-ERA | {hxcnn_ws['mae']:.2f} | {hxcnn_ws['rmse']:.2f} | {hxcnn_ws['bias']:+.2f} | {hxcnn_ws['corr']:.3f} |

RMSE improvement: **{ws_improv_pct:.1f}%**

## MSLP vs HURDAT
| Model | MAE (hPa) | RMSE (hPa) | Bias (hPa) | Corr |
|---|---|---|---|---|
| Raw FCN-ERA | {fcn_mslp['mae']:.2f} | {fcn_mslp['rmse']:.2f} | {fcn_mslp['bias']:+.2f} | {fcn_mslp['corr']:.3f} |
| HxCNN-ERA | {hxcnn_mslp['mae']:.2f} | {hxcnn_mslp['rmse']:.2f} | {hxcnn_mslp['bias']:+.2f} | {hxcnn_mslp['corr']:.3f} |

RMSE improvement: **{mslp_improv_pct:.1f}%**
{ri_lines}
## Output files
| File | Description |
|---|---|
| `hie_era_hxunet_predictions.csv` | Per-timestep HURDAT / FCN-ERA / HxCNN-ERA values |
| `hie_era_hxunet_metrics.csv` | MAE / RMSE / bias / correlation table |
| `hie_era_hxunet_wind_mslp_timeseries.png` | Wind speed and MSLP time series |
| `hie_era_hxunet_delta24_ri.png` | 24-hour delta wind with RI threshold |

## Notes
- **HxCNN-ERA**: CNN bias-correction model trained on ERA5-based FourCastNet output
  (`bias_correction_checkpoints_ERA/CNN_checkpoints.pth`)
- **Raw FCN-ERA**: FourCastNet-ERA predictions before bias correction (from `files/*_ERA.csv`)
- **HURDAT**: NHC best-track ground truth
- No training performed; all inference uses the pre-trained checkpoint
"""

readme_path = os.path.join(OUTPUT_DIR, 'README_results_summary.md')
with open(readme_path, 'w') as f:
    f.write(readme_text)
print(f"Saved {readme_path}")

# ── Terminal summary ──────────────────────────────────────────────────────────
W = 65
print('\n' + '=' * W)
print('ADVISOR DEMO SUMMARY  (ERA HxCNN vs HURDAT)'.center(W))
print('=' * W)
print(f"  Timesteps / samples : {n_samples}")
print(f"\n  Wind Speed RMSE (m/s)")
print(f"    Raw FCN-ERA  : {fcn_ws['rmse']:.2f}")
print(f"    HxCNN-ERA    : {hxcnn_ws['rmse']:.2f}   ({ws_improv_pct:+.1f}% vs FCN)")
print(f"\n  MSLP RMSE (hPa)")
print(f"    Raw FCN-ERA  : {fcn_mslp['rmse']:.2f}")
print(f"    HxCNN-ERA    : {hxcnn_mslp['rmse']:.2f}   ({mslp_improv_pct:+.1f}% vs FCN)")
if hxcnn_improves_ri is not None:
    print(f"\n  24-h delta wind (RI windows):")
    print(f"    FCN MAE   : {fcn_ri_mae:.2f} m/s")
    print(f"    HxCNN MAE : {hxcnn_ri_mae:.2f} m/s")
    print(f"    HxCNN improves RI prediction: {hxcnn_improves_ri}")
else:
    print(f"\n  24-h delta wind: no RI-like windows in this sample")
print('=' * W)
print(f"\n  Outputs -> {OUTPUT_DIR}")
