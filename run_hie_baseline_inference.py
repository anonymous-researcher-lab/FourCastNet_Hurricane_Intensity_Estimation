#!/usr/bin/env python3
"""
run_hie_baseline_inference.py
Inference-only baseline for Hurricane Intensity Estimation (HIE).
Converted from example/run_inference.ipynb — no training, ERA data first.

Outputs (hie_baseline_outputs/):
  hie_predictions_comparison.csv
  wind_pressure_timeseries.png
  delta24_wind_comparison.png
  summary_metrics.csv
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
    BiasCorrectionModel, VisionGNN,
    get_means_stds_from_hurdat_tracks,
    run_inference, calculate_mean_differences,
)

OUTPUT_DIR = os.path.join(REPO, 'hie_baseline_outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)

HURRICANE_PREFIX = '2024C5Beryl_2024063006'

ERA_SCALING   = os.path.join(REPO, 'ERA_scaling')
MERRA_SCALING = os.path.join(REPO, 'MERRA_scaling')
CKPT_ERA      = os.path.join(REPO, 'bias_correction_checkpoints_ERA')
CKPT_MERRA    = os.path.join(REPO, 'bias_correction_checkpoints_MERRA')
FILES_DIR     = os.path.join(REPO, 'files')

# ─── File collections ────────────────────────────────────────────────────────
era_nc_files   = sorted(glob.glob(os.path.join(FILES_DIR, 'ERA*.nc')))
merra_nc_files = sorted(glob.glob(os.path.join(FILES_DIR, 'MERRA*.nc')))
hurdat_csvs    = sorted(glob.glob(os.path.join(FILES_DIR, '*HURDAT.csv')))
era_csvs       = sorted(glob.glob(os.path.join(FILES_DIR, '*ERA.csv')))
merra_csvs     = sorted(glob.glob(os.path.join(FILES_DIR, '*MERRA.csv')))

files_era = {
    'cropped_data_files': era_nc_files,
    'hurdat_track_files': hurdat_csvs,
    'merra_track_files':  era_csvs,   # raw FCN-ERA predictions stored here
}
files_merra = {
    'cropped_data_files': merra_nc_files,
    'hurdat_track_files': hurdat_csvs,
    'merra_track_files':  merra_csvs,  # raw FCN-MERRA predictions stored here
}

# Pass HURDAT files explicitly to avoid the hardcoded NAS path in src.py
means_track_files = get_means_stds_from_hurdat_tracks(track_data_files=hurdat_csvs)

# ─── Models ──────────────────────────────────────────────────────────────────
# UNet checkpoint does not exist in this repo — CNN and GNN only.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

models_era = [
    BiasCorrectionModel(in_channels=20).to(device),
    VisionGNN(in_channels=20, out_channels=2).to(device),
]
model_paths_era = [
    os.path.join(CKPT_ERA, f'{m.name}_checkpoints.pth') for m in models_era
]

models_merra = [
    BiasCorrectionModel(in_channels=38).to(device),
    VisionGNN(in_channels=38, out_channels=2).to(device),
]
model_paths_merra = [
    os.path.join(CKPT_MERRA, f'{m.name}_checkpoints.pth') for m in models_merra
]

# ─── Run inference ────────────────────────────────────────────────────────────
print("\nRunning ERA inference ...")
output_tracks_era = []
for model, mpath in zip(models_era, model_paths_era):
    res = run_inference(
        model=model, model_weights=mpath,
        files=files_era, hurricane_prefix=HURRICANE_PREFIX,
        means_track_files=means_track_files,
        cropped_data_meanstd_folder=ERA_SCALING,
    )
    output_tracks_era.append(res)
    print(f"  HxCNN/GNN-ERA [{model.name}] done")

print("\nRunning MERRA inference ...")
output_tracks_merra = []
for model, mpath in zip(models_merra, model_paths_merra):
    res = run_inference(
        model=model, model_weights=mpath,
        files=files_merra, hurricane_prefix=HURRICANE_PREFIX,
        means_track_files=means_track_files,
        cropped_data_meanstd_folder=MERRA_SCALING,
    )
    output_tracks_merra.append(res)
    print(f"  HxCNN/GNN-MERRA [{model.name}] done")

fhours = np.arange(len(output_tracks_era[0]['merra'])) * 6

# Unpack results — index 0 = CNN, index 1 = GNN
hurdat      = output_tracks_era[0]['hurdat']
fcn_era     = output_tracks_era[0]['merra']    # raw FCN-ERA
fcn_merra   = output_tracks_merra[0]['merra']  # raw FCN-MERRA
cnn_era     = output_tracks_era[0]['model']
gnn_era     = output_tracks_era[1]['model']
cnn_merra   = output_tracks_merra[0]['model']
gnn_merra   = output_tracks_merra[1]['model']

diffs_era   = [calculate_mean_differences(t) for t in output_tracks_era]
diffs_merra = [calculate_mean_differences(t) for t in output_tracks_merra]

# ─── 1. Save comparison CSV ───────────────────────────────────────────────────
df = pd.DataFrame({
    'forecast_hour':        fhours,
    'hurdat_vmax_ms':       hurdat['vmax'].values,
    'hurdat_mslp_hPa':      hurdat['mslp'].values,
    'fcn_era_ws_ms':        fcn_era['ws'].values,
    'fcn_era_mslp_hPa':     fcn_era['mslp'].values,
    'fcn_merra_ws_ms':      fcn_merra['ws'].values,
    'fcn_merra_mslp_hPa':   fcn_merra['mslp'].values,
    'hxcnn_era_ws_ms':      cnn_era['ws'].values,
    'hxcnn_era_mslp_hPa':   cnn_era['mslp'].values,
    'hxgnn_era_ws_ms':      gnn_era['ws'].values,
    'hxgnn_era_mslp_hPa':   gnn_era['mslp'].values,
    'hxcnn_merra_ws_ms':    cnn_merra['ws'].values,
    'hxcnn_merra_mslp_hPa': cnn_merra['mslp'].values,
    'hxgnn_merra_ws_ms':    gnn_merra['ws'].values,
    'hxgnn_merra_mslp_hPa': gnn_merra['mslp'].values,
})
csv_path = os.path.join(OUTPUT_DIR, 'hie_predictions_comparison.csv')
df.to_csv(csv_path, index=False)
print(f"\nSaved {csv_path}  ({len(df)} rows)")

# ─── 2. Wind speed & MSLP time series plot ────────────────────────────────────
colors = {'HURDAT': 'k', 'FCN': 'b', 'HxCNN': 'r', 'HxGNN': 'g',
          'FCN_M': 'steelblue', 'HxCNN_M': 'tomato', 'HxGNN_M': 'seagreen'}

fig, axs = plt.subplots(2, 2, figsize=(16, 9))

# ERA wind speed
axs[0, 0].plot(fhours, hurdat['vmax'].values,    'k-o', label='HURDAT')
axs[0, 0].plot(fhours, fcn_era['ws'].values,     'b-o', label=f'FCN-ERA | bias {diffs_era[0]["merra_ws"]:.1f} m/s')
axs[0, 0].plot(fhours, cnn_era['ws'].values,     'r-o', label=f'HxCNN-ERA | bias {diffs_era[0]["model_ws"]:.1f} m/s')
axs[0, 0].plot(fhours, gnn_era['ws'].values,     'g-o', label=f'HxGNN-ERA | bias {diffs_era[1]["model_ws"]:.1f} m/s')
axs[0, 0].set_ylabel('Wind Speed (m/s)')
axs[0, 0].set_title('(a) ERA-FCN Wind Speed')
axs[0, 0].legend(fontsize=9, frameon=False)
axs[0, 0].set_xticks(np.arange(0, fhours[-1] + 1, 12))

# ERA MSLP
axs[0, 1].plot(fhours, hurdat['mslp'].values,    'k-o', label='HURDAT')
axs[0, 1].plot(fhours, fcn_era['mslp'].values,   'b-o', label=f'FCN-ERA | bias {diffs_era[0]["merra_mslp"]:.1f} hPa')
axs[0, 1].plot(fhours, cnn_era['mslp'].values,   'r-o', label=f'HxCNN-ERA | bias {diffs_era[0]["model_mslp"]:.1f} hPa')
axs[0, 1].plot(fhours, gnn_era['mslp'].values,   'g-o', label=f'HxGNN-ERA | bias {diffs_era[1]["model_mslp"]:.1f} hPa')
axs[0, 1].set_ylabel('MSLP (hPa)')
axs[0, 1].set_title('(b) ERA-FCN MSLP')
axs[0, 1].legend(fontsize=9, frameon=False)
axs[0, 1].set_xticks(np.arange(0, fhours[-1] + 1, 12))

# MERRA wind speed
axs[1, 0].plot(fhours, hurdat['vmax'].values,      'k-o', label='HURDAT')
axs[1, 0].plot(fhours, fcn_merra['ws'].values,     'b-o', label=f'FCN-MERRA | bias {diffs_merra[0]["merra_ws"]:.1f} m/s')
axs[1, 0].plot(fhours, cnn_merra['ws'].values,     'r-o', label=f'HxCNN-MERRA | bias {diffs_merra[0]["model_ws"]:.1f} m/s')
axs[1, 0].plot(fhours, gnn_merra['ws'].values,     'g-o', label=f'HxGNN-MERRA | bias {diffs_merra[1]["model_ws"]:.1f} m/s')
axs[1, 0].set_ylabel('Wind Speed (m/s)')
axs[1, 0].set_xlabel('Forecast Hour')
axs[1, 0].set_title('(c) MERRA-FCN Wind Speed')
axs[1, 0].legend(fontsize=9, frameon=False)
axs[1, 0].set_xticks(np.arange(0, fhours[-1] + 1, 12))

# MERRA MSLP
axs[1, 1].plot(fhours, hurdat['mslp'].values,      'k-o', label='HURDAT')
axs[1, 1].plot(fhours, fcn_merra['mslp'].values,   'b-o', label=f'FCN-MERRA | bias {diffs_merra[0]["merra_mslp"]:.1f} hPa')
axs[1, 1].plot(fhours, cnn_merra['mslp'].values,   'r-o', label=f'HxCNN-MERRA | bias {diffs_merra[0]["model_mslp"]:.1f} hPa')
axs[1, 1].plot(fhours, gnn_merra['mslp'].values,   'g-o', label=f'HxGNN-MERRA | bias {diffs_merra[1]["model_mslp"]:.1f} hPa')
axs[1, 1].set_ylabel('MSLP (hPa)')
axs[1, 1].set_xlabel('Forecast Hour')
axs[1, 1].set_title('(d) MERRA-FCN MSLP')
axs[1, 1].legend(fontsize=9, frameon=False)
axs[1, 1].set_xticks(np.arange(0, fhours[-1] + 1, 12))

plt.suptitle(f'Hurricane Beryl  |  IC: {HURRICANE_PREFIX}', fontweight='bold')
plt.tight_layout()
ts_path = os.path.join(OUTPUT_DIR, 'wind_pressure_timeseries.png')
plt.savefig(ts_path, dpi=150)
plt.close()
print(f"Saved {ts_path}")

# ─── 3. 24-hour wind change (RI evaluation) ───────────────────────────────────
def delta24(arr):
    """Forward 24-h difference (4 × 6-h steps)."""
    a = np.asarray(arr, dtype=float)
    d = np.full_like(a, np.nan)
    d[:-4] = a[4:] - a[:-4]
    return d

d_hurdat    = delta24(hurdat['vmax'].values)
d_fcn_era   = delta24(fcn_era['ws'].values)
d_cnn_era   = delta24(cnn_era['ws'].values)
d_gnn_era   = delta24(gnn_era['ws'].values)
d_fcn_merra = delta24(fcn_merra['ws'].values)
d_cnn_merra = delta24(cnn_merra['ws'].values)
d_gnn_merra = delta24(gnn_merra['ws'].values)

valid = ~np.isnan(d_hurdat)

fig, axs = plt.subplots(1, 2, figsize=(14, 5))
RI_THRESH = 15.4  # ~30 kt / 24 h

for ax, (d_fcn, d_cnn, d_gnn, tag) in zip(
    axs,
    [(d_fcn_era, d_cnn_era, d_gnn_era, 'ERA'),
     (d_fcn_merra, d_cnn_merra, d_gnn_merra, 'MERRA')],
):
    ax.plot(fhours[valid], d_hurdat[valid],   'k-o', label='HURDAT')
    ax.plot(fhours[valid], d_fcn[valid],      'b-o', label=f'FCN-{tag}')
    ax.plot(fhours[valid], d_cnn[valid],      'r-o', label=f'HxCNN-{tag}')
    ax.plot(fhours[valid], d_gnn[valid],      'g-o', label=f'HxGNN-{tag}')
    ax.axhline(0,         color='gray',   linestyle='--', linewidth=0.8)
    ax.axhline(RI_THRESH, color='purple', linestyle=':',  linewidth=1.2,
               label=f'RI threshold ({RI_THRESH} m/s)')
    ax.set_xlabel('Forecast Hour')
    ax.set_ylabel('Δ Wind Speed (m/s) / 24 h')
    ax.set_title(f'24-h Wind Change — {tag}')
    ax.legend(fontsize=9, frameon=False)
    ax.set_xticks(np.arange(0, fhours[-1] + 1, 12))

plt.suptitle(f'RI Evaluation — Hurricane Beryl  |  IC: {HURRICANE_PREFIX}', fontweight='bold')
plt.tight_layout()
ri_path = os.path.join(OUTPUT_DIR, 'delta24_wind_comparison.png')
plt.savefig(ri_path, dpi=150)
plt.close()
print(f"Saved {ri_path}")

# ─── 4. Metrics ───────────────────────────────────────────────────────────────
def metrics(pred, obs, name):
    mae  = mean_absolute_error(obs, pred)
    rmse = np.sqrt(mean_squared_error(obs, pred))
    bias = float(np.mean(pred - obs))
    corr, _ = pearsonr(pred, obs)
    return {'model': name, 'mae': mae, 'rmse': rmse, 'bias': bias, 'corr': corr}

hurdat_ws   = hurdat['vmax'].values.astype(float)
hurdat_mslp = hurdat['mslp'].values.astype(float)

rows = []
for arr, label in [
    (fcn_era['ws'].values,    'FCN_ERA_WS'),
    (cnn_era['ws'].values,    'HxCNN_ERA_WS'),
    (gnn_era['ws'].values,    'HxGNN_ERA_WS'),
    (fcn_merra['ws'].values,  'FCN_MERRA_WS'),
    (cnn_merra['ws'].values,  'HxCNN_MERRA_WS'),
    (gnn_merra['ws'].values,  'HxGNN_MERRA_WS'),
]:
    rows.append(metrics(arr.astype(float), hurdat_ws, label))

for arr, label in [
    (fcn_era['mslp'].values,    'FCN_ERA_MSLP'),
    (cnn_era['mslp'].values,    'HxCNN_ERA_MSLP'),
    (gnn_era['mslp'].values,    'HxGNN_ERA_MSLP'),
    (fcn_merra['mslp'].values,  'FCN_MERRA_MSLP'),
    (cnn_merra['mslp'].values,  'HxCNN_MERRA_MSLP'),
    (gnn_merra['mslp'].values,  'HxGNN_MERRA_MSLP'),
]:
    rows.append(metrics(arr.astype(float), hurdat_mslp, label))

metrics_df = pd.DataFrame(rows)
metrics_df.to_csv(os.path.join(OUTPUT_DIR, 'summary_metrics.csv'), index=False)

# ─── 5. Terminal summary ──────────────────────────────────────────────────────
W = 60
print('\n' + '=' * W)
print('SUMMARY METRICS  (vs HURDAT)'.center(W))
print('=' * W)
print(f"{'Model':<24} {'MAE':>7} {'RMSE':>7} {'Bias':>8} {'Corr':>7}  Unit")
print('-' * W)
for _, r in metrics_df.iterrows():
    unit = 'm/s' if r['model'].endswith('WS') else 'hPa'
    print(f"{r['model']:<24} {r['mae']:>7.2f} {r['rmse']:>7.2f} {r['bias']:>+8.2f} {r['corr']:>7.3f}  {unit}")
print('=' * W)
print(f"\nOutputs → {OUTPUT_DIR}")
