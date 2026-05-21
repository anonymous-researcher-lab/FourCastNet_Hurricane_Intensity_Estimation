# ERA HxCNN Advisor Demo Results
**Storm:** Hurricane Beryl (C5)  |  **IC:** 2024C5Beryl_2024063006  |  **Timesteps:** 20 (6-h intervals, 0–114 h)

## Wind Speed vs HURDAT
| Model | MAE (m/s) | RMSE (m/s) | Bias (m/s) | Corr |
|---|---|---|---|---|
| Raw FCN-ERA | 38.69 | 39.52 | -38.69 | 0.376 |
| HxCNN-ERA | 10.14 | 12.54 | +9.88 | 0.445 |

RMSE improvement: **68.3%**

## MSLP vs HURDAT
| Model | MAE (hPa) | RMSE (hPa) | Bias (hPa) | Corr |
|---|---|---|---|---|
| Raw FCN-ERA | 39.89 | 42.05 | +39.89 | 0.049 |
| HxCNN-ERA | 15.58 | 19.05 | -15.04 | 0.379 |

RMSE improvement: **54.7%**

### RI-like window (Δ24h wind ≥ 15.4 m/s)
| Model | MAE (m/s) |
|---|---|
| Raw FCN-ERA | 14.35 |
| HxCNN-ERA | 15.43 |

**HxCNN improves 24-h delta wind during RI-like windows: False**

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
