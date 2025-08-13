
import os
import tomli as toml
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, brier_score_loss
import pickle

# Load config
if os.getenv('ADL_CONFIG_PATH') is None:
    with open('config.toml', 'rb') as f:
        config = toml.load(f)
else:
    with open(os.getenv('ADL_CONFIG_PATH'), 'rb') as f:
        config = toml.load(f)

root_out = os.path.join(config['paths']['model_output'], config['ensemble']['name'])
v3_path = os.path.join(root_out, 'v3')
v4_path = os.path.join(root_out, 'v4')
os.makedirs(v3_path, exist_ok=True)

def _normalize_predictions_dims(da: xr.DataArray) -> xr.DataArray:
    # Ensure first dim is data_id
    if 'data_id' not in da.dims:
        if 'sample' in da.dims:
            da = da.rename({'sample': 'data_id'})
        else:
            da = da.rename({da.dims[0]: 'data_id'})
    if 'model_id' not in da.dims and 'model' in da.dims:
        da = da.rename({'model': 'model_id'})
    if 'data_id' not in da.coords:
        da = da.assign_coords(data_id=np.arange(da.sizes['data_id']))
    if 'model_id' in da.dims and 'model_id' not in da.coords:
        da = da.assign_coords(model_id=np.arange(da.sizes['model_id']))
    return da

def _prune_by_ids(da: xr.DataArray, excluded_ids):
    if not excluded_ids:
        return da
    return da.where(~da.data_id.isin(excluded_ids), drop=True)

def _extract_positive_predictions(da: xr.DataArray) -> np.ndarray:
    # 3D with prediction_value coord -> select 'positive_prediction'
    if 'prediction_value' in da.dims or 'prediction_value' in da.coords:
        if 'prediction_value' in da.coords:
            if 'positive_prediction' in da.coords['prediction_value'].values:
                da_pos = da.sel(prediction_value='positive_prediction')
            else:
                # fallback: assume index 1
                da_pos = da.isel(prediction_value=1)
        else:
            da_pos = da.isel(prediction_value=1)
        return da_pos.values  # shape (samples, models)
    # 2D: already (samples, models) positive probs
    return da.values

def _extract_labels_from_da(da: xr.DataArray) -> np.ndarray:
    # If 3D with 'positive_actual', take first model entry (same across models)
    if 'prediction_value' in da.dims or 'prediction_value' in da.coords:
        if 'prediction_value' in da.coords and 'positive_actual' in da.coords['prediction_value'].values:
            lab_da = da.sel(prediction_value='positive_actual').isel(model_id=0)
            return lab_da.values.astype(int)
        else:
            # Fallback: index 3 assumed to be positive_actual
            lab_da = da.isel(prediction_value=3).isel(model_id=0)
            return lab_da.values.astype(int)
    # No labels embedded (2D), return None
    return None

# Load prediction files
preds_test = xr.open_dataarray(os.path.join(v4_path, "test_predictions.nc"))
preds_val  = xr.open_dataarray(os.path.join(v4_path, "val_predictions.nc"))

# Normalize dims
preds_test = _normalize_predictions_dims(preds_test)
preds_val  = _normalize_predictions_dims(preds_val)

# Prune if configured
excluded_ids = config['ensemble'].get('excluded_ids', [])
if config['operation'].get('exclude_blank_ids', False) and len(excluded_ids) > 0:
    preds_test = _prune_by_ids(preds_test, excluded_ids)
    preds_val  = _prune_by_ids(preds_val, excluded_ids)

# Extract labels directly from prediction files (ensures perfect alignment)
labels_test = _extract_labels_from_da(preds_test)
labels_val  = _extract_labels_from_da(preds_val)

# If labels not embedded (e.g., 2D files), fall back to ensemble_statistics.nc
if labels_test is None or labels_val is None:
    ens_stats = xr.load_dataset(os.path.join(v4_path, "ensemble_statistics.nc"))
    labels_all = ens_stats['actual'].values.astype(int)
else:
    labels_all = np.concatenate([labels_test, labels_val], axis=0)

# Extract positive-class predictions (shape -> (samples, models))
arr_test = _extract_positive_predictions(preds_test)
arr_val  = _extract_positive_predictions(preds_val)

# Concatenate samples
all_outputs = np.concatenate([arr_test, arr_val], axis=0)

# Persist raw combined predictions for re-use
with open(os.path.join(v3_path, "full_predictions.pk"), 'wb') as f:
    pickle.dump(all_outputs, f)

# Sanity check alignment
if all_outputs.shape[0] != labels_all.shape[0]:
    raise RuntimeError(f"Length mismatch: predictions have {all_outputs.shape[0]} samples, labels have {labels_all.shape[0]}")

# Compute metrics vs number of models
def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    binids = np.digitize(probs, bins) - 1
    ece = 0.0
    for b in range(n_bins):
        mask = (binids == b)
        if np.any(mask):
            acc = (np.round(probs[mask]) == labels[mask]).mean()
            conf = probs[mask].mean()
            ece += np.abs(acc - conf) * mask.mean()
    return float(ece)

results = []
n_models = all_outputs.shape[1]

for i in range(1, n_models + 1):
    subset = all_outputs[:, :i]      # (samples, i)
    mean   = subset.mean(axis=1)     # (samples,)

    ece    = compute_ece(mean, labels_all, n_bins=10)
    acc    = accuracy_score(labels_all, np.round(mean))
    f1     = f1_score(labels_all, np.round(mean))
    auc    = roc_auc_score(labels_all, mean)
    brier  = brier_score_loss(labels_all, mean)

    results.append({
        "n_models": i,
        "ece": ece,
        "acc": acc,
        "f1": f1,
        "auc": auc,
        "brier": brier
    })

results_df = pd.DataFrame(results)
csv_path = os.path.join(v3_path, "sensitivity_analysis.csv")
results_df.to_csv(csv_path, index=False)

# Plot ECE
plt.figure(figsize=(6,4))
plt.plot(results_df["n_models"], results_df["ece"], marker='o', label="Ensemble")
plt.title("ECE vs število modelov")
plt.xlabel("# modelov")
plt.ylabel("ECE")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(v3_path, "sensitivity_analysis_ece.png"))
plt.close()

# Plot Accuracy
plt.figure(figsize=(6,4))
plt.plot(results_df["n_models"], results_df["acc"], marker='o', label="Accuracy")
plt.title("Točnost vs število modelov")
plt.xlabel("# modelov")
plt.ylabel("Accuracy")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(v3_path, "sensitivity_analysis_accuracy.png"))
plt.close()

print("\\n✅ Sensitivity analiza zaključena in shranjena v:", v3_path)

# ---------- Reliability diagrams za 1..N modelov ----------
import os

rel_dir = os.path.join(v3_path, "reliability")
os.makedirs(rel_dir, exist_ok=True)

def reliability_points(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10):
    """Vrne (bin_centers, avg_conf, avg_acc) za reliability diagram."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    bin_ids = np.digitize(probs, bins[1:-1], right=False)

    avg_conf = np.full(n_bins, np.nan)
    avg_acc  = np.full(n_bins, np.nan)
    for b in range(n_bins):
        m = (bin_ids == b)
        if np.any(m):
            avg_conf[b] = probs[m].mean()
            avg_acc[b]  = (np.round(probs[m]) == labels[m]).mean()
    return bin_centers, avg_conf, avg_acc

for i in range(1, n_models + 1):
    mean_i = all_outputs[:, :i].mean(axis=1)          # (samples,)
    centers, confs, accs = reliability_points(mean_i, labels_all, n_bins=10)

    # plot
    plt.figure(figsize=(6,6))
    plt.plot([0,1],[0,1], linestyle="--")             # idealna diagonala
    valid = ~np.isnan(accs)
    plt.scatter(confs[valid], accs[valid], s=20)
    plt.xlabel("Povprečna samozavest (bin)")
    plt.ylabel("Povprečna točnost (bin)")
    plt.title(f"Reliability diagram – {i} model(ov)")
    plt.grid(True)
    plt.tight_layout()
    outp = os.path.join(rel_dir, f"reliability_{i:02d}.png")
    plt.savefig(outp)
    plt.close()
# ----------------------------------------------------------
