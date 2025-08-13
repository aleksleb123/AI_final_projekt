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

v3_path = os.path.join(config['paths']['model_output'], config['ensemble']['name'], 'v3')
os.makedirs(v3_path, exist_ok=True)

v4_path = os.path.join(config['paths']['model_output'], config['ensemble']['name'], 'v4')
ensemble_stats = xr.load_dataset(os.path.join(v4_path, "ensemble_statistics.nc"))

if os.path.exists(os.path.join(v3_path, "full_predictions.pk")):
    with open(os.path.join(v3_path, "full_predictions.pk"), 'rb') as f:
        all_outputs = pickle.load(f)
else:
        preds_ens = xr.open_dataarray(os.path.join(v4_path, "test_predictions.nc"))
        preds_val = xr.open_dataarray(os.path.join(v4_path, "val_predictions.nc"))

        # če obstaja dimenzija 'sample', jo preimenuj v 'data_id'
        if "data_id" not in preds_ens.dims and "sample" in preds_ens.dims:
            preds_ens = preds_ens.rename({"sample": "data_id"})
        if "data_id" not in preds_val.dims and "sample" in preds_val.dims:
            preds_val = preds_val.rename({"sample": "data_id"})

        arr_test = preds_ens.values
        arr_val = preds_val.values

        # če so podatki 3D (samples, models, 4) → vzemi index 1 = positive_prediction
        if arr_test.ndim == 3:
            arr_test = arr_test[:, :, 1]
        if arr_val.ndim == 3:
            arr_val = arr_val[:, :, 1]

        all_preds = np.concatenate([arr_test, arr_val], axis=0)

        with open(os.path.join(v3_path, "full_predictions.pk"), 'wb') as f:
            pickle.dump(all_preds, f)

        all_outputs = all_preds

    # zdaj preberemo prave labele iz ensemble_statistics.nc
labels = ensemble_stats['actual'].values

results = []

for i in range(1, all_outputs.shape[1] + 1):
    subset = all_outputs[:, :i]
    mean = subset.mean(axis=1)
    ece_bins = np.linspace(0, 1, 11)
    binids = np.digitize(mean, ece_bins) - 1
    ece = 0
    for b in range(10):
        bin_mask = binids == b
        if np.any(bin_mask):
            bin_acc = (np.round(mean[bin_mask]) == labels[bin_mask]).mean()
            bin_conf = mean[bin_mask].mean()
            ece += np.abs(bin_acc - bin_conf) * bin_mask.mean()

    acc = accuracy_score(labels, np.round(mean))
    f1 = f1_score(labels, np.round(mean))
    auc = roc_auc_score(labels, mean)
    brier = brier_score_loss(labels, mean)

    results.append({
        "n_models": i,
        "ece": ece,
        "acc": acc,
        "f1": f1,
        "auc": auc,
        "brier": brier
    })

results_df = pd.DataFrame(results)
results_df.to_csv(os.path.join(v3_path, "sensitivity_analysis.csv"), index=False)

# Plot ECE
plt.figure(figsize=(6,4))
plt.plot(results_df["n_models"], results_df["ece"], marker='o', label="Ensemble")

# Primerjaj z MC če obstaja
mc_path = os.path.join(v4_path, "mc_statistics.nc")
if os.path.exists(mc_path):
    mc_stats = xr.load_dataset(mc_path)
    mc_mean = mc_stats['mean'].values
    mc_labels = mc_stats['actual'].values

    # MC ECE
    mc_binids = np.digitize(mc_mean, ece_bins) - 1
    mc_ece = 0
    for b in range(10):
        bin_mask = mc_binids == b
        if np.any(bin_mask):
            bin_acc = (np.round(mc_mean[bin_mask]) == mc_labels[bin_mask]).mean()
            bin_conf = mc_mean[bin_mask].mean()
            mc_ece += np.abs(bin_acc - bin_conf) * bin_mask.mean()

    plt.axhline(mc_ece, linestyle='--', color='orange', label=f"MC Dropout ECE = {mc_ece:.3f}")

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

print("\n✅ Sensitivity analiza zaključena in shranjena v:", v3_path)