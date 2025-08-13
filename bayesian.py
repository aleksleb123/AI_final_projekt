import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir))  # All/
sys.path.insert(0, parent_dir)



import torch
import pandas as pd
from pandas.core.internals.managers import BlockManager
from utils.data.datasets import ADNIDataset

torch.serialization.add_safe_globals([
    ADNIDataset,
    pd.DataFrame,
    BlockManager
])





# ✅ Celovita verzija threshold_xarray.py
import os
import tomli as toml
import numpy as np
import xarray as xr
import torch
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, brier_score_loss
import matplotlib.pyplot as plt
import seaborn as sns

from utils.system import force_init_cudnn
from utils.ensemble import load_models, prune_models

# Load config
if os.getenv('ADL_CONFIG_PATH') is None:
    with open('config.toml', 'rb') as f:
        config = toml.load(f)
else:
    with open(os.getenv('ADL_CONFIG_PATH'), 'rb') as f:
        config = toml.load(f)

V4_PATH = os.path.join(config['paths']['model_output'], config['ensemble']['name'], 'v4')
os.makedirs(V4_PATH, exist_ok=True)

models_dir = os.path.join(config['paths']['model_output'], config['ensemble']['name'], 'models')
models, model_descs = load_models(models_dir, config['training']['device'])
models, model_descs = prune_models(models, model_descs, models_dir, config['ensemble']['prune_threshold'])

# Load datasets
test_dataset = torch.load(
    os.path.join(config['paths']['model_output'], config['ensemble']['name'], 'test_dataset.pt'),
    weights_only=False
)

val_dataset = torch.load(
    os.path.join(config['paths']['model_output'], config['ensemble']['name'], 'val_dataset.pt'),
    weights_only=False
)


# Run ensemble and save predictions
def run_ensemble(dataset, name):
    all_outputs = []
    all_targets = []
    for data, target in tqdm(dataset, desc=f"Predicting on {name}"):
        mri, xls = data
        mri = mri.unsqueeze(0)
        xls = xls.unsqueeze(0)
        data = (mri, xls)
        outputs = [model(data)[0][0].item() for model in models]
        all_outputs.append(outputs)
        all_targets.append(target[1].item())
    preds = xr.DataArray(np.array(all_outputs), dims=["sample", "model"], name=f"{name}_predictions")
    preds.to_netcdf(os.path.join(V4_PATH, f"{name}_predictions.nc"))
    return np.array(all_targets), np.array(all_outputs)

test_labels, test_outputs = run_ensemble(test_dataset, "test")
val_labels, val_outputs = run_ensemble(val_dataset, "val")

# Combine
y_true = np.concatenate([test_labels, val_labels])
y_pred_all = np.concatenate([test_outputs, val_outputs])
y_pred_mean = y_pred_all.mean(axis=1)
y_pred_std = y_pred_all.std(axis=1)
entropy = - y_pred_mean * np.log2(y_pred_mean + 1e-12) - (1 - y_pred_mean) * np.log2(1 - y_pred_mean + 1e-12)
confidence = np.maximum(y_pred_mean, 1 - y_pred_mean)

# Save ensemble statistics
ensemble_stats = xr.Dataset({
    "mean": ("sample", y_pred_mean),
    "stdev": ("sample", y_pred_std),
    "entropy": ("sample", entropy),
    "confidence": ("sample", confidence),
    "predicted": ("sample", np.round(y_pred_mean)),
    "actual": ("sample", y_true),
    "correct": ("sample", (np.round(y_pred_mean) == y_true).astype(int))
})
ensemble_stats.to_netcdf(os.path.join(V4_PATH, "ensemble_statistics.nc"))

# Če obstajajo MC dropout rezultati, primerjaj
mc_stats_path = os.path.join(V4_PATH, "mc_statistics.nc")
if os.path.exists(mc_stats_path):
    mc_stats = xr.load_dataset(mc_stats_path)
    print("\n📊 MC Dropout Metrics:")
    print(f"→ AUC:    {roc_auc_score(mc_stats['actual'], mc_stats['mean']):.4f}")
    print(f"→ ACC:    {accuracy_score(mc_stats['actual'], np.round(mc_stats['mean'])):.4f}")
    print(f"→ F1:     {f1_score(mc_stats['actual'], np.round(mc_stats['mean'])):.4f}")
    print(f"→ Brier:  {brier_score_loss(mc_stats['actual'], mc_stats['mean']):.4f}")

    # Primerjava pokritosti
    plt.figure(figsize=(6,4))
    sns.histplot(ensemble_stats["stdev"], label="Ensemble", color="blue", stat="density", bins=20, kde=True)
    sns.histplot(mc_stats["stdev"], label="MC Dropout", color="orange", stat="density", bins=20, kde=True, alpha=0.7)
    plt.title("Primerjava negotovosti (standardna deviacija)")
    plt.xlabel("Stdev")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(V4_PATH, "compare_stdev_histogram.png"))
    plt.close()

    # Reliability diagram za MC
    def reliability_curve(probs, labels, n_bins=10):
        bins = np.linspace(0.0, 1.0, n_bins + 1)
        binids = np.digitize(probs, bins) - 1
        prob_true = np.zeros(n_bins)
        prob_pred = np.zeros(n_bins)
        for i in range(n_bins):
            mask = binids == i
            if np.any(mask):
                prob_true[i] = labels[mask].mean()
                prob_pred[i] = probs[mask].mean()
        return prob_pred, prob_true

    prob_pred, prob_true = reliability_curve(mc_stats['mean'].values, mc_stats['actual'].values)
    plt.figure(figsize=(5,5))
    plt.plot(prob_pred, prob_true, marker='o', label="MC Dropout")
    plt.plot([0,1],[0,1], '--', color='gray')
    plt.title("Reliability diagram (MC Dropout)")
    plt.xlabel("Confidence")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(V4_PATH, "reliability_diagram_mc.png"))
    plt.close()

# Ensemble Metrics
print("\n📊 Ensemble Metrics:")
print(f"→ AUC:    {roc_auc_score(y_true, y_pred_mean):.4f}")
print(f"→ ACC:    {accuracy_score(y_true, np.round(y_pred_mean)):.4f}")
print(f"→ F1:     {f1_score(y_true, np.round(y_pred_mean)):.4f}")
print(f"→ Brier:  {brier_score_loss(y_true, y_pred_mean):.4f}")

# Reliability diagram
def reliability_curve(probs, labels, n_bins=10):
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    binids = np.digitize(probs, bins) - 1
    prob_true = np.zeros(n_bins)
    prob_pred = np.zeros(n_bins)
    for i in range(n_bins):
        mask = binids == i
        if np.any(mask):
            prob_true[i] = labels[mask].mean()
            prob_pred[i] = probs[mask].mean()
    return prob_pred, prob_true

prob_pred, prob_true = reliability_curve(y_pred_mean, y_true)
plt.figure(figsize=(5,5))
plt.plot(prob_pred, prob_true, marker='o', label="Ensemble")
plt.plot([0,1],[0,1], '--', color='gray')
plt.title("Reliability diagram (Ensemble)")
plt.xlabel("Confidence")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(V4_PATH, "reliability_diagram.png"))
plt.close()

# Confidence histogram
plt.figure(figsize=(6,4))
sns.histplot(y_pred_mean, bins=20, kde=True, color='steelblue')
plt.title("Histogram model confidence")
plt.xlabel("Confidence")
plt.ylabel("Frequency")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(V4_PATH, "confidence_histogram.png"))
plt.close()

# Entropy histogram
plt.figure(figsize=(6,4))
sns.histplot(entropy, bins=20, kde=True, color='darkred')
plt.title("Histogram entropije")
plt.xlabel("Entropy")
plt.ylabel("Frequency")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(V4_PATH, "entropy_histogram.png"))
plt.close()

print("\n✅ Vse metrika, statistike in grafi shranjeni v:", V4_PATH)
