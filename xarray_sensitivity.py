import math
import itertools as it
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random as rand
import os

from sklearn.metrics import accuracy_score, roc_auc_score
import threshold_xarray as txr

# ================== KOMBINACIJE MODELOV ==================

def get_combinations(iterable, r, n_combinations):
    possible_combinations = math.comb(len(iterable), r)
    if n_combinations > possible_combinations:
        raise ValueError(
            f'Zahtevano kombinacij ({n_combinations}) presega možno število ({possible_combinations})'
        )
    elif n_combinations == possible_combinations:
        return list(it.combinations(iterable, r))
    else:
        combinations = set()
        while len(combinations) < n_combinations:
            combination = tuple(sorted(rand.sample(iterable, r)))
            combinations.add(combination)
        return list(combinations)

# ================== NASTAVITVE ==================

print('Loading Config...')
config = txr.load_config()
ENSEMBLE_PATH = os.path.join(config['paths']['model_output'], config['ensemble']['name'])
V4_PATH = os.path.join(ENSEMBLE_PATH, 'v4')
os.makedirs(V4_PATH, exist_ok=True)
print('Config Loaded')

# ================== NALAGANJE PODATKOV ==================

test_predictions = xr.open_dataarray(os.path.join(V4_PATH, 'test_predictions.nc'))
val_predictions = xr.open_dataarray(os.path.join(V4_PATH, 'val_predictions.nc'))

# Preimenuj dimenzije, če niso pravilno imenovane
if 'dim_0' in test_predictions.dims:
    test_predictions = test_predictions.rename({'dim_0': 'data_id'})
if 'dim_0' in val_predictions.dims:
    val_predictions = val_predictions.rename({'dim_0': 'data_id'})

# Odstrani izločene ID-je, če je zahtevano
if config['operation']['exclude_blank_ids']:
    excluded_data_ids = config['ensemble']['excluded_ids']
    test_predictions = txr.prune_data(test_predictions, excluded_data_ids)
    val_predictions = txr.prune_data(val_predictions, excluded_data_ids)

# Združi test in val podatke
predictions = xr.concat([test_predictions, val_predictions], dim='data_id')

# ================== PRIPRAVA KOMBINACIJ ==================

models = list(predictions.coords['model_id'].values.tolist())
combos = {}
for i in range(1, len(models) + 1):  # i = število modelov v ansamblu
    combos[i] = get_combinations(models, i, min(50, math.comb(len(models), i)))

# ================== IZRAČUN METRIK ==================

results = []

for num_models, model_combinations in combos.items():
    print(f"\n📊 {num_models} modeli ({len(model_combinations)} kombinacij)")
    for i, model_combination in enumerate(model_combinations):
        # Pretvori indekse v natančen tip, ki je uporabljen v xarray (int64 → int)
        model_combination = [int(m) for m in model_combination]
        model_preds = predictions.sel(model_id=model_combination)

        pos_preds = model_preds.sel(prediction_value="positive_prediction")
        mean_pred = pos_preds.mean(dim="model_id")

        true_labels = model_preds.sel(prediction_value="positive_actual").isel(model_id=0)

        y_true = true_labels.values.astype(int)
        y_pred = (mean_pred.values > 0.5).astype(int)

        acc = accuracy_score(y_true, y_pred)
        try:
            auc = roc_auc_score(y_true, mean_pred.values)
        except ValueError:
            auc = np.nan  # samo ena klasa – ni mogoče izračunati AUC

        results.append({
            "n_models": num_models,
            "combo_index": i,
            "accuracy": acc,
            "auc": auc
        })

# ================== SHRANJEVANJE ==================

df = pd.DataFrame(results)

# ===== Povzetek po številu modelov: mean, std, min, max + vizualizacija razpona =====
summary = (
    df.groupby("n_models")
      .agg(accuracy_mean=("accuracy", "mean"),
           accuracy_std =("accuracy", "std"),
           accuracy_min =("accuracy", "min"),
           accuracy_max =("accuracy", "max"),
           auc_mean     =("auc", "mean"),
           auc_std      =("auc", "std"),
           auc_min      =("auc", "min"),
           auc_max      =("auc", "max"))
      .reset_index()
)

# shrani tabelo povzetkov
summary_path = os.path.join(V4_PATH, "sensitivity_summary_stats.csv")
summary.to_csv(summary_path, index=False)

# --- Accuracy: povprečje + SD + min–max pas ---
acc_sum = summary.dropna(subset=["accuracy_mean"])
x = acc_sum["n_models"].values
y = acc_sum["accuracy_mean"].values
yerr = acc_sum["accuracy_std"].values
ymin = acc_sum["accuracy_min"].values
ymax = acc_sum["accuracy_max"].values

plt.figure()
plt.plot(x, y, marker="o", label="Povprečje")
plt.fill_between(x, ymin, ymax, alpha=0.2, label="Min–Max")
plt.errorbar(x, y, yerr=yerr, fmt="none", capsize=4)  # ±1 SD
plt.xlabel("Število modelov v ansamblu")
plt.ylabel("Točnost")
plt.title("Točnost: povprečje ±SD in min–max po kombinacijah")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(V4_PATH, "accuracy_mean_std_minmax.png"))
plt.close()

# --- AUC: povprečje + SD + min–max pas (izpusti skupine, kjer je AUC NaN) ---
auc_sum = summary.dropna(subset=["auc_mean"])
if len(auc_sum) > 0:
    x = auc_sum["n_models"].values
    y = auc_sum["auc_mean"].values
    yerr = auc_sum["auc_std"].values
    ymin = auc_sum["auc_min"].values
    ymax = auc_sum["auc_max"].values

    plt.figure()
    plt.plot(x, y, marker="o", label="Povprečje")
    plt.fill_between(x, ymin, ymax, alpha=0.2, label="Min–Max")
    plt.errorbar(x, y, yerr=yerr, fmt="none", capsize=4)  # ±1 SD
    plt.xlabel("Število modelov v ansamblu")
    plt.ylabel("AUC")
    plt.title("AUC: povprečje ±SD in min–max po kombinacijah")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(V4_PATH, "auc_mean_std_minmax.png"))
    plt.close()
# ===============================================================================


df.to_csv(os.path.join(V4_PATH, "sensitivity_results.csv"), index=False)

# ================== GRAFI ==================

plt.figure()
df.groupby("n_models")["accuracy"].mean().plot(marker='o')
plt.xlabel("Število modelov v ansamblu")
plt.ylabel("Povprečna točnost")
plt.title("Točnost glede na število modelov")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(V4_PATH, "accuracy_vs_n_models.png"))
plt.close()

plt.figure()
df.groupby("n_models")["auc"].mean().plot(marker='o')
plt.xlabel("Število modelov v ansamblu")
plt.ylabel("Povprečen AUC")
plt.title("AUC glede na število modelov")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(V4_PATH, "auc_vs_n_models.png"))
plt.close()

print("\n✅ Analiza končana. Rezultati in grafi shranjeni v:", V4_PATH)
