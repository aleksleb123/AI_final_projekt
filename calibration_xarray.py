# calibration_xarray.py
import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

import threshold_xarray as th


# ----------------------------- Utils: metrike & grafi ----------------------------- #

def compute_ece(y_true, y_prob, n_bins=15):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    assert y_true.shape[0] == y_prob.shape[0], "y_true in y_prob morata imeti enako dolžino."

    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    inds = np.digitize(y_prob, bin_boundaries[1:-1], right=False)

    ece = 0.0
    for b in range(n_bins):
        in_bin = inds == b
        if not np.any(in_bin):
            continue
        acc = y_true[in_bin].mean()
        conf = y_prob[in_bin].mean()
        w = np.mean(in_bin)
        ece += w * abs(acc - conf)
    return float(ece)


def plot_reliability_diagram(y_true, y_prob, save_path, n_bins=15):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    inds = np.digitize(y_prob, bin_edges[1:-1], right=False)

    accs, confs = [], []
    for b in range(n_bins):
        mask = inds == b
        if not np.any(mask):
            accs.append(np.nan); confs.append(np.nan); continue
        accs.append(y_true[mask].mean())
        confs.append(y_prob[mask].mean())

    accs = np.array(accs, dtype=float)
    confs = np.array(confs, dtype=float)

    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], linestyle="--")
    valid = ~np.isnan(accs)
    plt.scatter(confs[valid], accs[valid], s=20)
    plt.xlabel("Povprečna samozavest (bin)")
    plt.ylabel("Povprečna točnost (bin)")
    plt.title("Reliability diagram")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_confidence_histogram(y_prob, save_path, n_bins=20):
    y_prob = np.asarray(y_prob).astype(float)
    plt.figure(figsize=(6, 4))
    plt.hist(y_prob, bins=n_bins)
    plt.xlabel("Samozavest (P[y=1])")
    plt.ylabel("Število vzorcev")
    plt.title("Histogram samozavesti")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_coverage_curve(y_prob, y_true, save_path, steps=20):
    y_prob = np.asarray(y_prob).astype(float)
    y_true = np.asarray(y_true).astype(int)

    deltas = np.linspace(0.0, 0.5, steps)
    coverages, accuracies = [], []
    for d in deltas:
        mask = np.abs(y_prob - 0.5) >= d
        coverage = np.mean(mask) if mask.size else np.nan
        if np.any(mask):
            acc = (y_true[mask] == (y_prob[mask] >= 0.5)).mean()
        else:
            acc = np.nan
        coverages.append(coverage); accuracies.append(acc)

    plt.figure(figsize=(6, 4))
    plt.plot(coverages, accuracies, marker="o")
    plt.xlabel("Coverage")
    plt.ylabel("Accuracy")
    plt.title("Coverage vs. Accuracy")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def _extract_labels_from_stats(stats: xr.DataArray):
    """Vrne int vektor label, če so prisotne; sicer None."""
    arr = stats.sel(statistic="actual").values
    if np.isnan(arr).any():
        return None
    return arr.astype(int)


# ----------------------------- Glavni potek: kalibracija ----------------------------- #

if __name__ == "__main__":
    print("Loading config ...")
    config = th.load_config()
    ENSEMBLE_PATH = os.path.join(config["paths"]["model_output"], config["ensemble"]["name"])
    V4_PATH = os.path.join(ENSEMBLE_PATH, "v4")
    os.makedirs(V4_PATH, exist_ok=True)
    print(f"Config loaded. Output dir: {V4_PATH}")

    # 1) Naloži napovedi, ki jih je ustvaril ensemble_predict.py
    print("Loading predictions ...")
    val_preds = xr.open_dataarray(os.path.join(V4_PATH, "val_predictions.nc"))
    test_preds = xr.open_dataarray(os.path.join(V4_PATH, "test_predictions.nc"))
    print("Predictions loaded.")

    # 2) Izračunaj statistike (compute_ensemble_statistics je patchan, podpira 3D iz ensemble_predict in 2D fallback)
    print("Calculating statistics ...")
    val_stats = th.compute_ensemble_statistics(val_preds)
    test_stats = th.compute_ensemble_statistics(test_preds)
    print("Statistics ready.")

    # 3) Izlušči samozavest in labele
    val_confidences = np.asarray(val_stats.sel(statistic="mean").values, dtype=float)
    test_confidences = np.asarray(test_stats.sel(statistic="mean").values, dtype=float)

    # Najprej poskusi prebrati labele iz stats (to deluje, če si .nc ustvaril z ensemble_predict.py)
    val_labels = _extract_labels_from_stats(val_stats)
    test_labels = _extract_labels_from_stats(test_stats)

    # Fallback: če stats nimajo labelov (npr. če imaš stare 2D .nc), poskusi naložiti .npy
    if val_labels is None or test_labels is None:
        print("Labels not embedded in .nc — trying fallback to .npy ...")
        val_labels_path = os.path.join(V4_PATH, "val_labels.npy")
        test_labels_path = os.path.join(V4_PATH, "test_labels.npy")
        if not (os.path.exists(val_labels_path) and os.path.exists(test_labels_path)):
            raise FileNotFoundError(
                "Manjkajo ground-truth labele. Zaženi najprej ensemble_predict.py (ki vgradi labele v .nc) "
                "ALI pa priskrbi val_labels.npy in test_labels.npy v V4_PATH."
            )
        val_labels = np.load(val_labels_path).astype(int)
        test_labels = np.load(test_labels_path).astype(int)

    # 4) Platt scaling (fit na validation, apply na test)
    print("Fitting Platt scaling on validation ...")
    lr = LogisticRegression(solver="lbfgs")
    lr.fit(val_confidences.reshape(-1, 1), val_labels)
    calibrated_test_probs = lr.predict_proba(test_confidences.reshape(-1, 1))[:, 1]
    print("Calibration done.")

    # 5) Shrani in poročaj ECE
    np.save(os.path.join(V4_PATH, "test_probs_calibrated.npy"), calibrated_test_probs)

    ece_before = compute_ece(test_labels, test_confidences, n_bins=15)
    ece_after = compute_ece(test_labels, calibrated_test_probs, n_bins=15)
    print(f"ECE test (before): {ece_before:.4f}")
    print(f"ECE test (after) : {ece_after:.4f}")

    # 6) Grafi
    print("Plotting figures ...")
    plot_reliability_diagram(
        test_labels, calibrated_test_probs,
        save_path=os.path.join(V4_PATH, "reliability_diagram.png"), n_bins=15
    )
    plot_confidence_histogram(
        calibrated_test_probs,
        save_path=os.path.join(V4_PATH, "confidence_histogram.png"), n_bins=20
    )
    plot_coverage_curve(
        calibrated_test_probs, test_labels,
        save_path=os.path.join(V4_PATH, "coverage_curve.png"), steps=20
    )

    print("All done. Figures saved to V4_PATH.")
