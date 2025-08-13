import utils.ensemble as ens
import os
import tomli as toml
import math
import torch
import numpy as np
import xarray as xr
from utils.system import force_init_cudnn

# CONFIGURATION
if os.getenv('ADL_CONFIG_PATH') is None:
    with open('config.toml', 'rb') as f:
        config = toml.load(f)
else:
    with open(os.getenv('ADL_CONFIG_PATH'), 'rb') as f:
        config = toml.load(f)

# Force cuDNN initialization
force_init_cudnn(config['training']['device'])

# Paths
model_base = os.path.join(config['paths']['model_output'], config['ensemble']['name'])
models_dir = os.path.join(model_base, 'models')
results_file = os.path.join(model_base, f"ensemble_test_results_{config['ensemble']['prune_threshold']}.txt")
xarray_dir = os.path.join(model_base, "v4")
os.makedirs(xarray_dir, exist_ok=True)

from torch.serialization import add_safe_globals
from utils.models.cnn import CNN, CNN_Image_Section
from utils.models.layers import ConvBlock
from torch.nn import (
    Conv3d, BatchNorm3d, ELU, ReLU, LeakyReLU, Dropout, Dropout3d,
    Linear, MaxPool3d, AdaptiveAvgPool3d, Sigmoid, Sequential, Module
)

# Dovoli vse uporabljene razrede v tvojem CNN modelu
add_safe_globals([
    CNN, CNN_Image_Section, ConvBlock,
    Conv3d, BatchNorm3d, ELU, ReLU, LeakyReLU,
    Dropout, Dropout3d, Linear, MaxPool3d, AdaptiveAvgPool3d,
    Sigmoid, Sequential, Module
])


# Load models
models, model_descs = ens.load_models(models_dir, config['training']['device'])
models, model_descs = ens.prune_models(
    models, model_descs, models_dir, config['ensemble']['prune_threshold']
)

# Load test & val datasets
test_dataset_path = os.path.join(model_base, 'test_dataset.pt')
val_dataset_path = os.path.join(model_base, 'val_dataset.pt')

from utils.data.datasets import ADNIDataset
add_safe_globals([ADNIDataset])

test_dataset = torch.load(test_dataset_path, weights_only=False)
val_dataset = torch.load(val_dataset_path, weights_only=False)


def evaluate_dataset(dataset, dataset_name):
    correct = 0
    total = 0
    predictions = []
    actual = []
    stdevs = []
    all_yes_votes = []
    all_no_votes = []
    all_model_outputs = []

    for data, target in dataset:
        mri, xls = data
        mri = mri.unsqueeze(0)
        xls = xls.unsqueeze(0)
        data = (mri, xls)

        outputs = []
        for model in models:
            with torch.no_grad():
                pred = model(data)[0][0].item()
                outputs.append(pred)

        outputs = np.array(outputs)
        mean = np.mean(outputs)
        variance = np.var(outputs)
        yes_votes = (outputs > 0.5).astype(int).tolist()
        no_votes = (outputs <= 0.5).astype(int).tolist()

        stdevs.append(np.sqrt(variance))
        predicted = round(mean)
        expected = target[1].item()

        total += 1
        correct += int(predicted == expected)

        predictions.append(mean)
        actual.append(expected)
        all_yes_votes.append(yes_votes)
        all_no_votes.append(no_votes)
        all_model_outputs.append(outputs)

    accuracy = correct / total

    if dataset_name == "test":
        with open(results_file, 'w') as f:
            f.write('Accuracy: ' + str(accuracy) + '\n')
            f.write('Correct: ' + str(correct) + '\n')
            f.write('Total: ' + str(total) + '\n')
            f.write('\n')
            for exp, pred, stdev, yv, nv in zip(actual, predictions, stdevs, all_yes_votes, all_no_votes):
                f.write(
                    f"{exp}, [{pred:.4f}], {stdev:.4f}, {yv}, {nv}\n"
                )

        np.savez(
            os.path.join(model_base, "ensemble_preds_test.npz"),
            predictions=np.array(predictions),
            actual=np.array(actual),
            stdev=np.array(stdevs),
            yes_votes=np.array(all_yes_votes),
            no_votes=np.array(all_no_votes),
            all_model_outputs=np.array(all_model_outputs),
        )

    # Save xarray
    # Pripravi podatke v obliki (samples, models, 4)
    # 4 elementi: [neg_pred, pos_pred, neg_actual, pos_actual]
    all_outputs = []

    for outputs, label in zip(all_model_outputs, actual):
        sample_outputs = []
        for pred in outputs:
            neg_pred = 1 - pred
            pos_pred = pred
            neg_actual = 1 - label
            pos_actual = label
            sample_outputs.append([neg_pred, pos_pred, neg_actual, pos_actual])
        all_outputs.append(sample_outputs)

    # Pretvori v numpy array: shape (samples, models, 4)
    all_outputs_np = np.array(all_outputs)

    xr_data = xr.DataArray(
        data=all_outputs_np,
        dims=["data_id", "model_id", "prediction_value"],
        coords={
            "data_id": np.arange(len(all_outputs_np)),
            "model_id": np.arange(len(models)),
            "prediction_value": [
                "negative_prediction",
                "positive_prediction",
                "negative_actual",
                "positive_actual",
            ],
        },
        name=f"{dataset_name}_predictions"
    )
    xr_data.to_netcdf(os.path.join(xarray_dir, f"{dataset_name}_predictions.nc"))
    xr_data.name = f"{dataset_name}_predictions"
    xr_data.to_netcdf(os.path.join(xarray_dir, f"{dataset_name}_predictions.nc"))

    print(f"→ {dataset_name.capitalize()} accuracy: {accuracy:.4f} (saved)")

evaluate_dataset(test_dataset, "test")
evaluate_dataset(val_dataset, "val")

print("\n✅ Ensemble evaluation complete.")
print(f"→ Results saved to: {results_file}")
print(f"→ .nc files saved to: {xarray_dir}/")

