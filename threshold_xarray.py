# Rewritten Program to use xarray instead of pandas for thresholding

import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir))  # All/
sys.path.insert(0, parent_dir)



import torch
from utils.data.datasets import ADNIDataset


torch.serialization.add_safe_globals({
    "utils.data.datasets.ADNIDataset": ADNIDataset
})



import xarray as xr
import torch
import numpy as np
import os
import glob
import tomli as toml
from tqdm import tqdm
from utils import metrics as met


import matplotlib.pyplot as plt
import matplotlib.ticker as mtick



# The datastructures for this file are as follows
# models_dict: Dictionary - {model_id: model}
# predictions: DataArray - (data_id, model_id, prediction_value) - Prediction value has coords ['negative_prediction', 'positive_prediction', 'negative_actual', 'positive_actual']
# ensemble_statistics: DataArray - (data_id, statistic) - Statistic has coords ['mean', 'stdev', 'entropy', 'confidence', 'correct', 'predicted', 'actual']
# thresholded_predictions: DataArray - (quantile, statistic, metric) - Metric has coords ['accuracy, 'f1'] - only use 'stdev', 'entropy', 'confidence' for statistic

# Additionally, we also have the thresholds and statistics for the individual models
# indv_statistics: DataArray - (data_id, model_id, statistic) - Statistic has coords ['mean', 'entropy', 'confidence', 'correct', 'predicted', 'actual'] - No stdev as it cannot be calculated for a single model
# indv_thresholds: DataArray - (model_id, quantile, statistic, metric) - Metric has coords ['accuracy', 'f1'] - only use 'entropy', 'confidence' for statistic

# Additionally, we have some for the sensitivity analysis for number of models
# sensitivity_statistics: DataArray - (data_id, model_count, statistic) - Statistic has coords ['accuracy', 'f1', 'ECE', 'MCE']


# Loads configuration dictionary
def load_config():
    if os.getenv('ADL_CONFIG_PATH') is None:
        with open('config.toml', 'rb') as f:
            config = toml.load(f)
    else:
        with open(os.getenv('ADL_CONFIG_PATH'), 'rb') as f:
            config = toml.load(f)

    return config


# Loads models into a dictionary
def load_models_v2(folder, device):
    glob_path = os.path.join(folder, '*.pt')
    model_files = glob.glob(glob_path)
    model_dict = {}

    for model_file in model_files:
        model = torch.load(model_file, map_location=device, weights_only=False)
        model_id = os.path.basename(model_file).split('_')[0]
        model_dict[model_id] = model

    if len(model_dict) == 0:
        raise FileNotFoundError('No models found in the specified directory: ' + folder)

    return model_dict


# Ensures that both mri and xls tensors in the data are unsqueezed and are on the correct device
def preprocess_data(data, device):
    mri, xls = data
    mri = mri.unsqueeze(0).to(device)
    xls = xls.unsqueeze(0).to(device)
    return (mri, xls)


# Loads datasets and returns concatenated test and validation datasets
def load_datasets(ensemble_path):
    return (
        torch.load(f'{ensemble_path}/test_dataset.pt', weights_only=False),
        torch.load(f'{ensemble_path}/val_dataset.pt', weights_only=False),
    )



# Gets the predictions for a set of models on a dataset
def get_ensemble_predictions(models, dataset, device, id_offset=0):
    zeros = np.zeros((len(dataset), len(models), 4))
    predictions = xr.DataArray(
        zeros,
        dims=('data_id', 'model_id', 'prediction_value'),
        coords={
            'data_id': range(id_offset, len(dataset) + id_offset),
            'model_id': list(models.keys()),
            'prediction_value': [
                'negative_prediction',
                'positive_prediction',
                'negative_actual',
                'positive_actual',
            ],
        },
    )

    for data_id, (data, target) in tqdm(
        enumerate(dataset), total=len(dataset), unit='images'
    ):
        dat = preprocess_data(data, device)
        actual = list(target.cpu().numpy())
        for model_id, model in models.items():
            with torch.no_grad():
                output = model(dat)
                prediction = output.cpu().numpy().tolist()[0]

                predictions.loc[
                    {'data_id': data_id + id_offset, 'model_id': model_id}
                ] = prediction + actual

    return predictions


def _normalize_dims_coords(da: xr.DataArray) -> xr.DataArray:
    # 1) sample -> data_id (ali prva dimenzija)
    if 'data_id' not in da.dims:
        if 'sample' in da.dims:
            da = da.rename({'sample': 'data_id'})
        else:
            da = da.rename({da.dims[0]: 'data_id'})
    # 2) model -> model_id (če obstaja)
    if 'model_id' not in da.dims:
        if 'model' in da.dims:
            da = da.rename({'model': 'model_id'})
    # 3) koordinati, če manjkajo
    if 'data_id' not in da.coords:
        da = da.assign_coords(data_id=np.arange(da.sizes['data_id']))
    if 'model_id' in da.dims and 'model_id' not in da.coords:
        da = da.assign_coords(model_id=np.arange(da.sizes['model_id']))
    return da



# Compute the ensemble statistics given an array of predictions
def compute_ensemble_statistics(predictions: xr.DataArray):
    # --- poravnaj imena dimenzij/koordinat ---
    predictions = _normalize_dims_coords(predictions)

    # Ali imamo 3D z 'prediction_value'?
    has_pv = ('prediction_value' in predictions.dims)

    n = predictions.sizes['data_id']
    zeros = np.zeros((n, 7))
    ensemble_statistics = xr.DataArray(
        zeros,
        dims=('data_id', 'statistic'),
        coords={
            'data_id': predictions.data_id,
            'statistic': [
                'mean',
                'stdev',
                'entropy',
                'confidence',
                'correct',
                'predicted',
                'actual',
            ],
        },
    )

    # helper za izračun mean_neg, mean_pos, actual
    def _calc_row_stats(data: xr.DataArray):
        if has_pv:
            # originalna pot: 3D (data_id, model_id, prediction_value)
            # povpreči čez modele -> dobiš vektor po prediction_value
            mean_vec = data.mean(dim='model_id')
            # pričakujemo red: ['negative_prediction','positive_prediction', ...]
            # vzemi prvi dve (neg, pos)
            mean_neg = float(mean_vec.loc[{'prediction_value': 'negative_prediction'}].values)
            mean_pos = float(mean_vec.loc[{'prediction_value': 'positive_prediction'}].values)
            # actual
            actual = int(data.loc[{'prediction_value': 'positive_actual'}].values[0])
        else:
            # 2D pot: (data_id, model_id) z verjetnostjo pozitivnega razreda
            mean_pos = float(data.mean(dim='model_id').values)
            mean_neg = 1.0 - mean_pos
            actual = np.nan  # nimamo labela v 2D datotekah

        # stdev (po modelih) za pozitivni razred
        if has_pv:
            stdev = float(
                data.loc[{'prediction_value': 'positive_prediction'}].std(dim='model_id').values
            )
        else:
            stdev = float(data.std(dim='model_id').values)

        # entropija iz (Pneg, Ppos)
        probs = np.array([mean_neg, mean_pos], dtype=float)
        # zaščita pred log(0)
        probs = np.clip(probs, 1e-12, 1 - 1e-12)
        entropy = float((-probs * np.log(probs)).sum())

        confidence = float(max(mean_neg, mean_pos))
        predicted = int(np.argmax([mean_neg, mean_pos]))
        correct = float(np.nan) if np.isnan(actual) else float(actual == predicted)

        return mean_pos, stdev, entropy, confidence, correct, predicted, actual

    for did in predictions.data_id.values:
        if has_pv:
            data = predictions.loc[{'data_id': did}]
        else:
            data = predictions.loc[{'data_id': did}]
        mean_pos, stdev, entropy, confidence, correct, predicted, actual = _calc_row_stats(data)

        ensemble_statistics.loc[{'data_id': did}] = [
            mean_pos,
            stdev,
            entropy,
            confidence,
            correct,
            predicted,
            actual,
        ]

    return ensemble_statistics




# Compute the thresholded predictions given an array of predictions
def compute_thresholded_predictions(input_stats: xr.DataArray):
    quantiles = np.linspace(0.00, 1.00, 21) * 100
    metrics = ['accuracy', 'f1']
    statistics = ['stdev', 'entropy', 'confidence']

    zeros = np.zeros((len(quantiles), len(statistics), len(metrics)))

    thresholded_predictions = xr.DataArray(
        zeros,
        dims=('quantile', 'statistic', 'metric'),
        coords={'quantile': quantiles, 'statistic': statistics, 'metric': metrics},
    )

    for statistic in statistics:
        # First, we must compute the quantiles for the statistic
        quantile_values = np.percentile(
            input_stats.sel(statistic=statistic).values, quantiles, axis=0
        )

        # Then, we must compute the metrics for each quantile
        for i, quantile in enumerate(quantiles):
            if low_to_high(statistic):
                mask = (
                    input_stats.sel(statistic=statistic) >= quantile_values[i]
                ).values
            else:
                mask = (
                    input_stats.sel(statistic=statistic) <= quantile_values[i]
                ).values

            # Filter the data based on the mask
            filtered_data = input_stats.where(
                input_stats.data_id.isin(np.where(mask)), drop=True
            )

            for metric in metrics:
                thresholded_predictions.loc[
                    {'quantile': quantile, 'statistic': statistic, 'metric': metric}
                ] = compute_metric(filtered_data, metric)

    return thresholded_predictions


# Truth function to determine if metric should be thresholded low to high or high to low
# Low confidence is bad, high entropy is bad, high stdev is bad
# So we threshold confidence low to high, entropy and stdev high to low
# So any values BELOW the cutoff are removed for confidence, and any values ABOVE the cutoff are removed for entropy and stdev
def low_to_high(stat):
    return stat in ['confidence']


# Compute a given metric on a DataArray of statstics
def compute_metric(arr, metric):
    if metric == 'accuracy':
        return np.mean(arr.loc[{'statistic': 'correct'}])
    elif metric == 'f1':
        return met.F1(
            arr.loc[{'statistic': 'predicted'}], arr.loc[{'statistic': 'actual'}]
        )
    elif metric == 'ece':
        true_labels = arr.loc[{'statistic': 'actual'}].values
        predicted_labels = arr.loc[{'statistic': 'predicted'}].values
        confidences = arr.loc[{'statistic': 'confidence'}].values

        return calculate_ece_stats(confidences, predicted_labels, true_labels)

    else:
        raise ValueError('Invalid metric: ' + metric)


# Graph a thresholded prediction for a given statistic and metric
def graph_thresholded_prediction(
    thresholded_predictions, statistic, metric, save_path, title, xlabel, ylabel
):
    data = thresholded_predictions.sel(statistic=statistic, metric=metric)

    x_data = data.coords['quantile'].values
    y_data = data.values

    fig, ax = plt.subplots()
    ax.plot(x_data, y_data, 'bx-', label='Ensemble')
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.xaxis.set_major_formatter(mtick.PercentFormatter())

    if not low_to_high(statistic):
        ax.invert_xaxis()

    plt.savefig(save_path)


# Graph all thresholded predictions
def graph_all_thresholded_predictions(thresholded_predictions, save_path):
    # Confidence Accuracy
    graph_thresholded_prediction(
        thresholded_predictions,
        'confidence',
        'accuracy',
        f'{save_path}/confidence_accuracy.png',
        'Coverage Analysis of Confidence vs. Accuracy',
        'Minimum Confidence Percentile Threshold',
        'Accuracy',
    )

    # Confidence F1
    graph_thresholded_prediction(
        thresholded_predictions,
        'confidence',
        'f1',
        f'{save_path}/confidence_f1.png',
        'Coverage Analysis of Confidence vs. F1 Score',
        'Minimum Confidence Percentile Threshold',
        'F1 Score',
    )

    # Entropy Accuracy
    graph_thresholded_prediction(
        thresholded_predictions,
        'entropy',
        'accuracy',
        f'{save_path}/entropy_accuracy.png',
        'Coverage Analysis of Entropy vs. Accuracy',
        'Maximum Entropy Percentile Threshold',
        'Accuracy',
    )

    # Entropy F1

    graph_thresholded_prediction(
        thresholded_predictions,
        'entropy',
        'f1',
        f'{save_path}/entropy_f1.png',
        'Coverage Analysis of Entropy vs. F1 Score',
        'Maximum Entropy Percentile Threshold',
        'F1 Score',
    )

    # Stdev Accuracy
    graph_thresholded_prediction(
        thresholded_predictions,
        'stdev',
        'accuracy',
        f'{save_path}/stdev_accuracy.png',
        'Coverage Analysis of Standard Deviation vs. Accuracy',
        'Maximum Standard Deviation Percentile Threshold',
        'Accuracy',
    )

    # Stdev F1
    graph_thresholded_prediction(
        thresholded_predictions,
        'stdev',
        'f1',
        f'{save_path}/stdev_f1.png',
        'Coverage Analysis of Standard Deviation vs. F1 Score',
        'Maximum Standard Deviation Percentile Threshold',
        'F1',
    )


# Graph two statistics against each other
def graph_statistics(stats, x_stat, y_stat, save_path, title, xlabel, ylabel):
    # Filter for correct predictions
    c_stats = stats.where(
        stats.data_id.isin(np.where((stats.sel(statistic='correct') == 1).values)),
        drop=True,
    )

    # Filter for incorrect predictions
    i_stats = stats.where(
        stats.data_id.isin(np.where((stats.sel(statistic='correct') == 0).values)),
        drop=True,
    )

    # x and y data for correct and incorrect predictions
    x_data_c = c_stats.sel(statistic=x_stat).values
    y_data_c = c_stats.sel(statistic=y_stat).values

    x_data_i = i_stats.sel(statistic=x_stat).values
    y_data_i = i_stats.sel(statistic=y_stat).values

    fig, ax = plt.subplots()
    ax.plot(x_data_c, y_data_c, 'go', label='Correct')
    ax.plot(x_data_i, y_data_i, 'ro', label='Incorrect')
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()

    plt.savefig(save_path)


# Prune the data based on excluded data_ids
def prune_data(data, excluded_data_ids):
    return data.where(~data.data_id.isin(excluded_data_ids), drop=True)

def _normalize_predictions_dims(da: xr.DataArray) -> xr.DataArray:
    """
    Poskrbi, da ima DataArray pričakovani imeni dimenzij in koordinate:
    - prva dimenzija: 'data_id' (preimenuje iz 'sample' ali prve neimenovane)
    - model dimenzija: 'model_id' (preimenuje iz 'model' po potrebi)
    - doda koordinato 'data_id', če je ni
    """
    # --- sample -> data_id ---
    if 'data_id' not in da.dims:
        if 'sample' in da.dims:
            da = da.rename({'sample': 'data_id'})
        else:
            # fallback: vzemi prvo dimenzijo kot data_id
            da = da.rename({da.dims[0]: 'data_id'})

    # --- model -> model_id ---
    if 'model_id' not in da.dims:
        if 'model' in da.dims:
            da = da.rename({'model': 'model_id'})
        # sicer pustimo obstoječe ime, če je že 'model_id' ali nekaj tretjega

    # --- zagotovimo koordinato data_id ---
    if 'data_id' not in da.coords:
        da = da.assign_coords(data_id=np.arange(da.sizes['data_id']))

    # (opcijsko: če nima koordinate model_id, lahko ustvarimo zaporedje)
    if 'model_id' in da.dims and 'model_id' not in da.coords:
        da = da.assign_coords(model_id=np.arange(da.sizes['model_id']))

    return da



# Calculate individual model statistics
def compute_individual_statistics(predictions: xr.DataArray):
    zeros = np.zeros((len(predictions.data_id), len(predictions.model_id), 6))

    indv_statistics = xr.DataArray(
        zeros,
        dims=('data_id', 'model_id', 'statistic'),
        coords={
            'data_id': predictions.data_id,
            'model_id': predictions.model_id,
            'statistic': [
                'mean',
                'entropy',
                'confidence',
                'correct',
                'predicted',
                'actual',
            ],
        },
    )

    for data_id in tqdm(
        predictions.data_id, total=len(predictions.data_id), unit='images'
    ):
        for model_id in predictions.model_id:
            data = predictions.loc[{'data_id': data_id, 'model_id': model_id}]
            mean = data[0:2]
            entropy = (-mean * np.log(mean)).sum()
            confidence = mean.max()
            actual = data[3]
            predicted = mean.argmax()
            correct = actual == predicted

            indv_statistics.loc[{'data_id': data_id, 'model_id': model_id}] = [
                mean[1],
                entropy,
                confidence,
                correct,
                predicted,
                actual,
            ]

    return indv_statistics


# Compute individual model thresholds
def compute_individual_thresholds(input_stats: xr.DataArray):
    quantiles = np.linspace(0.05, 0.95, 19) * 100
    metrics = ['accuracy', 'f1']
    statistics = ['entropy', 'confidence']

    zeros = np.zeros(
        (len(input_stats.model_id), len(quantiles), len(statistics), len(metrics))
    )

    indv_thresholds = xr.DataArray(
        zeros,
        dims=('model_id', 'quantile', 'statistic', 'metric'),
        coords={
            'model_id': input_stats.model_id,
            'quantile': quantiles,
            'statistic': statistics,
            'metric': metrics,
        },
    )

    for model_id in tqdm(
        input_stats.model_id, total=len(input_stats.model_id), unit='models'
    ):
        for statistic in statistics:
            # First, we must compute the quantiles for the statistic
            quantile_values = np.percentile(
                input_stats.sel(model_id=model_id, statistic=statistic).values,
                quantiles,
                axis=0,
            )

            # Then, we must compute the metrics for each quantile
            for i, quantile in enumerate(quantiles):
                if low_to_high(statistic):
                    mask = (
                        input_stats.sel(model_id=model_id, statistic=statistic)
                        >= quantile_values[i]
                    ).values
                else:
                    mask = (
                        input_stats.sel(model_id=model_id, statistic=statistic)
                        <= quantile_values[i]
                    ).values

                # Filter the data based on the mask
                filtered_data = input_stats.where(
                    input_stats.data_id.isin(np.where(mask)), drop=True
                )

                for metric in metrics:
                    indv_thresholds.loc[
                        {
                            'model_id': model_id,
                            'quantile': quantile,
                            'statistic': statistic,
                            'metric': metric,
                        }
                    ] = compute_metric(filtered_data, metric)

    return indv_thresholds


# Graph individual model thresholded predictions
def graph_individual_thresholded_predictions(
    indv_thresholds,
    ensemble_thresholds,
    statistic,
    metric,
    save_path,
    title,
    xlabel,
    ylabel,
):
    data = indv_thresholds.sel(statistic=statistic, metric=metric)
    e_data = ensemble_thresholds.sel(statistic=statistic, metric=metric)

    x_data = data.coords['quantile'].values
    y_data = data.values

    e_x_data = e_data.coords['quantile'].values
    e_y_data = e_data.values

    fig, ax = plt.subplots()
    for model_id in data.coords['model_id'].values:
        model_data = data.sel(model_id=model_id)
        ax.plot(x_data, model_data)

    ax.plot(e_x_data, e_y_data, 'kx-', label='Ensemble')

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.xaxis.set_major_formatter(mtick.PercentFormatter())

    if not low_to_high(statistic):
        ax.invert_xaxis()

    ax.legend()
    plt.savefig(save_path)


# Graph all individual thresholded predictions
def graph_all_individual_thresholded_predictions(
    indv_thresholds, ensemble_thresholds, save_path
):
    # Confidence Accuracy
    graph_individual_thresholded_predictions(
        indv_thresholds,
        ensemble_thresholds,
        'confidence',
        'accuracy',
        f'{save_path}/indv/confidence_accuracy.png',
        'Coverage Analysis of Confidence vs. Accuracy for All Models',
        'Minumum Confidence Percentile Threshold',
        'Accuracy',
    )

    # Confidence F1
    graph_individual_thresholded_predictions(
        indv_thresholds,
        ensemble_thresholds,
        'confidence',
        'f1',
        f'{save_path}/indv/confidence_f1.png',
        'Coverage Analysis of Confidence vs. F1 Score for All Models',
        'Minimum Confidence Percentile Threshold',
        'F1 Score',
    )

    # Entropy Accuracy
    graph_individual_thresholded_predictions(
        indv_thresholds,
        ensemble_thresholds,
        'entropy',
        'accuracy',
        f'{save_path}/indv/entropy_accuracy.png',
        'Coverage Analysis of Entropy vs. Accuracy for All Models',
        'Maximum Entropy Percentile Threshold',
        'Accuracy',
    )

    # Entropy F1
    graph_individual_thresholded_predictions(
        indv_thresholds,
        ensemble_thresholds,
        'entropy',
        'f1',
        f'{save_path}/indv/entropy_f1.png',
        'Coverage Analysis of Entropy vs. F1 Score for All Models',
        'Maximum Entropy Percentile Threshold',
        'F1 Score',
    )


# Calculate statistics of subsets of models for sensitivity analysis
def calculate_subset_statistics(predictions: xr.DataArray):
    # Calculate subsets for 1-49 models
    subsets = range(1, len(predictions.model_id))

    zeros = np.zeros(
        (len(predictions.data_id), len(subsets), 7)
    )  # Include stdev, but for 1 models set to NaN

    subset_stats = xr.DataArray(
        zeros,
        dims=('data_id', 'model_count', 'statistic'),
        coords={
            'data_id': predictions.data_id,
            'model_count': subsets,
            'statistic': [
                'mean',
                'stdev',
                'entropy',
                'confidence',
                'correct',
                'predicted',
                'actual',
            ],
        },
    )

    for data_id in tqdm(
        predictions.data_id, total=len(predictions.data_id), unit='images'
    ):
        for subset in subsets:
            data = predictions.sel(
                data_id=data_id, model_id=predictions.model_id[:subset]
            )
            mean = data.mean(dim='model_id')[0:2]
            stdev = data.std(dim='model_id')[1]
            entropy = (-mean * np.log(mean)).sum()
            confidence = mean.max()
            actual = data[0][3]
            predicted = mean.argmax()
            correct = actual == predicted

            subset_stats.loc[{'data_id': data_id, 'model_count': subset}] = [
                mean[1],
                stdev,
                entropy,
                confidence,
                correct,
                predicted,
                actual,
            ]

    return subset_stats


# Calculate Accuracy, F1 and ECE for subset stats - sensityvity analysis
def calculate_sensitivity_analysis(subset_stats: xr.DataArray):
    subsets = subset_stats.model_count
    stats = ['accuracy', 'f1', 'ece']

    zeros = np.zeros((len(subsets), len(stats)))

    sens_analysis = xr.DataArray(
        zeros,
        dims=('model_count', 'statistic'),
        coords={'model_count': subsets, 'statistic': stats},
    )

    for subset in tqdm(subsets, total=len(subsets), unit='model subsets'):

        data = subset_stats.sel(model_count=subset)
        acc = compute_metric(data, 'accuracy').item()
        f1 = compute_metric(data, 'f1').item()
        ece = compute_metric(data, 'ece').item()

        sens_analysis.loc[{'model_count': subset.item()}] = [acc, f1, ece]

    return sens_analysis


def graph_sensitivity_analysis(
    sens_analysis: xr.DataArray, statistic, save_path, title, xlabel, ylabel
):
    data = sens_analysis.sel(statistic=statistic)

    xdata = data.coords['model_count'].values
    ydata = data.values

    fig, ax = plt.subplots()
    ax.plot(xdata, ydata)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    plt.savefig(save_path)


def calculate_overall_stats(ensemble_statistics: xr.DataArray):
    accuracy = compute_metric(ensemble_statistics, 'accuracy')
    f1 = compute_metric(ensemble_statistics, 'f1')

    return {'accuracy': accuracy.item(), 'f1': f1.item()}


# https://towardsdatascience.com/expected-calibration-error-ece-a-step-by-step-visual-explanation-with-python-code-c3e9aa12937d
def calculate_ece_stats(confidences, predicted_labels, true_labels, bins=10):
    bin_boundaries = np.linspace(0, 1, bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = np.zeros(1)

    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = np.logical_and(
            confidences > bin_lower.item(), confidences <= bin_upper.item()
        )
        prob_in_bin = in_bin.mean()

        if prob_in_bin.item() > 0:
            accuracy_in_bin = true_labels[in_bin].mean()

            avg_confidence_in_bin = confidences[in_bin].mean()

            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prob_in_bin

    return ece


def plot_ece_graph(ece_stats, title, xlabel, ylabel, save_path):
    fix, ax = plt.subplot()


# Main Function
def main():
    print('Loading Config...')
    config = load_config()
    ENSEMBLE_PATH = os.path.join(config['paths']['model_output'], config['ensemble']['name'])
    V4_PATH = ENSEMBLE_PATH + '/v4'

    if not os.path.exists(V4_PATH):
        os.makedirs(V4_PATH)
    print('Config Loaded')

    # Load Datasets
    print('Loading Datasets...')
    (test_dataset, val_dataset) = load_datasets(ENSEMBLE_PATH)
    print('Datasets Loaded')

    # Get Predictions, either by running the models or loading them from a file
    if config['ensemble']['run_models']:
        # Load Models
        print('Loading Models...')
        device = torch.device(config['training']['device'])
        models = load_models_v2(f'{ENSEMBLE_PATH}/models/', device)
        print('Models Loaded')

        # Get Predictions
        print('Getting Predictions...')
        test_predictions = get_ensemble_predictions(models, test_dataset, device)
        val_predictions = get_ensemble_predictions(
            models, val_dataset, device, len(test_dataset)
        )
        print('Predictions Loaded')

        # Save Prediction
        test_predictions.to_netcdf(f'{V4_PATH}/test_predictions.nc')
        val_predictions.to_netcdf(f'{V4_PATH}/val_predictions.nc')

    else:
        test_predictions = xr.open_dataarray(f'{V4_PATH}/test_predictions.nc')
        val_predictions = xr.open_dataarray(f'{V4_PATH}/val_predictions.nc')

        # --- NORMALIZACIJA DIMENZIJ IN KOORDINAT ---

        test_predictions = _normalize_predictions_dims(test_predictions)
        val_predictions = _normalize_predictions_dims(val_predictions)

# Prune Data
    print('Pruning Data...')
    if config['operation']['exclude_blank_ids']:
        excluded_data_ids = config['ensemble']['excluded_ids']
        test_predictions = prune_data(test_predictions, excluded_data_ids)
        val_predictions = prune_data(val_predictions, excluded_data_ids)

    # Concatenate Predictions
    predictions = xr.concat([test_predictions, val_predictions], dim='data_id')

    # Compute Ensemble Statistics
    print('Computing Ensemble Statistics...')
    ensemble_statistics = compute_ensemble_statistics(predictions)
    ensemble_statistics.to_netcdf(f'{V4_PATH}/ensemble_statistics.nc')
    print('Ensemble Statistics Computed')

    # Compute Thresholded Predictions
    print('Computing Thresholded Predictions...')
    thresholded_predictions = compute_thresholded_predictions(ensemble_statistics)
    thresholded_predictions.to_netcdf(f'{V4_PATH}/thresholded_predictions.nc')
    print('Thresholded Predictions Computed')

    # Graph Thresholded Predictions
    print('Graphing Thresholded Predictions...')
    graph_all_thresholded_predictions(thresholded_predictions, V4_PATH)
    print('Thresholded Predictions Graphed')

    # Additional Graphs
    print('Graphing Additional Graphs...')
    # Confidence vs stdev
    graph_statistics(
        ensemble_statistics,
        'confidence',
        'stdev',
        f'{V4_PATH}/confidence_stdev.png',
        'Confidence and Standard Deviation for Predictions',
        'Confidence',
        'Standard Deviation',
    )
    print('Additional Graphs Graphed')

    # Compute Individual Statistics
    print('Computing Individual Statistics...')
    indv_statistics = compute_individual_statistics(predictions)
    indv_statistics.to_netcdf(f'{V4_PATH}/indv_statistics.nc')
    print('Individual Statistics Computed')

    # Compute Individual Thresholds
    print('Computing Individual Thresholds...')
    indv_thresholds = compute_individual_thresholds(indv_statistics)
    indv_thresholds.to_netcdf(f'{V4_PATH}/indv_thresholds.nc')
    print('Individual Thresholds Computed')

    # Graph Individual Thresholded Predictions
    print('Graphing Individual Thresholded Predictions...')
    if not os.path.exists(f'{V4_PATH}/indv'):
        os.makedirs(f'{V4_PATH}/indv')

    graph_all_individual_thresholded_predictions(
        indv_thresholds, thresholded_predictions, V4_PATH
    )
    print('Individual Thresholded Predictions Graphed')

    # Compute subset statistics and graph
    print('Computing Sensitivity Analysis...')
    subset_stats = calculate_subset_statistics(predictions)
    sens_analysis = calculate_sensitivity_analysis(subset_stats)
    graph_sensitivity_analysis(
        sens_analysis,
        'accuracy',
        f'{V4_PATH}/sens_analysis.png',
        'Sensitivity Analsis of Accuracy vs. # of Models',
        '# of Models',
        'Accuracy',
    )
    graph_sensitivity_analysis(
        sens_analysis,
        'ece',
        f'{V4_PATH}/sens_analysis_ece.png',
        'Sensitivity Analysis of ECE vs. # of Models',
        '# of Models',
        'ECE',
    )
    print(sens_analysis.sel(statistic='accuracy'))
    print(calculate_overall_stats(ensemble_statistics))


if __name__ == '__main__':
    main()
