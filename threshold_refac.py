from utils.data.datasets import ADNIDataset
import torch
import torch.serialization
torch.serialization.add_safe_globals([ADNIDataset])


import pandas as pd
import numpy as np
import os
import tomli as toml
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
from tqdm import tqdm
import utils.metrics as met
import matplotlib.ticker as ticker
import glob
import pickle as pk
import warnings

warnings.filterwarnings('error')


def plot_image_grid(image_ids, dataset, rows, path, titles=None):
    # odvrzi indekse izven meja
    max_idx = len(dataset) - 1
    valid = [(idx, (titles[i] if titles else None))
             for i, idx in enumerate(image_ids) if 0 <= idx <= max_idx]
    if not valid:
        raise ValueError("No valid indices to plot.")

    cols = (len(valid) + rows - 1) // rows
    fig, axs = plt.subplots(rows, cols, figsize=(3*cols, 3*rows))
    axs = np.atleast_1d(axs).reshape(-1)

    for k, (idx, title) in enumerate(valid):
        img = dataset[idx][0][0].squeeze().cpu().numpy()
        # sredinski rez, če je 3D
        if img.ndim == 3:
            img = img[:, :, img.shape[2] // 2]
        axs[k].imshow(img, cmap='gray')
        axs[k].axis('off')
        if title:
            axs[k].set_title(title, fontsize=9)

    # skrij neuporabljene osi
    for ax in axs[len(valid):]:
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(path, bbox_inches='tight', pad_inches=0)
    plt.close()



def plot_single_image(image_id, dataset, path, title=None):
    fig, ax = plt.subplots()
    image = dataset[image_id][0][0].squeeze().cpu().numpy()
    # We now have a 3d image of size (91, 109, 91), and we want to take a slice from the middle of the image
    image = image[:, :, 45]

    ax.imshow(image, cmap='gray')
    ax.axis('off')
    if title is not None:
        ax.set_title(title)

    plt.savefig(path)
    plt.close()


# Given a dataframe of the form {data_id: (stat_1, stat_2, ..., correct)}, plot the two statistics against each other and color by correctness
def plot_statistics_versus(
    stat_1, stat_2, xaxis_name, yaxis_name, title, dataframe, path, annotate=False
):
    # Get correct predictions and incorrect predictions dataframes
    corr_df = dataframe[dataframe['correct']]
    incorr_df = dataframe[~dataframe['correct']]

    # Plot the correct and incorrect predictions
    fig, ax = plt.subplots()
    ax.scatter(corr_df[stat_1], corr_df[stat_2], c='green', label='Correct')
    ax.scatter(incorr_df[stat_1], incorr_df[stat_2], c='red', label='Incorrect')
    ax.legend()
    ax.set_xlabel(xaxis_name)
    ax.set_ylabel(yaxis_name)
    ax.set_title(title)

    if annotate:
        print('DEBUG -- REMOVE: Annotating')
        # label correct points green
        for row in dataframe[[stat_1, stat_2]].itertuples():
            plt.text(row[1], row[2], row[0], fontsize=6, color='black')

    plt.savefig(path)
    plt.close()


# Models is a dictionary with the model ids as keys and the model data as values
def get_model_predictions(models, data):
    predictions = {}
    for model_id, model in models.items():
        model.eval()
        with torch.no_grad():
            # Get the predictions
            output = model(data)
            predictions[model_id] = output.detach().cpu().numpy()

    return predictions


def load_models_v2(folder, device):
    from utils.models.cnn import CNN  # ali ustrezna pot do tvoje CNN definicije
    import torch.serialization
    torch.serialization.add_safe_globals([CNN])

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


def ensemble_dataset_predictions(models, dataset, device):
    # For each datapoint, get the predictions of each model
    predictions = {}
    for i, (data, target) in tqdm(enumerate(dataset), total=len(dataset)):
        # Preprocess data
        data = preprocess_data(data, device)
        # Predictions is a dicionary of tuples, with the target as the first and the model predicions dictionary as the second
        # The key is the id of the image
        predictions[i] = (
            target.detach().cpu().numpy(),
            get_model_predictions(models, data),
        )

    return predictions


# Given a dictionary of predictions, select one model and eliminate the rest
def select_individual_model(predictions, model_id):
    selected_model_predictions = {}
    for key, value in predictions.items():
        selected_model_predictions[key] = (
            value[0],
            {model_id: value[1][str(model_id)]},
        )
    return selected_model_predictions


# Given a dictionary of predictions, select a subset of models and eliminate the rest
# predictions dictory of the form {data_id: (target, {model_id: prediction})}
def select_subset_models(predictions, model_ids):
    selected_model_predictions = {}
    for key, value in predictions.items():
        target = value[0]
        model_predictions = value[1]

        # Filter the model predictions, only keeping selected models
        selected_model_predictions[key] = (
            target,
            {model_id: model_predictions[str(model_id + 1)] for model_id in model_ids},
        )

    return selected_model_predictions


# Given a dictionary of predictions, calculate statistics (stdev, mean, entropy, correctness) for each result
# Returns a dataframe of the form {data_id: (mean, stdev, entropy, confidence, correct, predicted, actual)}
def calculate_statistics(predictions):
    # Create DataFrame with columns for each statistic
    stats_df = pd.DataFrame(
        columns=[
            'mean',
            'stdev',
            'entropy',
            'confidence',
            'correct',
            'predicted',
            'actual',
        ]
    )

    # First, loop through each prediction
    for key, value in predictions.items():

        target = value[0]
        model_predictions = list(value[1].values())

        print(f"→ Sample {key}: {[np.round(p, 3) for p in model_predictions]}")  # ✅ zdaj bo delovalo

        # Calculate the mean and stdev of predictions
        mean = np.squeeze(np.mean(model_predictions, axis=0))
        stdev = np.squeeze(np.std(model_predictions, axis=0))[1]

        # Calculate the entropy of the predictions
        entropy = met.entropy(mean)

        # Calculate confidence
        confidence = (np.max(mean) - 0.5) * 2

        # Calculate predicted and actual
        predicted = np.argmax(mean)
        actual = np.argmax(target)

        # Determine if the prediction is correct
        correct = predicted == actual

        # Add the statistics to the dataframe
        stats_df.loc[key] = [
            mean,
            stdev,
            entropy,
            confidence,
            correct,
            predicted,
            actual,
        ]

    return stats_df


# Takes in a dataframe of the form {data_id: statistic, ...} and calculates the thresholds for the statistic
# Output of the form DataFrame(index=threshold, columns=[accuracy, f1])
def conduct_threshold_analysis(statistics, statistic_name, low_to_high=True):
    # Izračun percentilov za izbrano statistiko
    percentile_df = statistics[statistic_name].quantile(
        q=np.linspace(0.05, 0.95, num=18)
    )

    # Inicializacija DataFrame-a za rezultate
    thresholds_pd = pd.DataFrame(index=percentile_df.index, columns=['accuracy', 'f1'])

    for percentile, value in percentile_df.items():
        # Filtriraj primere glede na threshold
        if low_to_high:
            filtered_statistics = statistics[statistics[statistic_name] < value]
        else:
            filtered_statistics = statistics[statistics[statistic_name] >= value]

        if len(filtered_statistics) == 0:
            print(f"⚠️  No samples after filtering at percentile {percentile:.2f} (value: {value:.4f})")
            accuracy = 0.0
            f1 = 0.0
        else:
            # Izračunaj accuracy
            accuracy = filtered_statistics['correct'].mean()

            # Pripravi vektorje
            predicted = filtered_statistics['predicted'].values
            actual = filtered_statistics['actual'].values

            # Robustno računaj F1
            if len(np.unique(actual)) < 2 or len(np.unique(predicted)) < 2:
                f1 = 0.0
            else:
                f1 = metrics.f1_score(actual, predicted, zero_division=0)

        thresholds_pd.loc[percentile] = [accuracy, f1]

    return thresholds_pd



# Takes a dictionary of the form {threshold: {metric: value}} for a given statistic and plots the metric against the threshold.
# Can plot an additional line if given (used for individual results)
def plot_threshold_analysis(
    thresholds_metric, title, x_label, y_label, path, additional_set=None, flip=False
):
    # Initialize the plot
    fig, ax = plt.subplots()

    # Get the thresholds and metrics
    thresholds = list(thresholds_metric.index)
    metric = list(thresholds_metric.values)

    # Plot the metric against the threshold
    plt.plot(thresholds, metric, 'bo-', label='Ensemble')

    if additional_set is not None:
        # Get the thresholds and metrics
        thresholds = list(additional_set.index)
        metric = list(additional_set.values)

        # Plot the metric against the threshold
        plt.plot(thresholds, metric, 'rx-', label='Individual')

    if flip:
        ax.invert_xaxis()

    # Add labels
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    ax.xaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))

    plt.savefig(path)
    plt.close()


# Code from https://stackoverflow.com/questions/16458340
# Returns the intersections of multiple dictionaries
def common_entries(*dcts):
    if not dcts:
        return
    for i in set(dcts[0]).intersection(*dcts[1:]):
        yield (i,) + tuple(d[i] for d in dcts)


# Given ensemble statistics, calculate overall stats (ECE, MCE, Brier Score, NLL)
def calculate_overall_statistics(ensemble_statistics):
    predicted = ensemble_statistics['predicted']
    actual = ensemble_statistics['actual']

    # New dataframe to store the statistics
    stats_df = pd.DataFrame(
        columns=['stat', 'ECE', 'MCE', 'Brier Score', 'NLL']
    ).set_index('stat')

    # Loop through and calculate the ECE, MCE, Brier Score, and NLL
    for stat in ['confidence', 'entropy', 'stdev', 'raw_confidence']:
        ece = met.ECE(predicted, ensemble_statistics[stat], actual)
        mce = met.MCE(predicted, ensemble_statistics[stat], actual)
        brier = met.brier_binary(ensemble_statistics[stat], actual)
        nll = met.nll_binary(ensemble_statistics[stat], actual)

        stats_df.loc[stat] = [ece, mce, brier, nll]

    return stats_df


# CONFIGURATION
def load_config():
    if os.getenv('ADL_CONFIG_PATH') is None:
        with open('config.toml', 'rb') as f:
            config = toml.load(f)
    else:
        with open(os.getenv('ADL_CONFIG_PATH'), 'rb') as f:
            config = toml.load(f)

    return config

def prune_dataset(dataset, pruned_ids):
    pruned_dataset = []
    for i, (data, target) in enumerate(dataset):
        if i not in pruned_ids:
            pruned_dataset.append((data, target))

    return pruned_dataset


def main():


    config = load_config()

    ENSEMBLE_PATH = os.path.join(config['paths']['model_output'], config['ensemble']['name'])


    V3_PATH = os.path.join(ENSEMBLE_PATH, 'v3')
    os.makedirs(f'{V3_PATH}/images/weird', exist_ok=True)
    os.makedirs(f'{V3_PATH}/images/normal', exist_ok=True)


    # Create the directory if it does not exist
    if not os.path.exists(V3_PATH):
        os.makedirs(V3_PATH)

    # Load the models
    device = torch.device(config['training']['device'])
    models = load_models_v2(f'{ENSEMBLE_PATH}/models/', device)

    print("✔️ Naloženih modelov:", len(models))



    # Load Dataset

    dataset = torch.load(os.path.join(ENSEMBLE_PATH, 'test_dataset.pt'), weights_only=False) + \
              torch.load(os.path.join(ENSEMBLE_PATH, 'val_dataset.pt'), weights_only=False)

    if config['ensemble']['run_models']:
        # Get thre predicitons of the ensemble
        ensemble_predictions = ensemble_dataset_predictions(models, dataset, device)

        # Save to file using pickle
        with open(f'{V3_PATH}/ensemble_predictions.pk', 'wb') as f:
            pk.dump(ensemble_predictions, f)
    else:
        # Load the predictions from file
        with open(f'{V3_PATH}/ensemble_predictions.pk', 'rb') as f:
            ensemble_predictions = pk.load(f)

    # Get the statistics and thresholds of the ensemble
    ensemble_statistics = calculate_statistics(ensemble_predictions)

    if len(models) > 1:
        # izračun + shranjevanje
        stdev_thresholds = conduct_threshold_analysis(
            ensemble_statistics, 'stdev', low_to_high=True
        )
        stdev_thresholds.to_csv(os.path.join(V3_PATH, 'stdev_threshold_analysis.csv'))

        # graf: Accuracy vs. stdev threshold
        plot_threshold_analysis(
            stdev_thresholds['accuracy'],
            'Stdev Threshold Analysis for Accuracy',
            'Maximum Stdev Percentile (High → Low)',
            'Accuracy',
            os.path.join(V3_PATH, 'stdev_threshold_analysis.png'),
            flip=True,
        )

        # graf: F1 vs. stdev threshold
        plot_threshold_analysis(
            stdev_thresholds['f1'],
            'Stdev Threshold Analysis for F1 Score',
            'Maximum Stdev Percentile (High → Low)',
            'F1 Score',
            os.path.join(V3_PATH, 'stdev_threshold_analysis_f1.png'),
            flip=True,
        )
    else:
        print("⚠️  Preskakujem stdev analizo – samo en model naložen.")


    entropy_thresholds = conduct_threshold_analysis(
        ensemble_statistics, 'entropy', low_to_high=True
    )
    confidence_thresholds = conduct_threshold_analysis(
        ensemble_statistics, 'confidence', low_to_high=False
    )

    raw_confidence = ensemble_statistics['confidence'].apply(lambda x: (x / 2) + 0.5)
    ensemble_statistics.insert(4, 'raw_confidence', raw_confidence)

    # Plot confidence vs standard deviation
    plot_statistics_versus(
        'raw_confidence',
        'stdev',
        'Confidence',
        'Standard Deviation',
        'Confidence vs Standard Deviation',
        ensemble_statistics,
        f'{V3_PATH}/confidence_vs_stdev.png',
        annotate=True,
    )

    # Filter dataset for where confidence < .7 and stdev < .1
    weird_results = ensemble_statistics.loc[
        (
            (ensemble_statistics['raw_confidence'] < 0.7)
            & (ensemble_statistics['stdev'] < 0.1)
        )
    ]
    normal_results = ensemble_statistics.loc[
        ~(
            (ensemble_statistics['raw_confidence'] < 0.7)
            & (ensemble_statistics['stdev'] < 0.1)
        )
    ]

    # izberi do 3 "weird" + do 3 "normal" ID-je, ki RES obstajajo
    weird_ids = weird_results.index.to_list()[:3]
    normal_ids = normal_results.index.to_list()[:3]
    image_ids = weird_ids + normal_ids
    titles = [f'Weird: {i}' for i in weird_ids] + [f'Normal: {i}' for i in normal_ids]

    # OPOMBA: če še nisi popravil plot_image_grid v bolj robustno verzijo,
    # poskrbi, da je število slik deljivo z 'rows'. Za varno uporabo daj rows=1 ali pa vzemi 4 ali 6 slik.
    plot_image_grid(
        image_ids,
        dataset,
        rows=2,  # daj rows=1, če boš pogosto imel liho število slik
        path=f'{V3_PATH}/image_grid.png',
        titles=titles,
    )

    # Get the data ids in a list
    # Plot the images
    if not os.path.exists(f'{V3_PATH}/images'):
        os.makedirs(f'{V3_PATH}/images/weird')
        os.makedirs(f'{V3_PATH}/images/normal')

    for i in weird_results.itertuples():
        id = i.Index
        conf = i.raw_confidence
        stdev = i.stdev

        plot_single_image(
            id,
            dataset,
            f'{V3_PATH}/images/weird/{id}.png',
            title=f'ID: {id}, Confidence: {conf}, Stdev: {stdev}',
        )

    for i in normal_results.itertuples():
        id = i.Index
        conf = i.raw_confidence
        stdev = i.stdev

        plot_single_image(
            id,
            dataset,
            f'{V3_PATH}/images/normal/{id}.png',
            title=f'ID: {id}, Confidence: {conf}, Stdev: {stdev}',
        )

    # Calculate overall statistics
    overall_statistics = calculate_overall_statistics(ensemble_statistics)

    # Print overall statistics
    print(overall_statistics)

    # Print overall ensemble statistics
    print('Ensemble Statistics')
    print(f"Accuracy: {ensemble_statistics['correct'].mean()}")
    print(
        f"F1 Score: {metrics.f1_score(ensemble_statistics['actual'], ensemble_statistics['predicted'])}"
    )

    # Get the predictions, statistics and thresholds an individual model
    indv_id = config['ensemble']['individual_id']
    indv_predictions = select_individual_model(ensemble_predictions, indv_id)
    indv_statistics = calculate_statistics(indv_predictions)

    # Calculate entropy and confidence thresholds for individual model
    indv_entropy_thresholds = conduct_threshold_analysis(
        indv_statistics, 'entropy', low_to_high=True
    )
    indv_confidence_thresholds = conduct_threshold_analysis(
        indv_statistics, 'confidence', low_to_high=False
    )


    # Plot the threshold analysis for entropy
    plot_threshold_analysis(
        entropy_thresholds['accuracy'],
        'Entropy Threshold Analysis for Accuracy',
        'Entropy Threshold',
        'Accuracy',
        f'{V3_PATH}/entropy_threshold_analysis.png',
        indv_entropy_thresholds['accuracy'],
        flip=True,
    )
    plot_threshold_analysis(
        entropy_thresholds['f1'],
        'Entropy Threshold Analysis for F1 Score',
        'Entropy Threshold',
        'F1 Score',
        f'{V3_PATH}/entropy_threshold_analysis_f1.png',
        indv_entropy_thresholds['f1'],
        flip=True,
    )

    # Plot the threshold analysis for confidence
    plot_threshold_analysis(
        confidence_thresholds['accuracy'],
        'Confidence Threshold Analysis for Accuracy',
        'Confidence Threshold',
        'Accuracy',
        f'{V3_PATH}/confidence_threshold_analysis.png',
        indv_confidence_thresholds['accuracy'],
    )
    plot_threshold_analysis(
        confidence_thresholds['f1'],
        'Confidence Threshold Analysis for F1 Score',
        'Confidence Threshold',
        'F1 Score',
        f'{V3_PATH}/confidence_threshold_analysis_f1.png',
        indv_confidence_thresholds['f1'],
    )


if __name__ == '__main__':
    main()
