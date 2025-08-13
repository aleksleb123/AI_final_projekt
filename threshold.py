import pandas as pd
import numpy as np
import os
import tomli as toml
import utils.ensemble as ens
import torch
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
from tqdm import tqdm
import utils.metrics as met
import itertools as it
import matplotlib.ticker as ticker


# Define plotting helper function
def plot_coverage(
    percentiles,
    ensemble_results,
    individual_results,
    title,
    x_lablel,
    y_label,
    save_path,
    flip=False,
):
    fig, ax = plt.subplots()
    plt.plot(
        percentiles,
        ensemble_results,
        'ob',
        label='Ensemble',
    )
    plt.plot(
        percentiles,
        individual_results,
        'xr',
        label='Individual (on entire dataset)',
    )
    plt.xlabel(x_lablel)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()
    if flip:
        plt.gca().invert_xaxis()
    ax.xaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0))
    plt.savefig(save_path)
    plt.close()


RUN = True

# CONFIGURATION
if os.getenv('ADL_CONFIG_PATH') is None:
    with open('config.toml', 'rb') as f:
        config = toml.load(f)
else:
    with open(os.getenv('ADL_CONFIG_PATH'), 'rb') as f:
        config = toml.load(f)


ENSEMBLE_PATH = os.path.join(config['paths']['model_output'], config['ensemble']['name'])

V2_PATH = os.path.join(ENSEMBLE_PATH, 'v2')


# Result is a 1x2 tensor, with the softmax of the 2 predicted classes
# Want to convert to a predicted class and a confidence
def output_to_confidence(result):
    predicted_class = torch.argmax(result).item()
    confidence = (torch.max(result).item() - 0.5) * 2

    return torch.Tensor([predicted_class, confidence])


# This function conducts tests on the models and returns the results, as well as saving the predictions and metrics
def get_predictions(config):
    models, model_descs = ens.load_models(
        f'{ENSEMBLE_PATH}/models/',
        config['training']['device'],
    )
    models = [model.to(config['training']['device']) for model in models]

    from utils.data.datasets import ADNIDataset
    import torch.serialization

    with torch.serialization.safe_globals([ADNIDataset]):
        test_set = torch.load(f'{ENSEMBLE_PATH}/test_dataset.pt', weights_only=False) + \
                   torch.load(f'{ENSEMBLE_PATH}/val_dataset.pt', weights_only=False)

    print(f'Loaded {len(test_set)} samples')

    # [([model results], labels)]
    results = []

    # [(class_1, class_2, true_label)]
    indv_results = []

    for _, (data, target) in tqdm(
        enumerate(test_set),
        total=len(test_set),
        desc='Getting predictions',
        unit='sample',
    ):
        mri, xls = data
        mri = mri.unsqueeze(0).to(config['training']['device'])
        xls = xls.unsqueeze(0).to(config['training']['device'])
        data = (mri, xls)
        res = []
        for j, model in enumerate(models):
            model.eval()
            with torch.no_grad():
                output = model(data)

                output = output.tolist()

                if j == 0:
                    indv_results.append((output[0][0], output[0][1], target[1].item()))

                res.append(output)
        results.append((res, target.tolist()))

    # The results are a list of tuples, where each tuple contains a list of model outputs and the true label
    # We want to convert this to 2 list of tuples, one with the ensemble predicted class, ensemble confidence and true label
    # And one with the ensemble predicted class, ensemble standard deviation and true label

    # [(ensemble predicted class, ensemble confidence, true label)]
    confidences = []

    # [(ensemble predicted class, ensemble standard deviation, true label)]
    stdevs = []

    # [(ensemble predicted class, ensemble entropy, true label)]
    entropies = []

    for result in results:
        model_results, true_label = result
        # Get the ensemble mean and variance with numpy, as these are lists
        mean = np.mean(model_results, axis=0)
        variance = np.var(model_results, axis=0)

        # Calculate the entropy
        entropy = -1 * np.sum(mean * np.log(mean))

        # Calculate confidence and standard deviation
        confidence = (np.max(mean) - 0.5) * 2
        stdev = np.sqrt(variance)

        # Get the predicted class
        predicted_class = np.argmax(mean)

        # Get the confidence and standard deviation of the predicted class
        pc_stdev = np.squeeze(stdev)[predicted_class]
        # Get the individual classes
        class_1 = mean[0][0]
        class_2 = mean[0][1]

        # Get the true label
        true_label = true_label[1]

        confidences.append((predicted_class, confidence, true_label, class_1, class_2))
        stdevs.append((predicted_class, pc_stdev, true_label, class_1, class_2))
        entropies.append((predicted_class, entropy, true_label, class_1, class_2))

    return results, confidences, stdevs, entropies, indv_results


if RUN:
    results, confs, stdevs, entropies, indv_results = get_predictions(config)
    # Convert to pandas dataframes
    confs_df = pd.DataFrame(
        confs,
        columns=['predicted_class', 'confidence', 'true_label', 'class_1', 'class_2'],
    )
    stdevs_df = pd.DataFrame(
        stdevs, columns=['predicted_class', 'stdev', 'true_label', 'class_1', 'class_2']
    )

    entropies_df = pd.DataFrame(
        entropies,
        columns=['predicted_class', 'entropy', 'true_label', 'class_1', 'class_2'],
    )

    indv_df = pd.DataFrame(indv_results, columns=['class_1', 'class_2', 'true_label'])

    if not os.path.exists(V2_PATH):
        os.makedirs(V2_PATH)

    confs_df.to_csv(f'{V2_PATH}/ensemble_confidences.csv')
    stdevs_df.to_csv(f'{V2_PATH}/ensemble_stdevs.csv')
    entropies_df.to_csv(f'{V2_PATH}/ensemble_entropies.csv')
    indv_df.to_csv(f'{V2_PATH}/individual_results.csv')
else:
    confs_df = pd.read_csv(f'{V2_PATH}/ensemble_confidences.csv')
    stdevs_df = pd.read_csv(f'{V2_PATH}/ensemble_stdevs.csv')
    entropies_df = pd.read_csv(f'{V2_PATH}/ensemble_entropies.csv')
    indv_df = pd.read_csv(f'{V2_PATH}/individual_results.csv')


# Plot confidence vs standard deviation, and change color of dots based on if they are correct
correct_conf = confs_df[confs_df['predicted_class'] == confs_df['true_label']]
incorrect_conf = confs_df[confs_df['predicted_class'] != confs_df['true_label']]

correct_stdev = stdevs_df[stdevs_df['predicted_class'] == stdevs_df['true_label']]
incorrect_stdev = stdevs_df[stdevs_df['predicted_class'] != stdevs_df['true_label']]

correct_ent = entropies_df[
    entropies_df['predicted_class'] == entropies_df['true_label']
]
incorrect_ent = entropies_df[
    entropies_df['predicted_class'] != entropies_df['true_label']
]

plot, ax = plt.subplots()
plt.scatter(
    correct_conf['confidence'],
    correct_stdev['stdev'],
    color='green',
    label='Correct Prediction',
)
plt.scatter(
    incorrect_conf['confidence'],
    incorrect_stdev['stdev'],
    color='red',
    label='Incorrect Prediction',
)
plt.xlabel('Confidence (Raw Value)')
plt.ylabel('Standard Deviation (Raw Value)')
plt.title('Confidence vs Standard Deviation')
plt.legend()
plt.savefig(f'{V2_PATH}/confidence_vs_stdev.png')

plt.close()

# Do the same for confidence vs entropy
plot, ax = plt.subplots()
plt.scatter(
    correct_conf['confidence'],
    correct_ent['entropy'],
    color='green',
    label='Correct Prediction',
)
plt.scatter(
    incorrect_conf['confidence'],
    incorrect_ent['entropy'],
    color='red',
    label='Incorrect Prediction',
)
plt.xlabel('Confidence (Raw Value)')
plt.ylabel('Entropy (Raw Value)')
plt.title('Confidence vs Entropy')
plt.legend()
plt.savefig(f'{V2_PATH}/confidence_vs_entropy.png')

plt.close()


# Calculate individual model accuracy and entropy
# Determine predicted class
indv_df['predicted_class'] = indv_df[['class_1', 'class_2']].idxmax(axis=1)
indv_df['predicted_class'] = indv_df['predicted_class'].apply(
    lambda x: 0 if x == 'class_1' else 1
)
indv_df['correct'] = indv_df['predicted_class'] == indv_df['true_label']
accuracy_indv = indv_df['correct'].mean()
f1_indv = met.F1(
    indv_df['predicted_class'].to_numpy(), indv_df['true_label'].to_numpy()
)
auc_indv = metrics.roc_auc_score(
    indv_df['true_label'].to_numpy(), indv_df['class_2'].to_numpy()
)
indv_df['entropy'] = -1 * indv_df[['class_1', 'class_2']].apply(
    lambda x: x * np.log(x), axis=0
).sum(axis=1)

# Calculate percentiles for confidence and standard deviation
quantiles_conf = confs_df.quantile(np.linspace(0, 1, 11), interpolation='lower')[
    'confidence'
]
quantiles_stdev = stdevs_df.quantile(np.linspace(0, 1, 11), interpolation='lower')[
    'stdev'
]

# Additionally for individual confidence
quantiles_indv_conf = indv_df.quantile(np.linspace(0, 1, 11), interpolation='lower')[
    'class_2'
]

# For indivual entropy
quantiles_indv_entropy = indv_df.quantile(np.linspace(0, 1, 11), interpolation='lower')[
    'entropy'
]

#

accuracies_conf = []
# Use the quantiles to calculate the coverage
iter_conf = it.islice(quantiles_conf.items(), 0, None)
for quantile in iter_conf:
    percentile = quantile[0]

    filt = confs_df[confs_df['confidence'] >= quantile[1]]
    accuracy = (
        filt[filt['predicted_class'] == filt['true_label']].shape[0] / filt.shape[0]
    )
    f1 = met.F1(filt['predicted_class'].to_numpy(), filt['true_label'].to_numpy())

    accuracies_conf.append({'percentile': percentile, 'accuracy': accuracy, 'f1': f1})

accuracies_df = pd.DataFrame(accuracies_conf)

indv_conf = []
# Use the quantiles to calculate the coverage
iter_conf = it.islice(quantiles_indv_conf.items(), 0, None)
for quantile in iter_conf:
    percentile = quantile[0]

    filt = indv_df[indv_df['class_2'] >= quantile[1]]
    accuracy = filt['correct'].mean()
    f1 = met.F1(filt['predicted_class'].to_numpy(), filt['true_label'].to_numpy())

    indv_conf.append({'percentile': percentile, 'accuracy': accuracy, 'f1': f1})

indv_conf_df = pd.DataFrame(indv_conf)

# Do the same for entropy
indv_entropy = []
iter_entropy = it.islice(quantiles_indv_entropy.items(), 0, None)
for quantile in iter_entropy:
    percentile = quantile[0]

    filt = indv_df[indv_df['entropy'] <= quantile[1]]
    accuracy = filt['correct'].mean()
    f1 = met.F1(filt['predicted_class'].to_numpy(), filt['true_label'].to_numpy())

    indv_entropy.append({'percentile': percentile, 'accuracy': accuracy, 'f1': f1})

indv_entropy_df = pd.DataFrame(indv_entropy)


# Plot the coverage for confidence and accuracy
plot_coverage(
    accuracies_df['percentile'],
    accuracies_df['accuracy'],
    indv_conf_df['accuracy'],
    'Confidence Accuracy Coverage Plot',
    'Minimum Confidence Percentile (Low to High)',
    'Accuracy',
    f'{V2_PATH}/coverage_conf.png',
)

# Plot the coverage for confidence and F1
plot_coverage(
    accuracies_df['percentile'],
    accuracies_df['f1'],
    indv_conf_df['f1'],
    'Confidence F1 Coverage Plot',
    'Minimum Confidence Percentile (Low to High)',
    'F1',
    f'{V2_PATH}/f1_coverage_conf.png',
)

# IZRAČUN: coverage po standardni deviaciji (manjkajoči del)
accuracies_stdev = []
iter_stdev = it.islice(quantiles_stdev.items(), 0, None)
for percentile, cutoff in iter_stdev:
    # nižja stdev je boljša → zadržimo vzorce z stdev ≤ cutoff
    filt = stdevs_df[stdevs_df['stdev'] <= cutoff]

    if len(filt) == 0:
        acc_val = np.nan
        f1_val = np.nan
    else:
        acc_val = (filt['predicted_class'] == filt['true_label']).mean()
        f1_val = met.F1(
            filt['predicted_class'].to_numpy(),
            filt['true_label'].to_numpy()
        )

    accuracies_stdev.append({
        'percentile': percentile,
        'accuracy': acc_val,
        'f1': f1_val
    })

accuracies_stdev_df = pd.DataFrame(accuracies_stdev)


# --- Standard Deviation Accuracy Coverage Plot (popravljeno) ---
fig, ax = plt.subplots()
plt.plot(
    accuracies_stdev_df['percentile'],
    accuracies_stdev_df['accuracy'],
    'ob',
    label='Ensemble'
)
plt.plot(
    accuracies_stdev_df['percentile'],
    [accuracy_indv] * len(accuracies_stdev_df['percentile']),
    'xr',
    label='Individual (on entire dataset)'
)
plt.xlabel('Maximum Standard Deviation Percentile (High to Low)')
plt.ylabel('Accuracy')
plt.title('Standard Deviation Accuracy Coverage Plot')
plt.legend()
plt.gca().invert_xaxis()  # ker filtriraš z "≤ quantile" (nižja stdev je boljša)
ax.xaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0))
plt.tight_layout()
plt.savefig(f'{V2_PATH}/coverage_stdev.png')
plt.close()

# --- Standard Deviation F1 Coverage Plot (naj ostane) ---
fig, ax = plt.subplots()
plt.plot(
    accuracies_stdev_df['percentile'], accuracies_stdev_df['f1'], 'ob', label='Ensemble'
)
plt.plot(
    accuracies_stdev_df['percentile'],
    [f1_indv] * len(accuracies_stdev_df['percentile']),
    'xr',
    label='Individual (on entire dataset)',
)
plt.xlabel('Maximum Standard Deviation Percentile (High to Low)')
plt.ylabel('F1')
plt.title('Standard Deviation F1 Coverage Plot')
plt.legend()
plt.gca().invert_xaxis()
ax.xaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0))
plt.tight_layout()
plt.savefig(f'{V2_PATH}/coverage_f1_stdev.png')
plt.close()



# Print overall accuracy
overall_accuracy = (
    confs_df[confs_df['predicted_class'] == confs_df['true_label']].shape[0]
    / confs_df.shape[0]
)
overall_f1 = met.F1(
    confs_df['predicted_class'].to_numpy(), confs_df['true_label'].to_numpy()
)
# Calculate ECE and MCE
conf_ece = met.ECE(
    confs_df['predicted_class'].to_numpy(),
    confs_df['confidence'].to_numpy(),
    confs_df['true_label'].to_numpy(),
)

stdev_ece = met.ECE(
    stdevs_df['predicted_class'].to_numpy(),
    stdevs_df['stdev'].to_numpy(),
    stdevs_df['true_label'].to_numpy(),
)


print(f'Overall accuracy: {overall_accuracy}, Overall F1: {overall_f1},')
print(f'Confidence ECE: {conf_ece}')
print(f'Standard Deviation ECE: {stdev_ece}')


# Repeat for entropy
quantiles_entropy = entropies_df.quantile(np.linspace(0, 1, 11), interpolation='lower')[
    'entropy'
]

accuracies_entropy = []
iter_entropy = it.islice(quantiles_entropy.items(), 0, None)
for quantile in iter_entropy:
    percentile = quantile[0]

    filt = entropies_df[entropies_df['entropy'] <= quantile[1]]
    accuracy = (
        filt[filt['predicted_class'] == filt['true_label']].shape[0] / filt.shape[0]
    )
    f1 = met.F1(filt['predicted_class'].to_numpy(), filt['true_label'].to_numpy())

    accuracies_entropy.append(
        {'percentile': percentile, 'accuracy': accuracy, 'f1': f1}
    )

accuracies_entropy_df = pd.DataFrame(accuracies_entropy)


# Plot the coverage for entropy and accuracy
plot_coverage(
    accuracies_entropy_df['percentile'],
    accuracies_entropy_df['accuracy'],
    indv_entropy_df['accuracy'],
    'Entropy Accuracy Coverage Plot',
    'Minimum Entropy Percentile (Low to High)',
    'Accuracy',
    f'{V2_PATH}/coverage_entropy.png',
)

# Plot the coverage for entropy and F1
plot_coverage(
    accuracies_entropy_df['percentile'],
    accuracies_entropy_df['f1'],
    indv_entropy_df['f1'],
    'Entropy F1 Coverage Plot',
    'Maximum Entropy Percentile (High to Low)',
    'F1',
    f'{V2_PATH}/f1_coverage_entropy.png',
    flip=True,
)
