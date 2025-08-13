import numpy as np
import sklearn.metrics as mt


# ECE from https://towardsdatascience.com/expected-calibration-error-ece-a-step-by-step-visual-explanation-with-python-code-c3e9aa12937d
def ECE(predicted_labels, confidences, true_labels, M=5):
    # Uniform M bins
    bin_boundaries = np.linspace(0, 1, M + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    # get correct/false
    accuracies = predicted_labels == true_labels

    ece = np.zeros(1)

    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # bin sample
        in_bin = np.logical_and(
            confidences > bin_lower.item(), confidences <= bin_upper.item()
        )
        prob_in_bin = in_bin.mean()

        if prob_in_bin > 0:
            accuracy_in_bin = accuracies[in_bin].mean()
            avg_confid = confidences[in_bin].mean()
            ece += np.abs(avg_confid - accuracy_in_bin) * prob_in_bin

    return ece[0]


# Maximum Calibration error - maximum of error per bin
def MCE(predicted_labels, confidences, true_labels, M=5):
    bin_boundaries = np.linspace(0, 1, M + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    # get correct/false
    accuracies = predicted_labels == true_labels

    mces = []

    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # bin sample
        in_bin = np.logical_and(
            confidences >= bin_lower.item(), confidences < bin_upper.item()
        )
        prob_in_bin = in_bin.mean()

        if prob_in_bin > 0:
            accuracy_in_bin = accuracies[in_bin].mean()
            avg_confid = confidences[in_bin].mean()
            mces.append(np.abs(avg_confid - accuracy_in_bin))

    return max(mces) if mces else 0.0



def F1(predicted_labels, true_labels):
    tp = np.sum(np.logical_and(predicted_labels == 1, true_labels == 1))
    fp = np.sum(np.logical_and(predicted_labels == 1, true_labels == 0))
    fn = np.sum(np.logical_and(predicted_labels == 0, true_labels == 1))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    if precision + recall == 0:
        return 0.0  # ali np.nan, če želiš kaznovati to kot neveljavno

    return 2 * (precision * recall) / (precision + recall)



# Uses sklearn's AUC function
# Requieres confidences to be the predicted probabilities for the positive class
def AUC(confidences, true_labels):
    fpr, tpr, _ = mt.roc_curve(true_labels, confidences)
    return mt.auc(fpr, tpr)


def entropy(confidences):
    return -np.sum(confidences * np.log(confidences))

### Negative Log Likelyhood for binary classification
def nll_binary(confidences, true_labels, epsilon=1e-12):
    confidences = np.clip(confidences, epsilon, 1 - epsilon)  # 🛡️ zaščita pred 0 in 1
    return -np.sum(np.log(confidences[true_labels == 1])) - np.sum(np.log(1 - confidences[true_labels == 0]))


### Breier score for binary classification
def brier_binary(confidences, true_labels):
    return np.mean((confidences - true_labels) ** 2)



