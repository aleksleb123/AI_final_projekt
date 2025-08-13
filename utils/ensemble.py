import torch
import os
from glob import glob


# This file contains functions to ensemble a folder of models and evaluate them on a test set, with included uncertainty estimation.


def load_models(folder, device):
    glob_path = os.path.join(folder, "*.pt")
    model_files = glob(glob_path)

    models = []
    model_descs = []

    for model_file in model_files:
        model = torch.load(model_file, map_location=device, weights_only=False)
        models.append(model)

        # Extract model description from filename
        desc = os.path.basename(model_file)
        model_descs.append(os.path.splitext(desc)[0])

    return models, model_descs


def ensemble_predict(models, input):
    predictions = []
    for model in models:
        model.eval()
        with torch.no_grad():
            # Apply model and extract positive class predictions
            output = model(input)[:, 1]
            predictions.append(output)

    # Calculate mean and variance of predictions
    predictions = torch.stack(predictions)
    mean = predictions.mean()
    variance = predictions.var()

    return mean, variance


def ensemble_predict_strict_classes(models, input):
    predictions = []
    for model in models:
        model.eval()
        with torch.no_grad():
            # Apply model and extract prediction
            output = model(input)
            _, predicted = torch.max(output.data, 1)
            predictions.append(predicted.item())

    pos_votes = len([p for p in predictions if p == 1])
    neg_votes = len([p for p in predictions if p == 0])

    return pos_votes / len(models), pos_votes, neg_votes


# Prune the ensemble by removing models with low accuracy on the test set, as determined in their tes_acc.txt files
def prune_models(models, model_descs, folder, threshold):
    new_models = []
    new_descs = []

    for model, desc in zip(models, model_descs):
        acc_path = os.path.join(folder, desc + "_test_acc.txt")
        with open(acc_path, "r") as f:
            acc = float(f.read())
            if acc >= threshold:
                new_models.append(model)
                new_descs.append(desc)

    return new_models, new_descs
