from utils.training import evaluate_accuracy


def test_model(model, test_loader, config):
    accuracy, predictions, actual = evaluate_accuracy(model, test_loader)

    return accuracy
