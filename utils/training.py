import torch
import pandas as pd
from tqdm import tqdm


def train_epoch(model, train_loader, val_loader, criterion, optimizer, config, epoch):
    model.train()
    train_loss = 0
    val_loss = 0
    for i, (data, target) in tqdm(
        enumerate(train_loader),
        total=len(train_loader),
        desc=" - Epoch " + str(epoch + 1) + "/" + str(config["training"]["max_epochs"]),
        unit="batch",
        disable=config["operation"]["silent"],
    ):
        optimizer.zero_grad()

        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)

    model.eval()
    with torch.no_grad():
        for i, (data, target) in enumerate(val_loader):
            output = model(data)
            loss = criterion(output, target)
            val_loss += loss.item()
        val_loss /= len(val_loader)

    return train_loss, val_loss


def evaluate_accuracy(model, loader):
    model.eval()
    correct = 0
    total = 0
    predictions = []
    actual = []

    with torch.no_grad():
        for data, target in loader:
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            _, expected = torch.max(target.data, 1)

            total += target.size(0)
            correct += (predicted == expected).sum().item()

            out = output[:, 1].tolist()
            predictions.extend(out)

            act = target[:, 1].tolist()
            actual.extend(act)

    return correct / total, predictions, actual


def train_model(model, train_loader, val_loader, criterion, optimizer, config):
    history = pd.DataFrame(
        columns=["Epoch", "Train Loss", "Val Loss", "Train Acc", "Val Acc"]
    ).set_index("Epoch")

    for epoch in range(config["training"]["max_epochs"]):
        train_loss, val_loss = train_epoch(
            model, train_loader, val_loader, criterion, optimizer, config, epoch
        )

        train_acc, _, _ = evaluate_accuracy(model, train_loader)
        val_acc, _, _ = evaluate_accuracy(model, val_loader)

        if config["operation"]["silent"] is False:
            print(
                f" --- Epoch {epoch + 1} - Train Loss: {round(train_loss, 3)}, Val Loss: {round(val_loss, 3)}, Train Acc: {round(train_acc, 3) * 100}%, Val Acc: {round(val_acc, 3) * 100}%"
            )

        history.loc[epoch] = [train_loss, val_loss, train_acc, val_acc]

    return history
