import torch
import torch.nn as nn
import torch.optim as optim
import shutil
import random as rand
import tomli as toml
import os
import warnings

import utils.models.cnn as cnn
from utils.data.datasets import prepare_datasets, initalize_dataloaders
import utils.training as train
import utils.testing as testn
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
# Generate seed for each set of runs

seed = rand.randint(0, 1000)

# Prepare data
train_dataset, val_dataset, test_dataset = prepare_datasets(
    config['paths']['mri_data'],
    config['paths']['xls_data'],
    config['dataset']['validation_split'],
    seed,
    config['training']['device'],
)
train_dataloader, val_dataloader, test_dataloader = initalize_dataloaders(
    train_dataset,
    val_dataset,
    test_dataset,
    config['hyperparameters']['batch_size'],
)

# Save datasets
model_folder_path = os.path.join(config['paths']['model_output'], str(config['model']['name']))
os.makedirs(model_folder_path, exist_ok=True)

torch.save(train_dataset, os.path.join(model_folder_path, 'train_dataset.pt'))
torch.save(val_dataset, os.path.join(model_folder_path, 'val_dataset.pt'))
torch.save(test_dataset, os.path.join(model_folder_path, 'test_dataset.pt'))

models_dir = os.path.join(model_folder_path, 'models')
os.makedirs(models_dir, exist_ok=True)

for i in range(config['training']['runs']):
    model = (
        cnn.CNN(
            config['model']['image_channels'],
            config['model']['clin_data_channels'],
            config['hyperparameters']['droprate'],
        )
        .float()
        .to(config['training']['device'])
    )
    criterion = nn.BCELoss()
    optimizer = optim.Adam(
        model.parameters(), lr=config['hyperparameters']['learning_rate']
    )

    runs_num = config['training']['runs']
    if not config['operation']['silent']:
        print(f'Training model {i + 1} / {runs_num} with seed {seed}...')

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        history = train.train_model(
            model, train_dataloader, val_dataloader, criterion, optimizer, config
        )

    tes_acc = testn.test_model(model, test_dataloader, config)

    model_save_path = os.path.join(models_dir, f"{i + 1}_s-{seed}")

    torch.save(model, model_save_path + '.pt')
    history.to_csv(model_save_path + '_history.csv', index=True)

    with open(model_save_path + '_test_acc.txt', 'w') as f:
        f.write(str(tes_acc))

    with open(os.path.join(model_folder_path, 'summary.txt'), 'a') as f:
        f.write(f'{i + 1}: Test Accuracy: {tes_acc}\n')

