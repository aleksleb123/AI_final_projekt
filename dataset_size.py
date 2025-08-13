# This file is just to tell how many total images there are in the dataset

import threshold_refac as tr
import torch
import os

config = tr.load_config()
ENSEMBLE_PATH = os.path.join(config['paths']['model_output'], config['ensemble']['name'])

test_dset = torch.load(f'{ENSEMBLE_PATH}/test_dataset.pt', weights_only=False)
val_dset = torch.load(f'{ENSEMBLE_PATH}/val_dataset.pt', weights_only=False)
train_dset = torch.load(f'{ENSEMBLE_PATH}/train_dataset.pt', weights_only=False)


print(
    f'Total number of images in dataset: {len(test_dset) + len(val_dset) + len(train_dset)}'
)
print(f'Test: {len(test_dset)}, Val: {len(val_dset)}, Train: {len(train_dset)}')


def preprocess_data(data, device):
    mri, xls = data
    mri = mri.unsqueeze(0).to(device)
    xls = xls.unsqueeze(0).to(device)
    return (mri, xls)


# Loop through images and determine how many are positive and negative
positive = 0
negative = 0
for _, (_, target) in enumerate(test_dset + train_dset + val_dset):
    actual = list(target.cpu().numpy())[1].item()
    if actual == 1:
        positive += 1
    else:
        negative += 1

print(f'Positive: {positive}, Negative: {negative}')
