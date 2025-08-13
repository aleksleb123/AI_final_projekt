# NEEDS TO BE FINISHED
# TODO CHECK ABOUT IMAGE DIMENSIONS
# TODO ENSURE ITERATION WORKS
import glob
import nibabel as nib
import random
import torch
from torch.utils.data import Dataset
import pandas as pd
from torch.utils.data import DataLoader
import math


"""
Prepares CustomDatasets for training, validating, and testing CNN
"""


def prepare_datasets(mri_dir, xls_file, val_split=0.2, seed=50, device=None):
    if device is None:
        device = torch.device('cpu')

    rndm = random.Random(seed)
    xls_data = pd.read_csv(xls_file)
    xls_data.columns = xls_data.columns.str.strip()  # odstrani presledke
    xls_data = xls_data.set_index('Image Data ID')
    raw_data = glob.glob(mri_dir + '*')
    AD_list = []
    NL_list = []

    # TODO Check that image is in CSV?
    for image in raw_data:
        if 'NL' in image:
            NL_list.append(image)
        elif 'AD' in image:
            AD_list.append(image)

    rndm.shuffle(AD_list)
    rndm.shuffle(NL_list)

    train_list, val_list, test_list = get_train_val_test(AD_list, NL_list, val_split)

    train_dataset = ADNIDataset(train_list, xls_data, device=device)
    val_dataset = ADNIDataset(val_list, xls_data, device=device)
    test_dataset = ADNIDataset(test_list, xls_data, device=device)

    return train_dataset, val_dataset, test_dataset

    # TODO  Normalize data? Later add / Exctract clinical data? Which data?


"""
Returns train_list, val_list and test_list in format [(image, id), ...] each
"""


def get_train_val_test(AD_list, NL_list, val_split):
    train_list, val_list, test_list = [], [], []
    # For the purposes of this split, the val_split constitutes the validation and testing split, as they are divided evenly

    # get the overall length of the data
    AD_len = len(AD_list)
    NL_len = len(NL_list)

    # First, determine the length of each of the sets
    AD_val_len = int(math.ceil(AD_len * val_split * 0.5))
    NL_val_len = int(math.ceil(NL_len * val_split * 0.5))

    AD_test_len = int(math.floor(AD_len * val_split * 0.5))
    NL_test_len = int(math.floor(NL_len * val_split * 0.5))

    AD_train_len = AD_len - AD_val_len - AD_test_len
    NL_train_len = NL_len - NL_val_len - NL_test_len

    # Add the data to the sets
    for i in range(AD_train_len):
        train_list.append((AD_list[i], 1))
    for i in range(NL_train_len):
        train_list.append((NL_list[i], 0))

    for i in range(AD_train_len, AD_train_len + AD_val_len):
        val_list.append((AD_list[i], 1))
    for i in range(NL_train_len, NL_train_len + NL_val_len):
        val_list.append((NL_list[i], 0))

    for i in range(AD_train_len + AD_val_len, AD_len):
        test_list.append((AD_list[i], 1))
    for i in range(NL_train_len + NL_val_len, NL_len):
        test_list.append((NL_list[i], 0))

    return train_list, val_list, test_list


class ADNIDataset(Dataset):
    def __init__(self, mri, xls: pd.DataFrame, device=torch.device('cpu')):
        self.mri_data = mri  # DATA IS A LIST WITH TUPLES (image_dir, class_id)
        self.xls_data = xls
        self.device = device

    def __len__(self):
        return len(self.mri_data)

    #OLD FUNCTION
    # def _xls_to_tensor(self, xls_data: pd.Series):
    #     # Get used data
    #
    #     # data = xls_data.loc[['Sex', 'Age (current)', 'PTID', 'DXCONFID (1=uncertain, 2= mild, 3= moderate, 4=high confidence)', 'Alz_csf']]
    #     data = xls_data.loc[['Sex', 'Age (current)']]
    #
    #     data.replace({'M': 0, 'F': 1}, inplace=True)
    #
    #     # Convert to tensor
    #     xls_tensor = torch.tensor(data.values.astype(float))
    #
    #     return xls_tensor

    def _xls_to_tensor(self, xls_data: pd.Series):
        data = xls_data.loc[['Sex', 'Age (current)']].copy()

        # Odstrani presledke iz stringovnih vrednosti
        data = data.apply(lambda x: x.strip() if isinstance(x, str) else x)

        # Pretvori spol v število
        data['Sex'] = 0 if data['Sex'].strip() == 'M' else 1

        # Pretvori v tensor
        xls_tensor = torch.tensor(data.values.astype(float))
        return xls_tensor

    def __getitem__(
        self, idx
    ):  # RETURNS TUPLE WITH IMAGE AND CLASS_ID, BASED ON INDEX IDX
        mri_path, class_id = self.mri_data[idx]
        mri = nib.load(mri_path)
        mri_data = mri.get_fdata()

        xls = self.xls_data.iloc[idx]

        # Convert xls data to tensor
        xls_tensor = self._xls_to_tensor(xls)
        mri_tensor = torch.from_numpy(mri_data).unsqueeze(0)

        class_id = torch.tensor([class_id])
        # Convert to one-hot and squeeze
        class_id = torch.nn.functional.one_hot(class_id, num_classes=2).squeeze(0)

        # Convert to float
        mri_tensor = mri_tensor.float().to(self.device)
        xls_tensor = xls_tensor.float().to(self.device)
        class_id = class_id.float().to(self.device)

        return (mri_tensor, xls_tensor), class_id


def initalize_dataloaders(
    training_data,
    val_data,
    test_data,
    batch_size=64,
):
    train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=(batch_size // 4), shuffle=True)
    val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=True)
    return train_dataloader, val_dataloader, test_dataloader
