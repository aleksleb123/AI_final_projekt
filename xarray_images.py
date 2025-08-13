import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir))  # All/
sys.path.insert(0, parent_dir)



import torch
import pandas as pd
from pandas.core.internals.managers import BlockManager
from utils.data.datasets import ADNIDataset

torch.serialization.add_safe_globals([
    ADNIDataset,
    pd.DataFrame,
    BlockManager
])


import os
import torch
import random
import matplotlib.pyplot as plt

# 📂 Nastavi pot do model folderja
MODEL_NAME = "cnn-50x30"
BASE_PATH = os.path.join("saved_models", MODEL_NAME)
V4_PATH = os.path.join(BASE_PATH, "v4")
os.makedirs(V4_PATH, exist_ok=True)

# 🔁 Naloži dataset
TEST_PATH = os.path.join(BASE_PATH, "test_dataset.pt")
test_dataset = torch.load(TEST_PATH, weights_only=False)

# 🎯 Funkcija: izberi naključno sliko z dano oznako (label)
def get_random_image_by_label(dataset, label=1):
    candidates = [i for i in range(len(dataset)) if dataset[i][1][1].item() == label]
    if not candidates:
        raise ValueError(f"No samples found with label={label}.")
    idx = random.choice(candidates)
    return dataset[idx][0][0].numpy(), idx  # [0][0] -> samo MRI, brez kanalne dimenzije

# 💾 Funkcija: shrani sliko
def plot_and_save_image(img, name):
    if img.shape[0] == 1:
        img = img[0]  # odstrani channel dimenzijo, postane (D, H, W)

    # izberi sredinsko slice (npr. po prvi dimenziji)
    slice_index = img.shape[0] // 2
    slice_2d = img[slice_index, :, :]  # (H, W)

    plt.imshow(slice_2d, cmap='gray')
    plt.axis('off')
    save_path = os.path.join(V4_PATH, name)
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()
    print(f"✔️ Shranjena: {save_path}")


# 🔄 Generiraj 3 pozitivne (AD) in 3 negativne (kontrola) slike
for i in range(1, 4):
    img, idx = get_random_image_by_label(test_dataset, label=1)
    plot_and_save_image(img, f"positive_{i:03d}.png")

for i in range(1, 4):
    img, idx = get_random_image_by_label(test_dataset, label=0)
    plot_and_save_image(img, f"negative_{i:03d}.png")

print("\n✅ Vse slike shranjene v:", V4_PATH)
