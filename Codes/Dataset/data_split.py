import os
import shutil
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor

# ==== PATHS ====
image_dir = 'Plots_ECG'         # folder with images
csv_path = 'labels_ECG.csv'    # CSV with 70200 rows
npy_path = 'ECG_segments.npy'    # NumPy array (70200, 256, 2)
output_dir = 'Data_split'        # output folder
os.makedirs(output_dir, exist_ok=True   )
# ==== CREATE OUTPUT STRUCTURE ====
splits = ['train', 'val', 'test']
for split in splits:
    os.makedirs(os.path.join(output_dir, split, 'images'), exist_ok=True)

# ==== LOAD DATA ====
df = pd.read_csv(csv_path,header=None)
arr = np.load(npy_path)
print(len(df))
assert len(df) == len(arr) == 70200, "Mismatch in number of samples!"

# ==== SHUFFLE ====
np.random.seed(42)
indices = np.random.permutation(len(df))

# ==== SPLIT ====
n_total = len(indices)
n_train = int(0.8 * n_total)
n_val = int(0.1 * n_total)
n_test = n_total - n_train - n_val

splits_idx = {
    "train": indices[:n_train],
    "val": indices[n_train:n_train + n_val],
    "test": indices[n_train + n_val:]
}

# ==== COPY FUNCTION (runs in parallel threads) ====
def copy_image(i, dest_dir):
    img_name = f"ecg_plot_{i}.png"
    src = os.path.join(image_dir, img_name)
    dst = os.path.join(dest_dir, img_name)
    if not os.path.exists(dst):
        try:
            shutil.copy2(src, dst)
        except Exception as e:
            print(f"Error copying {img_name}: {e}")

# ==== SPLIT CREATION ====
def process_split(split_name, split_indices):
    split_dir = os.path.join(output_dir, split_name)
    img_dir = os.path.join(split_dir, 'images')

    split_df = df.iloc[split_indices]
    split_arr = arr[split_indices]

    # Copy images in parallel
    with ThreadPoolExecutor(max_workers=16) as executor:
        executor.map(lambda i: copy_image(i, img_dir), split_indices)

    # Save corresponding CSV and NPY
    split_df.to_csv(os.path.join(split_dir, f"{split_name}.csv"), index=False)
    np.save(os.path.join(split_dir, f"{split_name}.npy"), split_arr)

    print(f"{split_name}: {len(split_indices)} samples processed.")

# ==== RUN ALL SPLITS ====
for name, idx in splits_idx.items():
    process_split(name, idx)

print("Done — created train/val/test splits with CSV, NPY, and image folders.")
