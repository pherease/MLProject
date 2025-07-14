import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from PIL import Image  # ✅ NEW

# ------------------------------
# PARAMETERS
# ------------------------------
input_dir = './data'
output_dir = './split_data'
split_ratios = (0.4, 0.3, 0.3)  # train, val, test
IMG_SIZE = (256, 256)  # ✅ You can change to (128,128) or (512,512) etc.

assert sum(split_ratios) == 1.0, "Splits must sum to 1.0"

# ------------------------------
# Helper to make dirs
# ------------------------------
def make_dirs(base, classes):
    for split in ['train', 'val', 'test']:
        for cls in classes:
            os.makedirs(os.path.join(base, split, cls), exist_ok=True)

# ------------------------------
# Helper to clear output_dir if exists
# ------------------------------
def clear_output_dir(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)

# ------------------------------
# Split data and prepare CSV records
# ------------------------------
clear_output_dir(output_dir)

classes = os.listdir(input_dir)
classes = [c for c in classes if os.path.isdir(os.path.join(input_dir, c))]
print(f"Found classes: {classes}")

make_dirs(output_dir, classes)

csv_data = {'train': [], 'val': [], 'test': []}

for cls in tqdm(classes, desc="Processing classes"):
    cls_dir = os.path.join(input_dir, cls)
    images = [f for f in os.listdir(cls_dir) if os.path.isfile(os.path.join(cls_dir, f))]

    train_val_split, test_split = train_test_split(images, test_size=split_ratios[2], random_state=42)
    train_split, val_split = train_test_split(
        train_val_split,
        test_size=split_ratios[1]/(split_ratios[0]+split_ratios[1]),
        random_state=42
    )

    # ------------------------------
    # Modified loop to resize & save
    # ------------------------------
    for split_name, split_images in zip(['train', 'val', 'test'], [train_split, val_split, test_split]):
        for img in tqdm(split_images, desc=f"{cls} - {split_name}", leave=False):
            src = os.path.join(cls_dir, img)
            dst = os.path.join(output_dir, split_name, cls, img)

            # Open, resize, and save
            with Image.open(src) as im:
                im = im.resize(IMG_SIZE, Image.LANCZOS)
                im.save(dst)

            csv_data[split_name].append({
                'category': cls,
                'indexInCategory': int(img.split('_')[-1].split('.')[0]),
                'filePath': f"split_data/{split_name}/{cls}/{img}"
            })

# ------------------------------
# Write CSV files
# ------------------------------
for split in ['train', 'val', 'test']:
    df = pd.DataFrame(csv_data[split])
    df.to_csv(os.path.join(output_dir, f"{split}.csv"), index=False)

print("✅ Data split complete, images resized, and CSV files created.")
