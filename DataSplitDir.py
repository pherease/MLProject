import os
import shutil
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# ------------------------------
# PARAMETERS
# ------------------------------
input_dir = './MLProject/data'
output_dir = './MLProject/split_data'
split_ratios = (0.4, 0.3, 0.3)  # train, val, test

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
# Split data
# ------------------------------
clear_output_dir(output_dir)

classes = os.listdir(input_dir)
classes = [c for c in classes if os.path.isdir(os.path.join(input_dir, c))]
print(f"Found classes: {classes}")

make_dirs(output_dir, classes)

for cls in tqdm(classes, desc="Processing classes"):
    cls_dir = os.path.join(input_dir, cls)
    images = [f for f in os.listdir(cls_dir) if os.path.isfile(os.path.join(cls_dir, f))]
    
    train_val_split, test_split = train_test_split(images, test_size=split_ratios[2], random_state=42)
    train_split, val_split = train_test_split(
        train_val_split, 
        test_size=split_ratios[1]/(split_ratios[0]+split_ratios[1]), 
        random_state=42
    )

    # Use progress bar for images
    for img in tqdm(train_split, desc=f"{cls} - train", leave=False):
        shutil.copy2(os.path.join(cls_dir, img), os.path.join(output_dir, 'train', cls, img))
    
    for img in tqdm(val_split, desc=f"{cls} - val", leave=False):
        shutil.copy2(os.path.join(cls_dir, img), os.path.join(output_dir, 'val', cls, img))
    
    for img in tqdm(test_split, desc=f"{cls} - test", leave=False):
        shutil.copy2(os.path.join(cls_dir, img), os.path.join(output_dir, 'test', cls, img))

print("✅ Data successfully split into train/val/test folders.")
