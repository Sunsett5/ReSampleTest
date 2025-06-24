import os
import shutil
import random

# === Settings ===
source_folder = '../samples/ffhq256'
destination_folder = './data/samples'

# === Step 1: Sample 100 unique numbers ===
num_samples = 20
selected_numbers = random.sample(range(69000, 70000), num_samples)

# === Step 2: Create destination folder if it doesn't exist ===
os.makedirs(destination_folder, exist_ok=True)

# === Step 3: Copy corresponding PNGs ===
for number in selected_numbers:
    filename = f"{number}.png"
    src_path = os.path.join(source_folder, filename)
    dst_path = os.path.join(destination_folder, filename)

    if os.path.exists(src_path):
        shutil.copy(src_path, dst_path)
    else:
        print(f"Warning: {filename} not found in source folder.")

print(f"{num_samples} files copied.")

