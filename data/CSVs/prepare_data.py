import os
import csv
from pathlib import Path

# Define your directories and labels
dir_label_pairs = [
    (r'F:\data\images', 'images'),
    (r'F:\data\labels', 'labels')
]

# Output CSV file
csv_filename = Path(__file__).resolve().parent / 'dataset.csv'

# Collect image paths and labels
images = []
labels = []

for dir_path, label in dir_label_pairs:
    for root, _, files in os.walk(dir_path):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp')):
                img_path = os.path.join(root, file)
                images.append(img_path)
            elif file.lower().endswith('.txt'):
                label_path = os.path.join(root, file)
                labels.append(label_path)

# Write to CSV
rows = list(zip(images, labels))
with open(csv_filename, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['images', 'labels'])  # Header
    writer.writerows(rows)


print(f"CSV file '{csv_filename}' created with {len(rows)} entries.")
