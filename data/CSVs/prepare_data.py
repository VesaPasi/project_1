import os
import csv

# Define your directories and labels
dir_label_pairs = [
    ('D:\SAMK - Machine vision and CNN\project_1\data\Xrays', 'xrays'),
    ('D:\SAMK - Machine vision and CNN\project_1\data\Masks', 'masks')
]

# Output CSV file
csv_filename = 'image_paths.csv'

# Collect image paths and labels
rows = []
for dir_path, label in dir_label_pairs:
    for root, _, files in os.walk(dir_path):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp')):
                full_path = os.path.join(root, file)
                rows.append([full_path, label])

# Write to CSV
with open(csv_filename, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['image_path', 'label'])  # Header
    writer.writerows(rows)

print(f"CSV file '{csv_filename}' created with {len(rows)} entries.")
