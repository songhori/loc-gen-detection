import os
import pandas as pd

# Directories
male_dir = "data/bodies/male"
female_dir = "data/bodies/female"
output_csv = "data/bodies/bodies_labels.csv"

# Collect data
rows = []

# Process male images
for filename in os.listdir(male_dir):
    if filename.endswith(('.jpg', '.png', '.jpeg')):  # Add other extensions if needed
        path = os.path.join('male', filename).replace('\\', '/')  # Ensure Linux-style path
        rows.append([0, path])  # Class 0 for male

# Process female images
for filename in os.listdir(female_dir):
    if filename.endswith(('.jpg', '.png', '.jpeg')):  # Add other extensions if needed
        path = os.path.join('female', filename).replace('\\', '/')  # Ensure Linux-style path
        rows.append([1, path])  # Class 1 for female

# Create DataFrame
columns = ["class", "path"]
df = pd.DataFrame(rows, columns=columns)

# Save to CSV
df.to_csv(output_csv, index=False)

print(f"CSV saved to {output_csv}")
