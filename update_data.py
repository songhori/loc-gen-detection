import pandas as pd
import os

# Paths
csv_file_path = "data/train2.csv"  # Path to your existing train.csv file
image_folder_path = "data/train2"  # Path to the root folder containing class folders

# Load the existing CSV
train_df = pd.read_csv(csv_file_path)

# Get the existing paths in the CSV as a set
existing_paths = set(train_df['path'].values)

# Collect new image paths
new_entries = []
existing_image_paths = set()

# Iterate over each class folder and its images
for class_folder in sorted(os.listdir(image_folder_path)):
    class_path = os.path.join(image_folder_path, class_folder)
    if os.path.isdir(class_path):
        for image_file in sorted(os.listdir(class_path)):
            image_path = f"{class_folder}/{image_file}"
            existing_image_paths.add(image_path)
            if image_path not in existing_paths:
                new_entries.append([int(class_folder), image_path, -1, -1])

# Create a DataFrame for the new entries
new_entries_df = pd.DataFrame(new_entries, columns=["class", "path", "male", "female"])

# Append new entries to the existing DataFrame
updated_train_df = pd.concat([train_df, new_entries_df], ignore_index=True)

# Remove rows with paths that don't exist in the images directory
updated_train_df = updated_train_df[updated_train_df['path'].isin(existing_image_paths)]

# Sort the DataFrame by the 'class' column
updated_train_df = updated_train_df.sort_values(by='class')

# Save the updated and sorted CSV
updated_csv_path = "data/train2.csv"
updated_train_df.to_csv(updated_csv_path, index=False)

print(f"Updated train2.csv saved as {updated_csv_path}")
