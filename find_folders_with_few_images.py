import os


base_directory = "data/train2"
folder_range = 487
image_threshold = 5  # Set your threshold here

def count_images_in_folder(folder_path, image_extensions):
    count = 0
    for file_name in os.listdir(folder_path):
        if file_name.lower().endswith(image_extensions):
            count += 1
    return count

def find_folders_with_few_images(base_directory, folder_range, image_threshold):
    image_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff')
    folders_with_few_images = []

    for folder_number in range(folder_range + 1):  # Start from 0 to folder_range (inclusive)
        folder_path = os.path.join(base_directory, str(folder_number))
        if os.path.exists(folder_path) and os.path.isdir(folder_path):
            image_count = count_images_in_folder(folder_path, image_extensions)
            if image_count < image_threshold:
                folders_with_few_images.append(folder_number)

    return folders_with_few_images

folders_with_few_images = find_folders_with_few_images(base_directory, folder_range, image_threshold)
print("Folders with fewer than {} images:".format(image_threshold))
print(folders_with_few_images)
