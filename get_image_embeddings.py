import os
import numpy as np
import torch
from PIL import Image
import open_clip
from tqdm import tqdm

# Set up CLIP model from open_clip with CUDA and QuickGELU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading CLIP model on {device}...")
model, _, preprocess = open_clip.create_model_and_transforms('ViT-B/32-quickgelu', pretrained='openai')
model = model.to(device)

image_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp')

# Function to compute image embeddings
def get_image_embedding(image_path):
    image = Image.open(image_path).convert("RGB")
    image = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = model.encode_image(image)
    return embedding.cpu().numpy()

# Base folder containing all the folders
base_folder = "data/train2"
embedding_dict = {}

# Compute embeddings for all images in all folders
for class_id in tqdm(range(488), desc="Processing Classes"):
    folder_path = os.path.join(base_folder, str(class_id))
    embeddings = []
    for image_name in tqdm(os.listdir(folder_path), desc=f"Class {class_id}", leave=False):
        if image_name.lower().endswith(image_extensions):
            image_path = os.path.join(folder_path, image_name)
            embedding = get_image_embedding(image_path)
            embeddings.append(embedding)
    embedding_dict[class_id] = np.array(embeddings)

# Save the dictionary to a file
np.save("data/embeddings_train.npy", embedding_dict)
print("Saved embeddings dictionary to data/embeddings_train.npy")
