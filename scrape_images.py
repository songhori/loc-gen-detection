import os
import requests
import time
import pandas as pd
from serpapi import GoogleSearch
from PIL import Image
from io import BytesIO
import numpy as np
import torch
import open_clip
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import urllib3

# Disable SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Setup retry mechanism
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


num_max_images = 40
num_min_images = 10


session = requests.Session()
retries = Retry(total=2, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
session.mount("http://", HTTPAdapter(max_retries=retries))
session.mount("https://", HTTPAdapter(max_retries=retries))

# Set up CLIP model from open_clip with CUDA and QuickGELU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading CLIP model on {device}...")
model, _, preprocess = open_clip.create_model_and_transforms('ViT-B/32-quickgelu', pretrained='openai')
model = model.to(device)

# Function to compute image embeddings
def get_image_embedding(image_path):
    image = Image.open(image_path).convert("RGB")
    image = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = model.encode_image(image).cpu().numpy().squeeze()
        embedding = embedding.reshape(1, -1)  # Ensure 2D shape
    return embedding

def download_images(query, save_folder, num_old_images, num_new_images, num_max_images=50):
    num_all_images = num_old_images + num_new_images

    if not os.path.exists(save_folder):
        os.makedirs(save_folder, exist_ok=True)
    params = {
        "engine": "google",
        "q": query,
        "tbm": "isch",
        "num": 100,
        "api_key": os.getenv('SERPAPI_API_KEY'),
        "start": num_new_images
    }

    search = GoogleSearch(params)
    results = search.get_dict()

    if 'images_results' not in results:
        print("Key 'images_results' not found in the response.")
        return []

    downloaded_images = []
    while len(downloaded_images) + num_all_images < num_max_images and results['images_results']:
        for img_info in results['images_results']:
            if len(downloaded_images) + num_all_images >= num_max_images:
                break
            try:
                img_url = img_info.get('original')
                if not img_url:
                    continue
                img_data = session.get(img_url, verify=False).content  # Disable SSL verification here
                img_name = f"image_{len(downloaded_images) + num_new_images}.jpg"
                img_path = os.path.join(save_folder, img_name)

                img_pil = Image.open(BytesIO(img_data))
                if img_pil.size[0] >= 300 and img_pil.size[1] >= 300:
                    img_pil.save(img_path)
                    downloaded_images.append(img_path)
                    print(f"Downloaded: {img_path}")
                else:
                    print(f"Skipped low-resolution image: {img_url}")
            except requests.exceptions.RequestException as req_err:
                print(f"Request error: {req_err}")
            except Exception as e:
                print(f"Failed to download image: {e}")

        # If fewer images than needed, attempt to get more results
        if len(downloaded_images) + num_all_images < num_max_images:
            params["start"] = len(results['images_results'])
            search = GoogleSearch(params)
            results = search.get_dict()

    print(f"Total downloaded images: {len(downloaded_images)}")
    return downloaded_images

# Load the precomputed embeddings dictionary
embedding_dict = np.load("data/embeddings_train.npy", allow_pickle=True).item()

# Step 2: Read the CSV file
csv_path = "data/class_info.csv"
df = pd.read_csv(csv_path)

# Base folder containing all the folders
base_folder = "data/train2"

# Step 3: Iterate through each row in the CSV
for index, row in tqdm(df.iloc[1:].iterrows(), total=df.shape[0]-1, desc="Processing Rows"):
    class_id = row["class"]
    location = row["location"]
    if pd.isna(location):
        continue

    folder_path = os.path.join(base_folder, str(class_id))
    new_folder_path = os.path.join(folder_path, "downloaded_images")

    num_old_images = len([item for item in os.listdir(folder_path) if '.' in item])

    if num_min_images <= num_old_images:
        continue

    if not os.path.exists(new_folder_path):
        os.makedirs(new_folder_path)

    num_new_images = len([item for item in os.listdir(new_folder_path) if '.' in item])

    if num_old_images + num_new_images >= num_max_images:
        continue

    print(f"Processing Class {class_id}: {location}")

    # Download images from Google search
    downloaded_images = download_images(location, new_folder_path, num_old_images=num_old_images, num_new_images=num_new_images, num_max_images=num_max_images)

    # Step 4: Filter downloaded images using CLIP
    if downloaded_images:
        folder_embeddings = np.array(embedding_dict[class_id])

        # Ensure folder_embeddings is 2D
        if folder_embeddings.ndim == 1:
            folder_embeddings = folder_embeddings.reshape(1, -1)

        for img_path in tqdm(downloaded_images, desc=f"Filtering images for Class {class_id}", leave=False):
            try:
                embedding = get_image_embedding(img_path)
                
                # Ensure embedding is 2D
                if embedding.ndim == 1:
                    embedding = embedding.reshape(1, -1)
                elif embedding.ndim == 3:
                    embedding = embedding.reshape(embedding.shape[0], -1)
                
                max_similarity = np.max(cosine_similarity(embedding, folder_embeddings))

                if max_similarity < 0.7:
                    os.remove(img_path)
                    print(f"Removed low-similarity image: {img_path} (similarity: {max_similarity})")

                if .99 < max_similarity <= 1:
                    os.remove(img_path)
                    print(f"Removed matching image: {img_path} (similarity: {max_similarity})")

            except Exception as e:
                print(f"Failed to compute embedding for {img_path}: {e}")

    # Add a delay to avoid being blocked by Google
    time.sleep(5)
