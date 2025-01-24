import os
import torch
from PIL import Image
from tqdm import tqdm
from ultralytics import YOLO

# Configurations
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
yolo_confidence = 0.6
body_model = 'models/yolo/yolo11x.pt'
yolo_model = YOLO(body_model)
yolo_model.to(DEVICE)

# Directories
input_dir = "data/train2"
output_dir = "data/bodies"
os.makedirs(output_dir, exist_ok=True)

images = sorted(os.listdir(input_dir))

# Function to crop and save images
def crop_and_save_images(images, input_dir, output_dir):
    for idx, path in enumerate(tqdm(images, total=len(images))):
        image_path = os.path.join(input_dir, path)
        image = Image.open(image_path).convert("RGB")

        # Run YOLO model for body detection
        detections = yolo_model.predict(image, verbose=False, conf=yolo_confidence, classes=[0], device=DEVICE)

        # Check if any detections are present
        if detections[0].boxes.xyxy.numel() == 0:
            continue

        # Crop and save each detected region
        for i, bb in enumerate(detections[0].boxes.xyxy):
            bb = bb.cpu().numpy().astype(int)
            cropped_img = image.crop([bb[0], bb[1], bb[2], bb[3]])

            # Save cropped image
            save_path = os.path.join(output_dir, f"{idx}_{i}.jpg")
            cropped_img.save(save_path)

# Crop and save images
crop_and_save_images(images, input_dir, output_dir)

print("Cropping complete! Cropped images saved to:", output_dir)
