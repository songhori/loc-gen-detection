# loc-gen-detection

**loc-gen-detection** is a machine learning project developed for the 5th Iran's Face Cup Competition (FaceCup) and was awarded first place in this prestigious AI challenge.

This project detects specific tourism locations in Iran and counts the number of males and females in images. The dataset consists of 488 distinct locations, covering Iran’s natural and historical sites. It leverages transfer learning and computer vision techniques for both location classification and gender counting.

---

## Project Overview
This project consists of three main areas:

### 1. Web Scraping
- **Objective:** Collect underrepresented class images to enhance model performance.
- **Tools Used:**
  - **SerpAPI:** Searches and retrieves relevant images.
  - **OpenClip:** Matches and filters collected images.

### 2. Transfer Learning for Location Detection
- **Objective:** Classify images into one of 488 tourism locations in Iran.
- **Models Tested/Used:**
  - ResNet50
  - ResNet100
  - ResNet150
  - ViT-B/16
  - ViT-L/16
  - ViT-H/14

### 3. Transfer Learning for Gender Counting
- **Objective:** Detect and count the number of males and females in images.
- **Models Tested/Used:**
  - YOLO11x
  - ResNet50
  - ResNet100
  - ResNet150
  - EfficientNet-B7
  - ViT-B/16
  - ViT-L/16
  - DeepFace

---

## Test/Train Data

The dataset for training and testing can be found here:

📂 [Google Drive - Dataset](https://drive.google.com/drive/folders/1sUJHwB3t_WChlbrD3Wa69BMxfzzWuKdZ)


## Models

All models including yolo11x, trained location detection and trained gender detection models can be found here:

📂 [Google Drive - Models](https://drive.google.com/drive/folders/1pzC3gR1nFJwmXM9Hp6Na6cVwHt8Dh5Tf?usp=sharing)

---

## Setup & Installation

### Download the Dataset
Extract the dataset files from the [Google Drive link](https://drive.google.com/drive/folders/1sUJHwB3t_WChlbrD3Wa69BMxfzzWuKdZ) and place them in the appropriate directory.

### Setup CUDA for GPU Usage
To utilize GPU acceleration, follow these steps:

1. **Install CUDA Toolkit:**  
   Download and install the CUDA Toolkit from [here](https://developer.nvidia.com/cuda-12-6-3-download-archive?target_os=Windows&target_arch=x86_64&target_version=11&target_type=exe_local): `cuda_12.6.3_561.17_windows.exe`. (or any other compatible version with your O.S.)

2. **Install cuDNN:**
   download and extract the cuDNN from [here](https://developer.download.nvidia.com/compute/cudnn/redist/cudnn/windows-x86_64/)
   Copy and paste the contents of the `cudnn-windows-x86_64-9.6.0.74_cuda12-archive` folder into the installed CUDA directory:  
   `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6`.

4. **Verify Environment Variables:**  
   Ensure the following paths are added to your system's environment variables:
   ```PATH:
     Add: `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin`  
     Add: `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\lib\x64`
   ```
   **INCLUDE (if not already added):**  
   ```
   Add: `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\include`
   ```

5. **Install PyTorch and Dependencies:**  
   Run the following command in your activated environment to install PyTorch and related libraries:  
   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
   ```
   (cu124 stands for cuda version 12.4)


6. **Verify cuDNN Installation:**
   After installation, verify that your PyTorch or TensorFlow environment can detect cuDNN:

   Run the following Python code to check cuDNN availability:

   ```python
   import torch
   print("CUDA Available:", torch.cuda.is_available())
   print("cuDNN Version:", torch.backends.cudnn.version())
   ```
   If torch.cuda.is_available() returns True and torch.backends.cudnn.version() returns a version like 8700 (for cuDNN 8.7), the installation is successful.


   **Other method for installing 'torch with cuda' which is more straightforward for Windows:**
   download the [wheel source](https://download.pytorch.org/whl/cu124/torch-2.5.1%2Bcu124-cp310-cp310-win_amd64.whl)
   and then install the torch by source:

   ```bash
   pip install torch-2.5.1+cu124-cp310-cp310-win_amd64.whl
   ```

   then you can separately install torchvision, torchaudio, torchstudio by using:
   ```bash
   pip install torch torchvision torchaudio torchstudio --index-url https://download.pytorch.org/whl/cu124
   ```
   remember to install ultralytics after installing torch and torchvision
   you can find more pytorch cuda versions [here](https://pytorch.org/)


## Install other packages and requirements
After installing torch and torchvision with cuda, you can install other packages:
   ```bash
   pip install -r requirements.txt
   ```


## Working with the Project
1. **Identifying Underrepresented Location Classes**
   Script: find_folders_with_few_images.py
   Finds location classes with fewer images and extracts their names.
   These names can be used to scrape additional images later.

2. **Generate Location Class Embeddings**
   Script: get_image_embeddings.py
   Uses the CLIP model to compute image embeddings for location classes.

3. **Scraping Additional Location Images**
   Script: scrape_images.py
   Uses SerpAPI to scrape images from Google Search.
   Requires setting SERPAPI_API_KEY in environment variables.

4. **Detecting & Cropping Bodies (Gender Data Preparation)**
   Script: crop_bodies_yolo.py
   Uses YOLO model to detect and crop male and female bodies.
   Note: Male/female dataset is not included; you must add your own images.

5. **Creating Labeled Gender Data**
   Script: create_bodies_labels.py
   Labels the gender dataset and generates a CSV for training.

6. **Fine-Tuning the Location Detection Model**
   Script: fine_tune_location.py
   Trains a location classification model with augmentation, regularization, and optimization.

7. **Fine-Tuning the Gender Detection Model**
   Script: fine_tune_gender.py
   Trains the gender detection model to classify males and females.

8. **Running Inference on Test Data**
   Script: main.py
   Uses trained models for location detection and gender counting.
   Outputs a CSV file containing predictions for test images.
   Results and Output
   The final output is a CSV file with the following columns for each test image:
   | Column | Description                                      |
   |--------|--------------------------------------------------|
   | A      | Path to the input image                          |
   | B      | Number of males detected                         |
   | C      | Number of females detected                       |
   | D - RW | Probability of each location class (sum of all probabilities is 1) |

## Contributions
   Contributions are welcome! Feel free to submit pull requests or report issues.

## License
   This project is open-source under the License of the Author Mohammad-Mahdi-Songhori.
