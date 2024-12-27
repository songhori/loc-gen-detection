# loc-detection
Detects Iran's specific historic/nature location, along with the number of males and females in the picture.
The train/test images are choosen from 488 different locations of Iran's nature and historic places.

This project uses 3 pre-trained Neural Network models.
1- yolo11x from ultralytics for body detection
2- fine-tuned efficientnet_b0 from torchvision for gender classification
3- fine-tuned resnet50 from torchvision for location detection

The output is a csv containing the following columns for each test image:
A: The path to the input image
B: Number of males
C: number of females
D-RW: The probability of each location class (sum of all probabilities is 1)
