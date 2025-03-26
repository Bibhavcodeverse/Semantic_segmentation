# Semantic_segmentation

# Semantic Segmentation using DeepLabV3

## Overview
This project implements **semantic segmentation** using the **DeepLabV3 model with a ResNet-101 backbone** from PyTorch's `torchvision` library. It processes an input image and generates a segmentation mask.

## Features
- Loads a **pre-trained DeepLabV3 model** for segmentation.
- Preprocesses images (resizing, normalization, and tensor conversion).
- Performs segmentation and generates a class mask.
- Visualizes the **original image and the segmented mask**.

## Dependencies
Ensure you have the following libraries installed:

```bash
pip install torch torchvision opencv-python numpy matplotlib



```

![Screenshot 2025-03-27 001733](https://github.com/user-attachments/assets/b8c6cb44-37a9-407c-8629-83e12e2efb96)


## How to Run
1. **Load the model**: The script initializes the DeepLabV3 model with pre-trained weights.
2. **Preprocess an image**: Resizes and normalizes the input.
3. **Perform segmentation**: Runs the model to generate a segmented mask.
4. **Display the results**: Shows both the original image and its segmentation mask.

### Example Usage
```python
from torchvision import models
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

# Load Model
model = models.segmentation.deeplabv3_resnet101(pretrained=True)
model.eval()

# Preprocess Image
def preprocess_image(image_path, target_size=(512, 512)):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, target_size)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

# Perform Segmentation
def segment_image(image_path):
    image = preprocess_image(image_path)
    with torch.no_grad():
        output = model(image)['out']
    return torch.argmax(output.squeeze(), dim=0).numpy()

# Load and segment an image
image_path = "path/to/image.jpg"  # Provide the actual image path
segmented_mask = segment_image(image_path)

# Display Results
original_image = cv2.imread(image_path)
original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(original_image)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(segmented_mask, cmap="jet")
plt.title("Segmented Mask")
plt.axis("off")

plt.show()
```

## Expected Output
The script will display the original image and the corresponding **segmented mask**.

## Notes
- Ensure that the image path is correctly set before running the script.
- The model segments objects based on COCO dataset classes.

## License
This project is open-source and available under the MIT License.

