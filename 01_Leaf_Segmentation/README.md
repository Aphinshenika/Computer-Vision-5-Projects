# Leaf Segmentation and Plant Growth Monitoring

## 1. Problem Statement

Plant diseases can affect crop health and productivity. Manual identification of diseased regions from leaf images can be time-consuming. Computer vision-based image segmentation can help identify and analyze disease-affected regions in leaf images.

This project implements a classical computer vision approach for segmenting disease regions from leaf images.

## 2. Objective

- Segment disease-affected regions from leaf images.
- Use HSV color-space analysis.
- Apply morphological operations to improve segmentation.
- Compare predicted masks with ground-truth masks.
- Evaluate performance using IoU, Dice Score, and Pixel Accuracy.
- Generate visual results for analysis.

## 3. Dataset

**Dataset:** Leaf Disease Segmentation Dataset

**Source:** Kaggle

The extracted augmented dataset used in this project contains:

- 2,940 leaf images
- 2,940 corresponding segmentation masks

Each image is paired with its corresponding mask using the same filename.

## 4. Methodology

### Step 1: Image Loading
Leaf images and corresponding masks are loaded using OpenCV.

### Step 2: Ground Truth Preparation
The segmentation mask is converted into a binary mask.

- White = target disease region
- Black = background

### Step 3: HSV Conversion
The input image is converted from BGR to HSV color space.

### Step 4: Disease Region Segmentation
HSV thresholding is used to identify regions with color characteristics associated with disease spots.

### Step 5: Morphological Processing
Morphological opening and closing operations are applied to reduce noise and improve the segmentation mask.

### Step 6: Performance Evaluation
The predicted mask is compared with the ground-truth mask using:

- Intersection over Union (IoU)
- Dice Score
- Pixel Accuracy

## 5. Tools and Technologies

- Python
- OpenCV
- NumPy
- Matplotlib
- Scikit-learn
- VS Code

## 6. Project Structure

```text
01_Leaf_Segmentation/
│
├── dataset/
│   ├── images/
│   └── masks/
│
├── results/
│   ├── evaluation_results.csv
│   └── sample_outputs/
│
├── screenshots/
│
├── src/
│   ├── main.py
│   └── save_samples.py
│
├── requirements.txt
└── README.md