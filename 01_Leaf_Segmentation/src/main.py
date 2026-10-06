import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import csv

# ==========================================
# PATHS
# ==========================================
BASE_DIR = Path(__file__).resolve().parent.parent

IMAGE_DIR = BASE_DIR / "dataset" / "images"
MASK_DIR = BASE_DIR / "dataset" / "masks"
RESULTS_DIR = BASE_DIR / "results"

RESULTS_DIR.mkdir(exist_ok=True)

# ==========================================
# GET IMAGE FILES
# ==========================================
image_files = sorted(IMAGE_DIR.glob("*.jpg"))

print("==========================================")
print("LEAF SEGMENTATION - FULL DATASET")
print("==========================================")
print(f"Images found: {len(image_files)}")

# ==========================================
# METRIC STORAGE
# ==========================================
iou_scores = []
dice_scores = []
accuracy_scores = []

processed = 0
skipped = 0

# ==========================================
# PROCESS ALL IMAGES
# ==========================================
for image_path in image_files:

    mask_path = MASK_DIR / (image_path.stem + ".png")

    if not mask_path.exists():
        skipped += 1
        continue

    image = cv2.imread(str(image_path))
    ground_truth = cv2.imread(
        str(mask_path),
        cv2.IMREAD_GRAYSCALE
    )

    if image is None or ground_truth is None:
        skipped += 1
        continue

    # ======================================
    # GROUND TRUTH
    # ======================================
    ground_truth_binary = ground_truth > 0

    # ======================================
    # HSV CONVERSION
    # ======================================
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    H = hsv[:, :, 0]
    S = hsv[:, :, 1]
    V = hsv[:, :, 2]

    # ======================================
    # DISEASE REGION SEGMENTATION
    # ======================================
    disease_regions = (
        (H > 115) &
        (H < 179) &
        (S > 50) &
        (V < 230)
    )

    predicted = disease_regions.astype(np.uint8) * 255

    # ======================================
    # MORPHOLOGICAL PROCESSING
    # ======================================
    kernel = np.ones((3, 3), np.uint8)

    predicted = cv2.morphologyEx(
        predicted,
        cv2.MORPH_OPEN,
        kernel
    )

    predicted = cv2.morphologyEx(
        predicted,
        cv2.MORPH_CLOSE,
        kernel
    )

    predicted_binary = predicted > 0

    # ======================================
    # IoU
    # ======================================
    intersection = np.logical_and(
        ground_truth_binary,
        predicted_binary
    ).sum()

    union = np.logical_or(
        ground_truth_binary,
        predicted_binary
    ).sum()

    iou = intersection / union if union != 0 else 0

    # ======================================
    # DICE
    # ======================================
    ground_pixels = ground_truth_binary.sum()
    predicted_pixels = predicted_binary.sum()

    dice = (
        2 * intersection /
        (ground_pixels + predicted_pixels)
        if (ground_pixels + predicted_pixels) != 0
        else 0
    )

    # ======================================
    # PIXEL ACCURACY
    # ======================================
    correct_pixels = np.sum(
        ground_truth_binary == predicted_binary
    )

    total_pixels = ground_truth_binary.size

    accuracy = correct_pixels / total_pixels

    # ======================================
    # STORE RESULTS
    # ======================================
    iou_scores.append(iou)
    dice_scores.append(dice)
    accuracy_scores.append(accuracy)

    processed += 1

    # ======================================
    # PROGRESS
    # ======================================
    if processed % 100 == 0:
        print(f"Processed: {processed}/{len(image_files)}")

# ==========================================
# FINAL RESULTS
# ==========================================
mean_iou = np.mean(iou_scores)
mean_dice = np.mean(dice_scores)
mean_accuracy = np.mean(accuracy_scores)

print()
print("==========================================")
print("FINAL DATASET RESULTS")
print("==========================================")

print(f"Processed Images : {processed}")
print(f"Skipped Images   : {skipped}")

print(f"Mean IoU         : {mean_iou:.4f}")
print(f"Mean IoU (%)     : {mean_iou * 100:.2f}%")

print(f"Mean Dice        : {mean_dice:.4f}")
print(f"Mean Dice (%)    : {mean_dice * 100:.2f}%")

print(f"Mean Accuracy    : {mean_accuracy:.4f}")
print(f"Mean Accuracy (%) : {mean_accuracy * 100:.2f}%")

print("==========================================")

# ==========================================
# SAVE METRICS
# ==========================================
csv_path = RESULTS_DIR / "evaluation_results.csv"

with open(csv_path, "w", newline="") as file:

    writer = csv.writer(file)

    writer.writerow([
        "Metric",
        "Score",
        "Percentage"
    ])

    writer.writerow([
        "Mean IoU",
        f"{mean_iou:.4f}",
        f"{mean_iou * 100:.2f}%"
    ])

    writer.writerow([
        "Mean Dice",
        f"{mean_dice:.4f}",
        f"{mean_dice * 100:.2f}%"
    ])

    writer.writerow([
        "Mean Pixel Accuracy",
        f"{mean_accuracy:.4f}",
        f"{mean_accuracy * 100:.2f}%"
    ])

print()
print("Results saved to:")
print(csv_path)