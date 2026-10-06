import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ==============================
# PATHS
# ==============================
BASE_DIR = Path(__file__).resolve().parent.parent

IMAGE_DIR = BASE_DIR / "dataset" / "images"
MASK_DIR = BASE_DIR / "dataset" / "masks"
RESULTS_DIR = BASE_DIR / "results" / "sample_outputs"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ==============================
# SELECT 5 SAMPLE IMAGES
# ==============================
image_files = sorted(IMAGE_DIR.glob("*.jpg"))[:5]

for index, image_path in enumerate(image_files, start=1):

    mask_path = MASK_DIR / (image_path.stem + ".png")

    image = cv2.imread(str(image_path))
    ground_truth = cv2.imread(
        str(mask_path),
        cv2.IMREAD_GRAYSCALE
    )

    if image is None or ground_truth is None:
        continue

    # Ground truth
    ground_truth_binary = (ground_truth > 0).astype(np.uint8) * 255

    # HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    H = hsv[:, :, 0]
    S = hsv[:, :, 1]
    V = hsv[:, :, 2]

    # Disease segmentation
    disease_regions = (
        (H > 115) &
        (H < 179) &
        (S > 50) &
        (V < 230)
    )

    predicted = disease_regions.astype(np.uint8) * 255

    # Morphological processing
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

    # ==============================
    # OVERLAY
    # ==============================
    overlay = image.copy()

    overlay[predicted > 0] = [0, 255, 0]

    overlay = cv2.addWeighted(
        image,
        0.6,
        overlay,
        0.4,
        0
    )

    # ==============================
    # CREATE FIGURE
    # ==============================
    plt.figure(figsize=(16, 5))

    plt.subplot(1, 4, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1, 4, 2)
    plt.imshow(ground_truth_binary, cmap="gray")
    plt.title("Ground Truth Mask")
    plt.axis("off")

    plt.subplot(1, 4, 3)
    plt.imshow(predicted, cmap="gray")
    plt.title("Predicted Mask")
    plt.axis("off")

    plt.subplot(1, 4, 4)
    plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    plt.title("Prediction Overlay")
    plt.axis("off")

    plt.tight_layout()

    # ==============================
    # SAVE IMAGE
    # ==============================
    output_path = RESULTS_DIR / f"sample_{index}.png"

    plt.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Saved: {output_path}")

print()
print("================================")
print("5 SAMPLE RESULTS SAVED!")
print("================================")