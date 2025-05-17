import cv2
import numpy as np
import matplotlib.pyplot as plt

def detect_fire_in_image(image_path):
    # Load image
    image = cv2.imread(fire_fig_1)
    original = image.copy()

    # Convert to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Fire-like HSV range
    lower = np.array([0, 50, 50])
    upper = np.array([35, 255, 255])

    # Create mask and apply
    mask = cv2.inRange(hsv, lower, upper)
    fire_pixels = cv2.countNonZero(mask)
    total_pixels = mask.shape[0] * mask.shape[1]
    fire_ratio = fire_pixels / total_pixels

    # Label based on threshold
    if fire_ratio > 0.01:
        label = "🔥 Fire Detected"
        color = (0, 0, 255)
    else:
        label = "✅ No Fire Detected"
        color = (0, 255, 0)

    # Draw label
    cv2.putText(original, label, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    # Display result
    image_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(8, 6))
    plt.imshow(image_rgb)
    plt.axis('off')
    plt.title(label)
    plt.show()

# Example usage
detect_fire_in_image("fire_image.jpg")  # Replace with your image file path
