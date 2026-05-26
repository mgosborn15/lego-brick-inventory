# test_knn_detection.py
import os
import cv2
import numpy as np
import joblib
from datetime import datetime

# Configuration
MODEL_PATH = 'models/knn_model.pkl'
METADATA_PATH = 'models/knn_metadata.pkl'
TEST_IMAGES_DIR = 'test_images'  # You'll put test photos here
RESULTS_DIR = 'results/detection_tests'
IMG_SIZE = (64, 64)

# Create directories
os.makedirs(TEST_IMAGES_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


def load_model():
    """
    Load trained KNN model and metadata
    """
    print("=" * 60)
    print("Loading KNN Model")
    print("=" * 60)

    knn = joblib.load(MODEL_PATH)
    metadata = joblib.load(METADATA_PATH)

    print(f"\n✓ Model loaded: {metadata['model_type']}")
    print(f"✓ Classes: {metadata['num_classes']}")
    print(f"✓ k neighbors: {metadata['k_neighbors']}")
    print(f"✓ Timestamp: {metadata['timestamp']}")
    print(f"{'=' * 60}\n")

    return knn, metadata


def detect_bricks(image, min_area=1000, max_area=500000):
    """
    Detect LEGO bricks using Canny edge detection + morphological operations
    Based on research showing this performs best for LEGO detection

    Args:
        image: Input image (BGR)
        min_area: Minimum contour area
        max_area: Maximum contour area

    Returns:
        List of bounding boxes [(x, y, w, h), ...]
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)

    # Canny edge detection
    # Parameters: lower threshold, upper threshold
    edges = cv2.Canny(blurred, 30, 100)

    # Save edge image for debugging
    cv2.imwrite('debug_edges.jpg', edges)

    # Morphological operations to close gaps in edges
    # This connects nearby edges to form complete brick outlines
    kernel = np.ones((7, 7), np.uint8)

    # Dilate to connect nearby edges
    dilated = cv2.dilate(edges, kernel, iterations=2)

    # Close to fill small gaps
    closed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel, iterations=3)

    # Save morphology result for debugging
    cv2.imwrite('debug_morphology.jpg', closed)

    # Find contours on the closed edge image
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    bounding_boxes = []

    print(f"  Found {len(contours)} contours")

    for contour in contours:
        area = cv2.contourArea(contour)

        if area < min_area or area > max_area:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / float(h) if h > 0 else 0

        # Print for debugging
        print(f"    Contour: area={area:.0f}, size={w}x{h}, ratio={aspect_ratio:.2f}", end="")

        # Filter by size and aspect ratio
        if w > 30 and h > 30 and 0.1 < aspect_ratio < 10:
            bounding_boxes.append((x, y, w, h))
            print(" ✓ KEPT")
        else:
            print(" ✗ rejected")

    print(f"  Kept {len(bounding_boxes)} bricks after filtering\n")

    return bounding_boxes


def classify_brick(brick_image, knn_model, img_size=(64, 64)):
    """
    Classify a single brick image using KNN

    Args:
        brick_image: Cropped brick image
        knn_model: Trained KNN model
        img_size: Size to resize to

    Returns:
        (predicted_class_idx, confidence)
    """
    # Resize
    brick_resized = cv2.resize(brick_image, img_size)

    # Flatten and normalize
    brick_flat = brick_resized.reshape(1, -1).astype(np.float32) / 255.0

    # Predict
    prediction = knn_model.predict(brick_flat)[0]

    # Get probabilities (proportion of k neighbors voting for this class)
    probabilities = knn_model.predict_proba(brick_flat)[0]
    confidence = probabilities[prediction]

    return prediction, confidence


def process_image(image_path, knn_model, class_names,
                  confidence_threshold=0.4, save_result=True):
    """
    Process a single image: detect bricks and classify them

    Args:
        image_path: Path to test image
        knn_model: Trained KNN model
        class_names: List of brick type names
        confidence_threshold: Minimum confidence to display (0-1)
        save_result: Whether to save annotated image

    Returns:
        Dictionary with detection results
    """
    print(f"\nProcessing: {os.path.basename(image_path)}")

    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"  ✗ Could not load image")
        return None

    original = image.copy()

    # Detect bricks
    boxes = detect_bricks(image)
    print(f"  Detected {len(boxes)} potential bricks")

    # Classify each detected brick
    results = []

    for i, (x, y, w, h) in enumerate(boxes):
        # Extract brick region
        brick_crop = image[y:y + h, x:x + w]

        # Classify
        pred_idx, confidence = classify_brick(brick_crop, knn_model, IMG_SIZE)
        pred_class = class_names[pred_idx]

        results.append({
            'brick_num': i + 1,
            'bbox': (x, y, w, h),
            'predicted_class': pred_class,
            'confidence': confidence
        })

        # Determine box color based on confidence
        if confidence >= 0.6:
            color = (0, 255, 0)  # GREEN - high confidence
            label = f"#{i + 1}: {pred_class} ({confidence:.0%})"
        elif confidence >= confidence_threshold:
            color = (0, 165, 255)  # ORANGE - medium confidence
            label = f"#{i + 1}: {pred_class} ({confidence:.0%})"
        else:
            color = (0, 0, 255)  # RED - low confidence
            label = f"#{i + 1}: ? ({confidence:.0%})"

        # Draw bounding box (thicker line)
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 3)

        # Draw label background
        font_scale = 1
        font_thickness = 2

        (label_w, label_h), _ = cv2.getTextSize(label,
                                                cv2.FONT_HERSHEY_SIMPLEX,
                                                font_scale, font_thickness)
        cv2.rectangle(image, (x, y - label_h - 15),
                      (x + label_w + 10, y), color, -1)

        # Draw label text
        cv2.putText(image, label, (x + 5, y - 8),  # Slight offset
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                    (255, 255, 255), font_thickness)

        print(f"  Brick #{i + 1}: {pred_class} ({confidence:.1%} confidence)")

    # Add summary text to image
    summary = f"KNN Detection: {len(boxes)} bricks found"
    cv2.putText(image, summary, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(image, summary, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)

    # Save result
    if save_result:
        output_name = f"detected_{os.path.basename(image_path)}"
        output_path = os.path.join(RESULTS_DIR, output_name)
        cv2.imwrite(output_path, image)
        print(f"  ✓ Saved to: {output_path}")

    return {
        'image_path': image_path,
        'num_detected': len(boxes),
        'results': results,
        'output_image': image
    }


def process_directory(test_dir, knn_model, class_names):
    """
    Process all images in a directory
    """
    print("=" * 60)
    print("Processing Test Images")
    print("=" * 60)

    # Get all image files
    image_files = [f for f in os.listdir(test_dir)
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))
                   and not f.startswith('.')]

    if not image_files:
        print(f"\n⚠️  No images found in {test_dir}/")
        print(f"Please add test images (photos of LEGO bricks) to this folder.")
        return

    print(f"\nFound {len(image_files)} test images\n")

    all_results = []

    for img_file in image_files:
        img_path = os.path.join(test_dir, img_file)
        result = process_image(img_path, knn_model, class_names)

        if result:
            all_results.append(result)

    # Summary
    print("\n" + "=" * 60)
    print("Detection Summary")
    print("=" * 60)

    total_detected = sum(r['num_detected'] for r in all_results)
    avg_confidence = []

    for result in all_results:
        for brick in result['results']:
            avg_confidence.append(brick['confidence'])

    if avg_confidence:
        print(f"\nTotal images processed: {len(all_results)}")
        print(f"Total bricks detected: {total_detected}")
        print(f"Average detection confidence: {np.mean(avg_confidence):.1%}")
        print(f"High confidence (>60%): {sum(1 for c in avg_confidence if c > 0.6)} bricks")
        print(f"Medium confidence (40-60%): {sum(1 for c in avg_confidence if 0.4 <= c <= 0.6)} bricks")
        print(f"Low confidence (<40%): {sum(1 for c in avg_confidence if c < 0.4)} bricks")

    print(f"\n✓ Results saved to: {RESULTS_DIR}/")
    print("=" * 60 + "\n")


def create_sample_instructions():
    """
    Print instructions for creating test images
    """
    print("=" * 60)
    print("HOW TO CREATE TEST IMAGES")
    print("=" * 60)
    print("""
1. Setup:
   • Use a WHITE background (paper, desk, etc.)
   • Good lighting (avoid harsh shadows)
   • Place 3-10 LEGO bricks spread out
   • Don't overlap bricks too much

2. Take photos:
   • Use your phone camera
   • Shoot from above (overhead view works best)
   • Keep bricks in focus
   • Save as .jpg or .png

3. Add to project:
   • Copy photos to: test_images/
   • Name them: test1.jpg, test2.jpg, etc.

4. Run this script:
   • python3 test_knn_detection.py
   • Check results in: results/detection_tests/

Example setup:
   [White paper on desk]
   [5-6 LEGO bricks spread out]
   [Phone camera overhead]
   [Snap photo]

""")


def main():
    """
    Main detection testing pipeline
    """
    print("\n" + "=" * 60)
    print("🧱 KNN LEGO BRICK DETECTION TESTER")
    print("=" * 60 + "\n")

    # Load model
    knn, metadata = load_model()
    class_names = metadata['class_names']

    # Check if test images exist
    if not os.path.exists(TEST_IMAGES_DIR) or \
            not os.listdir(TEST_IMAGES_DIR):
        print(f"⚠️  No test images found in '{TEST_IMAGES_DIR}/'")
        create_sample_instructions()

        print("=" * 60)
        print("NEXT STEPS:")
        print("=" * 60)
        print("1. Create test_images/ folder (done automatically)")
        print("2. Add photos of LEGO bricks to that folder")
        print("3. Run this script again")
        print("=" * 60 + "\n")
        return

    # Process all test images
    process_directory(TEST_IMAGES_DIR, knn, class_names)

    print("=" * 60)
    print("DONE!")
    print("=" * 60)
    print(f"Check annotated images in: {RESULTS_DIR}/")
    print("=" * 60 + "\n")


if __name__ == '__main__':
    main()