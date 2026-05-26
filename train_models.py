import os
import time
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from datetime import datetime

# Configuration
DATASET_PATH = 'brick-images-subset'
MODELS_DIR = 'models'
RESULTS_DIR = 'results'
IMG_SIZE = (64, 64)  # Resize all images to 64x64
TEST_SIZE = 0.2  # 80% train, 20% test
RANDOM_STATE = 42  # For reproducibility
K_NEIGHBORS = 3  # Number of neighbors for KNN

# Create directories
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(os.path.join(RESULTS_DIR, 'confusion_matrices'), exist_ok=True)


def load_dataset(max_images_per_class=100, use_renders=True, use_photos=True):
    """
    Load images from brick-images-subset

    Args:
        max_images_per_class: Limit images per class for faster training
        use_renders: Include rendered images
        use_photos: Include real photos

    Returns:
        X: numpy array of images
        y: numpy array of labels
        class_names: list of brick IDs
    """
    print("=" * 60)
    print("Loading Dataset")
    print("=" * 60)

    X = []
    y = []
    class_names = []

    # Determine which splits to use
    splits = []
    if use_photos:
        splits.append('photos')
    if use_renders:
        splits.append('renders')

    print(f"\nUsing: {', '.join(splits)}")
    print(f"Image size: {IMG_SIZE}")
    print(f"Max images per class: {max_images_per_class}\n")

    # Get brick types (class names)
    first_split = os.path.join(DATASET_PATH, splits[0])
    brick_types = sorted([d for d in os.listdir(first_split)
                          if os.path.isdir(os.path.join(first_split, d))
                          and not d.startswith('.')])

    class_names = brick_types
    print(f"Found {len(class_names)} brick types\n")

    # Load images for each class
    for class_idx, brick_id in enumerate(class_names):
        print(f"Loading {brick_id} ({class_idx + 1}/{len(class_names)})...", end=' ')

        class_images = []

        for split in splits:
            brick_path = os.path.join(DATASET_PATH, split, brick_id)

            if not os.path.exists(brick_path):
                continue

            # Get all image files
            images = [f for f in os.listdir(brick_path)
                      if f.endswith(('.jpg', '.png', '.jpeg'))
                      and not f.startswith('.')]

            # Load images
            for img_name in images:
                img_path = os.path.join(brick_path, img_name)

                # Read and resize image
                img = cv2.imread(img_path)
                if img is None:
                    continue

                img = cv2.resize(img, IMG_SIZE)
                class_images.append(img)

        # Limit images per class
        if len(class_images) > max_images_per_class:
            import random
            random.seed(RANDOM_STATE)
            class_images = random.sample(class_images, max_images_per_class)

        # Add to dataset
        X.extend(class_images)
        y.extend([class_idx] * len(class_images))

        print(f"{len(class_images)} images")

    # Convert to numpy arrays
    X = np.array(X)
    y = np.array(y)

    print(f"\n{'=' * 60}")
    print(f"Dataset loaded successfully!")
    print(f"Total images: {len(X):,}")
    print(f"Image shape: {X.shape}")
    print(f"Number of classes: {len(class_names)}")
    print(f"{'=' * 60}\n")

    return X, y, class_names


def train_knn(X_train, y_train, X_test, y_test, k=5):
    """
    Train K-Nearest Neighbors classifier
    """
    print("=" * 60)
    print(f"Training KNN (k={k})")
    print("=" * 60)

    # Flatten images for KNN (converts 64x64x3 to 12288 features)
    print("\nFlattening images...")
    X_train_flat = X_train.reshape(len(X_train), -1).astype(np.float32) / 255.0
    X_test_flat = X_test.reshape(len(X_test), -1).astype(np.float32) / 255.0

    print(f"Training data shape: {X_train_flat.shape}")
    print(f"Test data shape: {X_test_flat.shape}")

    # Train
    print(f"\nTraining KNN model with {k} neighbors...")
    print("This will take a minute or two...\n")

    start_time = time.time()

    knn = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)  # Use all CPU cores
    knn.fit(X_train_flat, y_train)

    train_time = time.time() - start_time

    # Evaluate on test set
    print("Evaluating on test set...")
    y_pred = knn.predict(X_test_flat)
    accuracy = accuracy_score(y_test, y_pred)

    # Evaluate on training set (to check for overfitting)
    print("Evaluating on training set...")
    y_train_pred = knn.predict(X_train_flat)
    train_accuracy = accuracy_score(y_train, y_train_pred)

    print(f"\n{'=' * 60}")
    print(f"KNN Results:")
    print(f"  Training time: {train_time:.2f}s")
    print(f"  Train accuracy: {train_accuracy:.4f} ({train_accuracy * 100:.2f}%)")
    print(f"  Test accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")

    # Check for overfitting
    if train_accuracy - accuracy > 0.1:
        print(f"  ⚠️  Warning: Possible overfitting (train-test gap: {(train_accuracy - accuracy) * 100:.1f}%)")
    else:
        print(f"  ✓ Good generalization (train-test gap: {(train_accuracy - accuracy) * 100:.1f}%)")

    print(f"{'=' * 60}\n")

    return knn, accuracy, y_pred, train_time


def plot_confusion_matrix(y_true, y_pred, class_names):
    """
    Plot and save confusion matrix
    """
    print("Generating confusion matrix...")

    cm = confusion_matrix(y_true, y_pred)

    # Create figure
    plt.figure(figsize=(20, 16))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names,
                cbar_kws={'label': 'Count'})
    plt.title(f'KNN Confusion Matrix (k={K_NEIGHBORS})', fontsize=16, pad=20)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    # Save
    save_path = os.path.join(RESULTS_DIR, 'confusion_matrices', 'knn_confusion_matrix.png')
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()

    print(f"✓ Saved to: {save_path}\n")


def print_classification_report(y_true, y_pred, class_names):
    """
    Print detailed classification metrics
    """
    print("=" * 60)
    print("Detailed Classification Report")
    print("=" * 60)

    report = classification_report(y_true, y_pred,
                                   target_names=class_names,
                                   zero_division=0)
    print(report)

    # Save to file
    report_path = os.path.join(RESULTS_DIR, 'knn_classification_report.txt')
    with open(report_path, 'w') as f:
        f.write(report)

    print(f"\n✓ Report saved to: {report_path}\n")


def analyze_per_class_accuracy(y_true, y_pred, class_names):
    """
    Show which brick types are easiest/hardest to classify
    """
    print("=" * 60)
    print("Per-Class Accuracy Analysis")
    print("=" * 60)

    cm = confusion_matrix(y_true, y_pred)

    # Calculate per-class accuracy
    per_class_acc = []
    for i in range(len(class_names)):
        if cm[i].sum() > 0:
            acc = cm[i, i] / cm[i].sum()
            per_class_acc.append((class_names[i], acc, cm[i].sum()))
        else:
            per_class_acc.append((class_names[i], 0.0, 0))

    # Sort by accuracy
    per_class_acc.sort(key=lambda x: x[1], reverse=True)

    print("\nTop 10 BEST classified brick types:")
    print(f"{'Brick ID':<15} {'Accuracy':<12} {'Test Samples'}")
    print("-" * 45)
    for brick_id, acc, count in per_class_acc[:10]:
        print(f"{brick_id:<15} {acc * 100:>6.2f}%      {count:>4}")

    print("\nTop 10 WORST classified brick types:")
    print(f"{'Brick ID':<15} {'Accuracy':<12} {'Test Samples'}")
    print("-" * 45)
    for brick_id, acc, count in per_class_acc[-10:]:
        print(f"{brick_id:<15} {acc * 100:>6.2f}%      {count:>4}")

    print()


def save_model(knn, class_names):
    """
    Save trained KNN model and metadata
    """
    print("=" * 60)
    print("Saving Model")
    print("=" * 60)

    # Save KNN model
    knn_path = os.path.join(MODELS_DIR, 'knn_model.pkl')
    joblib.dump(knn, knn_path)
    print(f"✓ KNN model saved to: {knn_path}")

    # Save metadata
    metadata = {
        'class_names': class_names,
        'num_classes': len(class_names),
        'img_size': IMG_SIZE,
        'k_neighbors': K_NEIGHBORS,
        'model_type': 'KNN',
        'timestamp': datetime.now().isoformat()
    }
    metadata_path = os.path.join(MODELS_DIR, 'knn_metadata.pkl')
    joblib.dump(metadata, metadata_path)
    print(f"✓ Metadata saved to: {metadata_path}")

    # Print model info
    print(f"\nModel Information:")
    print(f"  • Classes: {len(class_names)}")
    print(f"  • k: {K_NEIGHBORS}")
    print(f"  • Image size: {IMG_SIZE}")

    print(f"{'=' * 60}\n")


def save_summary(results):
    """
    Save text summary of training results
    """
    summary_path = os.path.join(RESULTS_DIR, 'knn_training_summary.txt')

    with open(summary_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("KNN LEGO Brick Classification - Training Summary\n")
        f.write("=" * 60 + "\n\n")

        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("Configuration:\n")
        f.write(f"  • Dataset: {DATASET_PATH}\n")
        f.write(f"  • Image size: {IMG_SIZE}\n")
        f.write(f"  • k neighbors: {K_NEIGHBORS}\n")
        f.write(f"  • Train/test split: {int((1 - TEST_SIZE) * 100)}/{int(TEST_SIZE * 100)}\n\n")

        f.write("Results:\n")
        f.write(f"  • Number of classes: {results['num_classes']}\n")
        f.write(f"  • Total images: {results['total_images']:,}\n")
        f.write(f"  • Training images: {results['train_images']:,}\n")
        f.write(f"  • Test images: {results['test_images']:,}\n")
        f.write(f"  • Training time: {results['train_time']:.2f}s\n")
        f.write(f"  • Test accuracy: {results['accuracy']:.4f} ({results['accuracy'] * 100:.2f}%)\n")

        f.write("\n" + "=" * 60 + "\n")

    print(f"✓ Summary saved to: {summary_path}\n")

def main():
    """
    Main training pipeline
    """
    print("\n" + "=" * 60)
    print("🧱 KNN LEGO BRICK CLASSIFIER - TRAINING")
    print("=" * 60 + "\n")

    # Step 1: Load dataset
    print("STEP 1: Loading dataset...\n")
    X, y, class_names = load_dataset(
        max_images_per_class=100,  # Use 100 images per class for speed
        use_renders=True,
        use_photos=True
    )

    # Step 2: Split into train/test
    print("STEP 2: Splitting into train/test sets...\n")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    print(f"Training set: {len(X_train):,} images")
    print(f"Test set: {len(X_test):,} images\n")

    # Step 3: Train KNN
    print("STEP 3: Training KNN model...\n")
    knn, accuracy, y_pred, train_time = train_knn(
        X_train, y_train, X_test, y_test, k=K_NEIGHBORS
    )

    # Step 4: Generate visualizations
    print("STEP 4: Generating visualizations...\n")
    plot_confusion_matrix(y_test, y_pred, class_names)

    # Step 5: Detailed analysis
    print("STEP 5: Analyzing results...\n")
    print_classification_report(y_test, y_pred, class_names)
    analyze_per_class_accuracy(y_test, y_pred, class_names)

    # Step 6: Save model
    print("STEP 6: Saving model...\n")
    save_model(knn, class_names)

    # Step 7: Save summary
    print("STEP 7: Saving training summary...\n")
    results = {
        'num_classes': len(class_names),
        'total_images': len(X),
        'train_images': len(X_train),
        'test_images': len(X_test),
        'train_time': train_time,
        'accuracy': accuracy
    }
    save_summary(results)

    # Final summary
    print("=" * 60)
    print("✅ TRAINING COMPLETE!")
    print("=" * 60)
    print(f"\nFinal Results:")
    print(f"  • Model: KNN (k={K_NEIGHBORS})")
    print(f"  • Classes: {len(class_names)}")
    print(f"  • Test Accuracy: {accuracy * 100:.2f}%")
    print(f"  • Training Time: {train_time:.2f}s")
    print(f"\nSaved Files:")
    print(f"  • Model: {os.path.join(MODELS_DIR, 'knn_model.pkl')}")
    print(f"  • Metadata: {os.path.join(MODELS_DIR, 'knn_metadata.pkl')}")
    print(f"  • Confusion Matrix: {os.path.join(RESULTS_DIR, 'confusion_matrices', 'knn_confusion_matrix.png')}")
    print(f"  • Classification Report: {os.path.join(RESULTS_DIR, 'knn_classification_report.txt')}")
    print(f"  • Summary: {os.path.join(RESULTS_DIR, 'knn_training_summary.txt')}")
    print(f"\n{'=' * 60}\n")

if __name__ == '__main__':
    main()