# organize_dataset.py
import os
import pandas as pd
import shutil
from collections import Counter

# Paths
DATASET_PATH = 'brick-images'
INVENTORY_CSV = 'brick-box-inventory.csv'
OUTPUT_PATH = 'brick-images-subset'  # Filtered dataset


def analyze_dataset():
    """
    Analyze what brick types you have and how many images
    """
    print("=" * 60)
    print("LEGO Dataset Analysis")
    print("=" * 60)

    brick_counts = {}

    for split in ['photos', 'renders']:
        split_path = os.path.join(DATASET_PATH, split)

        if not os.path.exists(split_path):
            print(f"Warning: {split_path} not found!")
            continue

        print(f"\nAnalyzing {split}/...")

        brick_types = [d for d in os.listdir(split_path)
                       if os.path.isdir(os.path.join(split_path, d))
                       and not d.startswith('.')]

        print(f"Found {len(brick_types)} brick types in {split}/")

        for brick_id in brick_types:
            brick_path = os.path.join(split_path, brick_id)

            # Count images
            images = [f for f in os.listdir(brick_path)
                      if f.endswith(('.jpg', '.png', '.jpeg'))
                      and not f.startswith('.')]

            num_images = len(images)

            if brick_id not in brick_counts:
                brick_counts[brick_id] = {'photos': 0, 'renders': 0}

            brick_counts[brick_id][split] = num_images

    # Display results
    print(f"\n{'=' * 60}")
    print(f"Total unique brick types in dataset: {len(brick_counts)}")
    print(f"{'=' * 60}\n")

    # Sort by total count
    sorted_bricks = sorted(
        brick_counts.items(),
        key=lambda x: x[1]['photos'] + x[1]['renders'],
        reverse=True
    )

    print(f"{'Brick ID':<15} {'Photos':<10} {'Renders':<10} {'Total':<10}")
    print("-" * 60)

    for brick_id, counts in sorted_bricks[:30]:  # Show top 30
        total = counts['photos'] + counts['renders']
        print(f"{brick_id:<15} {counts['photos']:<10} {counts['renders']:<10} {total:<10}")

    if len(sorted_bricks) > 30:
        print(f"\n... and {len(sorted_bricks) - 30} more brick types")

    # Summary stats
    total_photos = sum(c['photos'] for c in brick_counts.values())
    total_renders = sum(c['renders'] for c in brick_counts.values())

    print(f"\n{'=' * 60}")
    print(f"SUMMARY:")
    print(f"Total photos: {total_photos:,}")
    print(f"Total renders: {total_renders:,}")
    print(f"Grand total: {total_photos + total_renders:,} images")
    print(f"{'=' * 60}\n")

    return brick_counts


def load_inventory():
    """
    Load brick box inventory from CSV
    """
    print("=" * 60)
    print("Loading Brick Box Inventory")
    print("=" * 60)

    if not os.path.exists(INVENTORY_CSV):
        print(f"ERROR: {INVENTORY_CSV} not found!")
        return None

    df = pd.read_csv(INVENTORY_CSV)

    print(f"\nCSV loaded successfully!")
    print(f"Columns: {list(df.columns)}")
    print(f"Total rows: {len(df)}\n")

    # Get unique DesignIDs (shape, not color)
    if 'DesignID' not in df.columns:
        print("ERROR: 'DesignID' column not found!")
        print(f"Available columns: {list(df.columns)}")
        return None

    # Convert DesignID to string (in case it's numeric)
    df['DesignID'] = df['DesignID'].astype(str)

    unique_elements = df['DesignID'].unique()

    print(f"Unique brick types in your box: {len(unique_elements)}")
    print(f"\nFirst 20 ElementIDs:")
    for i, elem_id in enumerate(unique_elements[:20]):
        print(f"  {elem_id}")

    if len(unique_elements) > 20:
        print(f"  ... and {len(unique_elements) - 20} more")

    return df


def match_inventory_to_dataset(inventory_df, brick_counts):
    """
    Match brick box inventory to available dataset
    """
    print("\n" + "=" * 60)
    print("Matching Inventory to Dataset")
    print("=" * 60 + "\n")

    inventory_ids = set(inventory_df['DesignID'].astype(str).unique())
    dataset_ids = set(brick_counts.keys())

    # Find matches
    matched_ids = inventory_ids & dataset_ids  # Intersection
    missing_in_dataset = inventory_ids - dataset_ids

    print(f"Brick types in your box: {len(inventory_ids)}")
    print(f"Brick types in dataset: {len(dataset_ids)}")
    print(f"MATCHED: {len(matched_ids)} brick types ✓")
    print(f"Missing from dataset: {len(missing_in_dataset)} brick types\n")

    if missing_in_dataset:
        print("Missing brick types:")
        for elem_id in sorted(missing_in_dataset)[:10]:
            print(f"  ✗ {elem_id}")
        if len(missing_in_dataset) > 10:
            print(f"  ... and {len(missing_in_dataset) - 10} more")
        print()

    # Show matched bricks with image counts
    print(f"\n{'=' * 60}")
    print(f"MATCHED BRICKS (Available for Training)")
    print(f"{'=' * 60}\n")

    print(f"{'DesignID':<15} {'Photos':<10} {'Renders':<10} {'Total':<10}")
    print("-" * 60)

    matched_with_counts = []
    for elem_id in sorted(matched_ids):
        counts = brick_counts[elem_id]
        total = counts['photos'] + counts['renders']
        matched_with_counts.append((elem_id, counts, total))
        print(f"{elem_id:<15} {counts['photos']:<10} {counts['renders']:<10} {total:<10}")

    # Summary
    total_images = sum(c[2] for c in matched_with_counts)
    print(f"\n{'=' * 60}")
    print(f"TOTAL: {len(matched_ids)} brick types, {total_images:,} images")
    print(f"{'=' * 60}\n")

    return list(matched_ids)


def create_subset_dataset(matched_ids, brick_counts, max_images_per_type=1000):
    """
    Create a subset dataset with only matched brick types
    """
    print("=" * 60)
    print("Creating Filtered Dataset Subset")
    print("=" * 60 + "\n")

    if os.path.exists(OUTPUT_PATH):
        print(f"Warning: {OUTPUT_PATH} already exists!")
        response = input("Delete and recreate? (yes/no): ").lower()
        if response == 'yes':
            shutil.rmtree(OUTPUT_PATH)
            print("Deleted existing subset.")
        else:
            print("Cancelled. Using existing subset.")
            return

    os.makedirs(OUTPUT_PATH, exist_ok=True)

    total_copied = 0

    for split in ['photos', 'renders']:
        print(f"\nProcessing {split}/...")

        split_src = os.path.join(DATASET_PATH, split)
        split_dst = os.path.join(OUTPUT_PATH, split)

        os.makedirs(split_dst, exist_ok=True)

        for elem_id in matched_ids:
            src_folder = os.path.join(split_src, elem_id)
            dst_folder = os.path.join(split_dst, elem_id)

            if not os.path.exists(src_folder):
                continue

            # Get all images
            images = [f for f in os.listdir(src_folder)
                      if f.endswith(('.jpg', '.png', '.jpeg'))
                      and not f.startswith('.')]

            # Limit to max_images_per_type
            if len(images) > max_images_per_type:
                import random
                images = random.sample(images, max_images_per_type)

            # Copy images
            os.makedirs(dst_folder, exist_ok=True)

            for img_name in images:
                src_file = os.path.join(src_folder, img_name)
                dst_file = os.path.join(dst_folder, img_name)
                shutil.copy2(src_file, dst_file)

            total_copied += len(images)
            print(f"  ✓ {elem_id}: {len(images)} images")

    print(f"\n{'=' * 60}")
    print(f"Subset created successfully!")
    print(f"Location: {OUTPUT_PATH}")
    print(f"Total images: {total_copied:,}")
    print(f"Brick types: {len(matched_ids)}")
    print(f"{'=' * 60}\n")


if __name__ == '__main__':
    print("\n🧱 LEGO BRICK BOX DATASET ORGANIZER\n")

    # Step 1: Analyze full dataset
    print("STEP 1: Analyzing full dataset...")
    brick_counts = analyze_dataset()

    input("\nPress Enter to continue...")

    # Step 2: Load inventory
    print("\nSTEP 2: Loading brick box inventory...")
    inventory_df = load_inventory()

    if inventory_df is None:
        print("ERROR: Could not load inventory. Exiting.")
        exit(1)

    input("\nPress Enter to continue...")

    # Step 3: Match inventory to dataset
    print("\nSTEP 3: Matching inventory to dataset...")
    matched_ids = match_inventory_to_dataset(inventory_df, brick_counts)

    if len(matched_ids) == 0:
        print("ERROR: No matches found! Check DesignID column.")
        exit(1)

    input("\nPress Enter to continue...")

    # Step 4: Create subset
    print(f"\nSTEP 4: Creating subset dataset...")
    print(f"This will copy {len(matched_ids)} brick types to '{OUTPUT_PATH}/'")
    response = input("Continue? (yes/no): ").lower()

    if response == 'yes':
        create_subset_dataset(matched_ids, brick_counts, max_images_per_type=800)
        print("\n✅ DONE! You can now use 'brick-images-subset' for training.")
    else:
        print("\nCancelled. No files copied.")

    print("\n" + "=" * 60)
    print("Summary:")
    print(f"  • Original dataset: {len(brick_counts)} brick types")
    print(f"  • Your brick box: {len(inventory_df['DesignID'].unique())} brick types")
    print(f"  • Matched: {len(matched_ids)} brick types")
    print(f"  • Subset location: {OUTPUT_PATH}/")
    print("=" * 60)