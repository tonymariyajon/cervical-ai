"""
CRIC Cervical Cell Dataset Preprocessing Script
Generates a 224x224 nucleus-centered cell-level dataset from slide images and annotations.

Input:
- data/classifications.csv (annotations and nucleus coordinates)
- data/slide_splits.csv (slide-level train/val/test split)
- data/images/ (original 400 microscope slide images)

Output:
- data/cells/train/{class_name}/cell_xxxxx.png
- data/cells/val/{class_name}/cell_xxxxx.png
- data/cells/test/{class_name}/cell_xxxxx.png
- data/cells/cells_metadata.csv (index of all generated crops)
"""

import csv
import sys
import time
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
IMAGES_DIR = DATA_DIR / "images"
CLASSIFICATIONS_CSV = DATA_DIR / "classifications.csv"
SLIDE_SPLITS_CSV = DATA_DIR / "slide_splits.csv"
OUTPUT_CELLS_DIR = DATA_DIR / "cells"
OUTPUT_METADATA_CSV = OUTPUT_CELLS_DIR / "cells_metadata.csv"

CROP_SIZE = 224
HALF_CROP = CROP_SIZE // 2  # 112

CLASS_FOLDER_MAP = {
    "Negative for intraepithelial lesion": "Negative",
    "ASC-US": "ASC-US",
    "ASC-H": "ASC-H",
    "LSIL": "LSIL",
    "HSIL": "HSIL",
    "SCC": "SCC",
}

SPLIT_NAMES = ["train", "val", "test"]

def prepare_directories():
    """Create directory structure for train, val, and test splits."""
    for split in SPLIT_NAMES:
        for folder_name in CLASS_FOLDER_MAP.values():
            target_dir = OUTPUT_CELLS_DIR / split / folder_name
            target_dir.mkdir(parents=True, exist_ok=True)

def load_slide_splits():
    """Load slide to split mapping."""
    if not SLIDE_SPLITS_CSV.exists():
        raise FileNotFoundError(f"Missing slide splits file: {SLIDE_SPLITS_CSV}")
    
    slide_to_split = {}
    with open(SLIDE_SPLITS_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            slide_to_split[row["image_filename"]] = row["split"]
    return slide_to_split

def load_annotations():
    """Load cell annotations grouped by slide image."""
    if not CLASSIFICATIONS_CSV.exists():
        raise FileNotFoundError(f"Missing classifications file: {CLASSIFICATIONS_CSV}")

    annotations_by_slide = defaultdict(list)
    total_cells = 0
    with open(CLASSIFICATIONS_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            annotations_by_slide[row["image_filename"]].append({
                "cell_id": int(row["cell_id"]),
                "image_id": int(row["image_id"]),
                "image_filename": row["image_filename"],
                "bethesda_system": row["bethesda_system"],
                "nucleus_x": int(row["nucleus_x"]),
                "nucleus_y": int(row["nucleus_y"]),
            })
            total_cells += 1
    return annotations_by_slide, total_cells

def generate_crops():
    start_time = time.time()
    print("=" * 65)
    print("CRIC Cell Dataset Generation (224x224 Nucleus-Centered Crops)")
    print("=" * 65)

    prepare_directories()
    slide_to_split = load_slide_splits()
    annotations_by_slide, total_expected_cells = load_annotations()

    print(f"Loaded {len(annotations_by_slide)} slides containing {total_expected_cells} cell annotations.")
    print(f"Slide split mapping: {len(slide_to_split)} slides.")

    metadata_records = []
    processed_cells = 0
    boundary_cells_count = 0
    split_counter = Counter()
    class_counter = Counter()

    slide_filenames = sorted(annotations_by_slide.keys())
    total_slides = len(slide_filenames)

    for idx, slide_filename in enumerate(slide_filenames, 1):
        slide_path = IMAGES_DIR / slide_filename
        if not slide_path.exists():
            raise FileNotFoundError(f"Slide image not found: {slide_path}")

        split = slide_to_split.get(slide_filename)
        if not split:
            raise ValueError(f"Slide {slide_filename} not found in slide_splits.csv!")

        # Load slide image once
        with Image.open(slide_path) as img:
            img_rgb = img.convert("RGB")
            img_arr = np.array(img_rgb)  # Shape: (height, width, 3)

        img_h, img_w, _ = img_arr.shape

        # Apply symmetric reflection padding of HALF_CROP (112 px) around entire slide
        padded_arr = np.pad(
            img_arr,
            ((HALF_CROP, HALF_CROP), (HALF_CROP, HALF_CROP), (0, 0)),
            mode="reflect"
        )

        cells = annotations_by_slide[slide_filename]
        for cell in cells:
            cid = cell["cell_id"]
            x = cell["nucleus_x"]
            y = cell["nucleus_y"]
            cls_raw = cell["bethesda_system"]
            folder_name = CLASS_FOLDER_MAP.get(cls_raw)

            if not folder_name:
                raise ValueError(f"Unknown class '{cls_raw}' for cell {cid}")

            # Check if this cell required boundary reflection
            if (x - HALF_CROP < 0 or x + HALF_CROP > img_w or
                y - HALF_CROP < 0 or y + HALF_CROP > img_h):
                boundary_cells_count += 1

            # In padded array, crop centered on (x, y) is exactly [y : y + CROP_SIZE, x : x + CROP_SIZE]
            crop_arr = padded_arr[y : y + CROP_SIZE, x : x + CROP_SIZE]

            # Validate dimensions and format
            if crop_arr.shape != (CROP_SIZE, CROP_SIZE, 3):
                raise ValueError(f"Unexpected crop shape {crop_arr.shape} for cell {cid}")

            # Save crop
            cell_filename = f"cell_{cid:05d}.png"
            dest_path = OUTPUT_CELLS_DIR / split / folder_name / cell_filename
            crop_img = Image.fromarray(crop_arr)
            crop_img.save(dest_path, format="PNG")

            processed_cells += 1
            split_counter[split] += 1
            class_counter[folder_name] += 1

            # Store metadata record
            rel_crop_path = f"{split}/{folder_name}/{cell_filename}"
            metadata_records.append({
                "cell_id": cid,
                "image_id": cell["image_id"],
                "image_filename": slide_filename,
                "bethesda_system": cls_raw,
                "class_folder": folder_name,
                "split": split,
                "nucleus_x": x,
                "nucleus_y": y,
                "crop_path": rel_crop_path,
            })

        if idx % 50 == 0 or idx == total_slides:
            elapsed = time.time() - start_time
            print(f"  Processed {idx}/{total_slides} slides ({processed_cells}/{total_expected_cells} cells) [{elapsed:.1f}s]")

    # Save consolidated cells_metadata.csv
    print("\nSaving cells_metadata.csv...")
    metadata_records.sort(key=lambda r: r["cell_id"])
    with open(OUTPUT_METADATA_CSV, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "cell_id", "image_id", "image_filename", "bethesda_system",
            "class_folder", "split", "nucleus_x", "nucleus_y", "crop_path"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metadata_records)

    total_time = time.time() - start_time
    print(f"Dataset generation complete in {total_time:.1f}s.")
    print(f"Total cells generated: {processed_cells}")
    print(f"Boundary cells padded via reflection: {boundary_cells_count}")

def verify_dataset():
    print("\n" + "=" * 65)
    print("VERIFICATION OF GENERATED CELL DATASET")
    print("=" * 65)

    if not OUTPUT_METADATA_CSV.exists():
        raise FileNotFoundError(f"Missing {OUTPUT_METADATA_CSV}")

    with open(OUTPUT_METADATA_CSV, "r", encoding="utf-8") as f:
        records = list(csv.DictReader(f))

    print(f"1. Total metadata records: {len(records)}")

    # Check disk files
    all_pngs = list(OUTPUT_CELLS_DIR.glob("**/*.png"))
    print(f"2. Total PNG files on disk: {len(all_pngs)}")
    assert len(records) == len(all_pngs) == 11534, "Mismatch in total file count!"

    # Verify split counts
    split_counts = Counter(r["split"] for r in records)
    print("\n3. Crops per split:")
    for sp in ["train", "val", "test"]:
        print(f"   - {sp:5}: {split_counts[sp]:5} crops ({split_counts[sp]/len(records)*100:.1f}%)")

    # Verify class counts
    class_counts = Counter(r["class_folder"] for r in records)
    print("\n4. Crops per class:")
    for cls_name in ["Negative", "HSIL", "LSIL", "ASC-H", "ASC-US", "SCC"]:
        print(f"   - {cls_name:10}: {class_counts[cls_name]:5} crops")

    # Sample check of image dimensions and mode
    print("\n5. Checking sample image dimensions and color mode...")
    sample_indices = [0, 100, 1000, 5000, 10000, 11533]
    for idx in sample_indices:
        rec = records[idx]
        p = OUTPUT_CELLS_DIR / rec["crop_path"]
        assert p.exists(), f"Missing file: {p}"
        with Image.open(p) as im:
            assert im.size == (224, 224), f"Wrong size {im.size} for {p}"
            assert im.mode == "RGB", f"Wrong mode {im.mode} for {p}"
    print("   Verification PASSED: Sample images are strictly 224x224 RGB.")

    # Cross-reference with slide_splits.csv
    with open(SLIDE_SPLITS_CSV, "r", encoding="utf-8") as f:
        slide_splits = {r["image_filename"]: r["split"] for r in csv.DictReader(f)}

    for rec in records:
        expected_split = slide_splits[rec["image_filename"]]
        assert rec["split"] == expected_split, f"Split mismatch for cell {rec['cell_id']}!"

    print("6. Cross-reference with slide_splits.csv: 100% matched, zero leakage.")
    print("\n" + "=" * 65)
    print("ALL VERIFICATIONS SUCCESSFUL!")
    print("=" * 65)

if __name__ == "__main__":
    generate_crops()
    verify_dataset()
