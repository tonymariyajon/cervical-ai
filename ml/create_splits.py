"""
Reproducible Slide-Level Stratified Train/Val/Test Split for CRIC Cervix Dataset.

Ensures:
1. No data leakage: All cells belonging to the same slide (image_id) are kept strictly in the same split.
2. Stratification: All 6 Bethesda classes are balanced across train (~70%), val (~15%), and test (~15%).
3. Fully deterministic and reproducible using a fixed random seed (default: 42).
"""

import csv
import json
import random
from pathlib import Path
from collections import defaultdict, Counter

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CSV_PATH = DATA_DIR / "classifications.csv"

OUTPUT_SLIDE_SPLITS_CSV = DATA_DIR / "slide_splits.csv"
OUTPUT_SPLITS_JSON = DATA_DIR / "splits.json"

RANDOM_SEED = 42

def create_slide_splits(seed: int = RANDOM_SEED):
    random.seed(seed)

    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Classifications file not found: {CSV_PATH}")

    with open(CSV_PATH, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    # Aggregate cells by slide (image_id)
    slides = {}
    for r in rows:
        iid = int(r["image_id"])
        if iid not in slides:
            slides[iid] = {
                "image_id": iid,
                "image_filename": r["image_filename"],
                "classes": Counter(),
                "total_cells": 0
            }
        slides[iid]["classes"][r["bethesda_system"]] += 1
        slides[iid]["total_cells"] += 1

    total_slides = len(slides)
    target_slides = {
        "train": round(total_slides * 0.70),  # 280
        "val": round(total_slides * 0.15),    # 60
        "test": round(total_slides * 0.15)    # 60
    }
    target_proportions = {"train": 0.70, "val": 0.15, "test": 0.15}

    # Order classes from rarest to most common
    rarity_order = ["SCC", "ASC-H", "HSIL", "ASC-US", "LSIL", "Negative for intraepithelial lesion"]

    split_slides = {"train": [], "val": [], "test": []}
    split_class_cells = {"train": Counter(), "val": Counter(), "test": Counter()}
    assigned_slides = set()

    for cls in rarity_order:
        candidates = [iid for iid, s in slides.items() if iid not in assigned_slides and s["classes"][cls] > 0]
        # Sort deterministically by count of this class descending, then image_id
        candidates.sort(key=lambda x: (-slides[x]["classes"][cls], x))

        for iid in candidates:
            if iid in assigned_slides:
                continue

            best_split = None
            best_score = float("-inf")

            for split in ["train", "val", "test"]:
                if len(split_slides[split]) >= target_slides[split]:
                    continue

                expected_cells = target_proportions[split] * (
                    split_class_cells["train"][cls] + split_class_cells["val"][cls] + split_class_cells["test"][cls] + 1
                )
                actual_cells = split_class_cells[split][cls]
                deficit = expected_cells - actual_cells
                slide_capacity = target_slides[split] - len(split_slides[split])

                score = deficit * 10 + slide_capacity
                if score > best_score:
                    best_score = score
                    best_split = split

            if best_split is not None:
                split_slides[best_split].append(iid)
                assigned_slides.add(iid)
                for c, cnt in slides[iid]["classes"].items():
                    split_class_cells[best_split][c] += cnt

    # Assign remaining slides to splits with remaining capacity
    remaining = [iid for iid in sorted(slides.keys()) if iid not in assigned_slides]
    random.shuffle(remaining)
    for iid in remaining:
        for split in ["train", "val", "test"]:
            if len(split_slides[split]) < target_slides[split]:
                split_slides[split].append(iid)
                assigned_slides.add(iid)
                for c, cnt in slides[iid]["classes"].items():
                    split_class_cells[split][c] += cnt
                break

    # Sort slide IDs in each split for clean presentation
    for sp in split_slides:
        split_slides[sp].sort()

    # Save slide_splits.csv
    with open(OUTPUT_SLIDE_SPLITS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_id", "image_filename", "split", "cell_count"])
        for sp in ["train", "val", "test"]:
            for iid in split_slides[sp]:
                writer.writerow([iid, slides[iid]["image_filename"], sp, slides[iid]["total_cells"]])

    # Save splits.json
    splits_json_data = {
        "random_seed": seed,
        "target_proportions": target_proportions,
        "total_slides": total_slides,
        "total_cells": len(rows),
        "slide_counts": {sp: len(split_slides[sp]) for sp in split_slides},
        "cell_counts": {sp: sum(split_class_cells[sp].values()) for sp in split_slides},
        "class_counts": {
            cls: {sp: split_class_cells[sp][cls] for sp in ["train", "val", "test"]}
            for cls in rarity_order
        },
        "splits": {
            sp: [
                {"image_id": iid, "image_filename": slides[iid]["image_filename"], "cell_count": slides[iid]["total_cells"]}
                for iid in split_slides[sp]
            ]
            for sp in split_slides
        }
    }
    with open(OUTPUT_SPLITS_JSON, "w", encoding="utf-8") as f:
        json.dump(splits_json_data, f, indent=2)

    print("=" * 60)
    print(f"Slide-Level Split Complete (Seed = {seed})")
    print("=" * 60)
    print(f"Files created:")
    print(f"  - {OUTPUT_SLIDE_SPLITS_CSV.relative_to(PROJECT_ROOT)}")
    print(f"  - {OUTPUT_SPLITS_JSON.relative_to(PROJECT_ROOT)}")
    print("\nSlide Counts:")
    for sp in ["train", "val", "test"]:
        print(f"  {sp:5}: {len(split_slides[sp])} slides ({len(split_slides[sp])/total_slides*100:.1f}%)")

    total_cells = len(rows)
    print("\nCell Counts:")
    for sp in ["train", "val", "test"]:
        cnt = sum(split_class_cells[sp].values())
        print(f"  {sp:5}: {cnt} cells ({cnt/total_cells*100:.1f}%)")

    print("\nClass Distribution by Split:")
    header = f"{'Class':35} | {'Train':>12} | {'Val':>12} | {'Test':>12} | {'Total':>8}"
    print(header)
    print("-" * len(header))
    for c in ["Negative for intraepithelial lesion", "HSIL", "LSIL", "ASC-H", "ASC-US", "SCC"]:
        tr = split_class_cells["train"][c]
        va = split_class_cells["val"][c]
        te = split_class_cells["test"][c]
        tot = tr + va + te
        print(f"{c:35} | {tr:5} ({tr/tot*100:4.1f}%) | {va:5} ({va/tot*100:4.1f}%) | {te:5} ({te/tot*100:4.1f}%) | {tot:5}")

    # Disjointness check
    train_set = set(split_slides["train"])
    val_set = set(split_slides["val"])
    test_set = set(split_slides["test"])
    overlap = (train_set & val_set) | (train_set & test_set) | (val_set & test_set)
    assert len(overlap) == 0, f"Error: Overlapping image_ids found: {overlap}"
    assert len(train_set | val_set | test_set) == total_slides, "Error: Not all slides assigned!"
    print("\nVerification Passed: All 400 slides are strictly disjoint across the splits.")

if __name__ == "__main__":
    create_slide_splits(RANDOM_SEED)
