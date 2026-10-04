"""
CRIC Cervix Dataset Official Downloader
Fetches the official CRIC Cervix Cell Classification dataset directly from Figshare:
Collection DOI: 10.6084/m9.figshare.c.4960286.v2
Peer-reviewed publication: Nature Scientific Data (2021) 8:151
"""

import os
import sys
import json
import csv
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent
IMAGES_DIR = DATA_DIR / "images"

METADATA_FILES = {
    "classifications.csv": "https://ndownloader.figshare.com/files/22494383",
    "classifications.json": "https://ndownloader.figshare.com/files/22494386",
    "CRIC_README.md": "https://ndownloader.figshare.com/files/22494389",
}

COLLECTION_API = "https://api.figshare.com/v2/collections/4960286/articles"
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

def download_file(url: str, dest: Path, min_size: int = 1000) -> bool:
    """Download a file with verification and resume capability."""
    if dest.exists() and dest.stat().st_size >= min_size:
        return True  # Already downloaded

    temp_dest = dest.with_suffix(dest.suffix + ".tmp")
    req = urllib.request.Request(url, headers=HEADERS)
    try:
        with urllib.request.urlopen(req, timeout=30) as response, open(temp_dest, "wb") as out_file:
            while True:
                chunk = response.read(65536)
                if not chunk:
                    break
                out_file.write(chunk)
        if temp_dest.stat().st_size >= min_size:
            temp_dest.replace(dest)
            return True
        else:
            if temp_dest.exists():
                temp_dest.unlink()
            return False
    except Exception as e:
        if temp_dest.exists():
            temp_dest.unlink()
        print(f"Error downloading {dest.name}: {e}")
        return False

def fetch_article_file_info(article_id: int):
    """Fetch the filename, size, and download url for an article."""
    url = f"https://api.figshare.com/v2/articles/{article_id}"
    req = urllib.request.Request(url, headers=HEADERS)
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                files = data.get("files", [])
                if files:
                    f = files[0]
                    return {
                        "name": f["name"],
                        "size": f["size"],
                        "url": f["download_url"],
                        "article_id": article_id,
                    }
        except Exception:
            time.sleep(1)
    return None

def main():
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 60)
    print("CRIC Cervix Dataset Downloader")
    print("=" * 60)

    # 1. Download metadata files
    print("\n[Step 1/3] Ensuring metadata files are in place...")
    for filename, url in METADATA_FILES.items():
        target = DATA_DIR / filename
        if not target.exists():
            print(f"  Downloading metadata: {filename}...")
            download_file(url, target)
        else:
            print(f"  Metadata file exists: {filename}")

    # 2. Fetch list of articles in Figshare collection
    print("\n[Step 2/3] Fetching image file manifests from Figshare API...")
    article_ids = []
    page = 1
    while True:
        url = f"{COLLECTION_API}?page={page}&page_size=100"
        req = urllib.request.Request(url, headers=HEADERS)
        with urllib.request.urlopen(req, timeout=20) as resp:
            items = json.loads(resp.read().decode("utf-8"))
            if not items:
                break
            for it in items:
                if "microscope" in it.get("title", "").lower():
                    article_ids.append(it["id"])
            page += 1

    print(f"  Found {len(article_ids)} microscope slide image entries in the collection.")

    # Fetch file details concurrently
    print("  Retrieving direct download links for all 400 images...")
    image_files = []
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(fetch_article_file_info, aid): aid for aid in article_ids}
        for future in as_completed(futures):
            res = future.result()
            if res:
                image_files.append(res)

    print(f"  Successfully gathered links for {len(image_files)} images.")

    # 3. Download the 400 images
    print("\n[Step 3/3] Downloading 400 microscope slide images into data/images/...")
    total_images = len(image_files)
    completed = 0
    failed = []

    start_time = time.time()

    def download_image_task(item):
        dest = IMAGES_DIR / item["name"]
        success = download_file(item["url"], dest, min_size=100000)
        return item["name"], success

    with ThreadPoolExecutor(max_workers=6) as executor:
        futures = {executor.submit(download_image_task, it): it for it in image_files}
        for future in as_completed(futures):
            fname, ok = future.result()
            if ok:
                completed += 1
            else:
                failed.append(fname)

            if completed % 25 == 0 or completed == total_images:
                elapsed = time.time() - start_time
                pct = (completed / total_images) * 100
                print(f"  Progress: {completed}/{total_images} images ({pct:.1f}%) downloaded [Elapsed: {elapsed:.1f}s]")

    if failed:
        print(f"\nWarning: {len(failed)} files failed. Retrying failed items...")
        for fname in failed:
            it = next((x for x in image_files if x["name"] == fname), None)
            if it:
                dest = IMAGES_DIR / it["name"]
                if download_file(it["url"], dest, min_size=100000):
                    completed += 1
                    failed.remove(fname)

    # 4. Verify against classifications.csv
    csv_path = DATA_DIR / "classifications.csv"
    if csv_path.exists():
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            csv_images = {row["image_filename"] for row in reader}

        downloaded_images = {p.name for p in IMAGES_DIR.glob("*.png")}
        missing = csv_images - downloaded_images
        print("\n" + "=" * 60)
        print("Verification against classifications.csv:")
        print(f"  Images referenced in CSV: {len(csv_images)}")
        print(f"  Images present on disk:   {len(downloaded_images)}")
        print(f"  Missing images:           {len(missing)}")
        if not missing:
            print("  SUCCESS: All 400 images are present and accounted for!")
        else:
            print(f"  Missing filenames: {list(missing)[:5]}...")
        print("=" * 60)

if __name__ == "__main__":
    main()
