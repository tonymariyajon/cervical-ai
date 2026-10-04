"""
Download remaining CRIC Cervix slide images with rate-limiting and retry logic.
"""

import csv
import json
import time
import urllib.request
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent
IMAGES_DIR = DATA_DIR / "images"
CSV_PATH = DATA_DIR / "classifications.csv"
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

def download_file(url: str, dest: Path, min_size: int = 100000, max_retries: int = 5) -> bool:
    temp_dest = dest.with_suffix(dest.suffix + ".tmp")
    for attempt in range(1, max_retries + 1):
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=45) as response, open(temp_dest, "wb") as out_file:
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
        except Exception as e:
            if temp_dest.exists():
                temp_dest.unlink()
            wait_time = attempt * 3
            print(f"    [Attempt {attempt}/{max_retries}] Error: {e}. Retrying in {wait_time}s...")
            time.sleep(wait_time)
    return False

def get_download_url(article_id: str, max_retries: int = 4) -> str:
    url = f"https://api.figshare.com/v2/articles/{article_id}"
    for attempt in range(1, max_retries + 1):
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=20) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                files = data.get("files", [])
                if files:
                    return files[0]["download_url"]
        except Exception as e:
            time.sleep(attempt * 2)
    return None

def main():
    with open(CSV_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        filename_to_aid = {row["image_filename"]: row["image_doi"].split(".")[-1] for row in reader}

    existing = {p.name for p in IMAGES_DIR.glob("*.png") if p.stat().st_size > 100000}
    missing = sorted(list(set(filename_to_aid.keys()) - existing))

    total_missing = len(missing)
    print(f"Total missing images to download: {total_missing}")

    success_count = 0
    for idx, fname in enumerate(missing, 1):
        aid = filename_to_aid[fname]
        dest = IMAGES_DIR / fname
        print(f"[{idx}/{total_missing}] Fetching URL for {fname} (Article: {aid})...")
        dl_url = get_download_url(aid)
        if not dl_url:
            print(f"  FAILED to get URL for {fname}")
            continue

        print(f"  Downloading {fname}...")
        ok = download_file(dl_url, dest)
        if ok:
            print(f"  SUCCESS ({dest.stat().st_size / (1024*1024):.2f} MB)")
            success_count += 1
        else:
            print(f"  FAILED to download {fname}")

        # Gentle pause to avoid rate limiting
        time.sleep(1.0)

    # Final tally
    final_existing = {p.name for p in IMAGES_DIR.glob("*.png") if p.stat().st_size > 100000}
    final_missing = set(filename_to_aid.keys()) - final_existing
    print("\n" + "=" * 60)
    print(f"Final Status:")
    print(f"  Total required in CSV: {len(filename_to_aid)}")
    print(f"  Total valid on disk:   {len(final_existing)}")
    print(f"  Remaining missing:     {len(final_missing)}")
    if not final_missing:
        print("  ALL 400 IMAGES SUCCESSFULLY DOWNLOADED AND VERIFIED!")
    print("=" * 60)

if __name__ == "__main__":
    main()
