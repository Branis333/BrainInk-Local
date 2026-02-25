import io
import csv
import time
import random
from pathlib import Path
from google.cloud import vision
from google.api_core import exceptions as gax_exceptions

# Folder containing your handwritten images
IMAGE_FOLDER = "E:\\school\\BrainInk-Local\\ocr\\data"   # root folder (supports nested subfolders)
OUTPUT_CSV = "transcriptions.csv"
MAX_RETRIES = 5
INITIAL_BACKOFF_SECONDS = 1.0
MAX_BACKOFF_SECONDS = 20.0
REQUEST_TIMEOUT_SECONDS = 60
CALL_DELAY_SECONDS = 0.05


def _path_sort_key(path_obj: Path):
    parts = []
    for part in path_obj.parts:
        if part.isdigit():
            parts.append((0, int(part)))
        else:
            parts.append((1, part.lower()))
    return parts


def collect_images(image_root: str):
    root = Path(image_root)
    if not root.exists():
        raise FileNotFoundError(f"Image folder not found: {root.resolve()}")

    valid_exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
    image_paths = [
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in valid_exts
    ]
    image_paths.sort(key=lambda p: _path_sort_key(p.relative_to(root)))
    return root, image_paths

def _is_retryable_error(err: Exception) -> bool:
    return isinstance(
        err,
        (
            gax_exceptions.ServiceUnavailable,
            gax_exceptions.DeadlineExceeded,
            gax_exceptions.TooManyRequests,
            gax_exceptions.InternalServerError,
        ),
    )


def transcribe_image(client, image_path):
    with io.open(image_path, "rb") as image_file:
        content = image_file.read()

    image = vision.Image(content=content)

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.document_text_detection(
                image=image,
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
            break
        except Exception as err:
            if _is_retryable_error(err) and attempt < MAX_RETRIES:
                backoff = min(MAX_BACKOFF_SECONDS, INITIAL_BACKOFF_SECONDS * (2 ** (attempt - 1)))
                backoff = backoff + random.uniform(0.0, 0.35)
                print(
                    f"Retry {attempt}/{MAX_RETRIES} for {image_path} due to transient error: {err}. "
                    f"Sleeping {backoff:.2f}s"
                )
                time.sleep(backoff)
                continue

            print(f"Failed processing {image_path}: {err}")
            return ""

    if response.error.message:
        print(f"Error processing {image_path}: {response.error.message}")
        return ""

    if response.full_text_annotation:
        return response.full_text_annotation.text
    else:
        return ""


def load_processed_paths(csv_path: Path):
    processed = set()
    if not csv_path.exists():
        return processed

    with open(csv_path, mode="r", newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        if not reader.fieldnames or "image_path" not in reader.fieldnames:
            return processed

        for row in reader:
            image_path = (row.get("image_path") or "").strip()
            if image_path:
                processed.add(image_path)
    return processed

def main():
    client = vision.ImageAnnotatorClient()

    root, image_files = collect_images(IMAGE_FOLDER)
    print(f"Found {len(image_files)} images under: {root.resolve()}")

    if not image_files:
        print("No images found. Check IMAGE_FOLDER path and extensions.")
        return

    output_path = Path(OUTPUT_CSV)
    processed_paths = load_processed_paths(output_path)
    print(f"Already processed in CSV: {len(processed_paths)}")

    mode = "a" if output_path.exists() else "w"
    with open(output_path, mode=mode, newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        if mode == "w":
            writer.writerow(["image_path", "transcription"])

        for idx, image_path in enumerate(image_files):
            rel_path = image_path.relative_to(root).as_posix()
            if rel_path in processed_paths:
                continue

            print(f"Processing {idx+1}/{len(image_files)}: {rel_path}")

            try:
                text = transcribe_image(client, str(image_path))
                writer.writerow([rel_path, text])
                csv_file.flush()
                processed_paths.add(rel_path)
                if CALL_DELAY_SECONDS > 0:
                    time.sleep(CALL_DELAY_SECONDS)
            except Exception as err:
                print(f"Unexpected error on {rel_path}: {err}")
                writer.writerow([rel_path, ""])
                csv_file.flush()
                processed_paths.add(rel_path)

    print("Done. Transcriptions saved to", output_path)

if __name__ == "__main__":
    main()