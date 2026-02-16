import argparse
import json
import os
from typing import Any, Dict, List, Tuple


REQUIRED_KEYS = ["image_path", "prompt", "rubric", "criteria", "grade_json"]


def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for i, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise SystemExit(f"Invalid JSON on line {i}: {e}\nLine: {line[:200]}")
    return rows


def _try_import_pil():
    try:
        from PIL import Image  # type: ignore

        return Image
    except Exception:
        return None


def _validate_row(row: Dict[str, Any], base_dir: str, pil_image) -> Tuple[List[str], List[str]]:
    errors: List[str] = []
    warnings: List[str] = []

    for k in REQUIRED_KEYS:
        if k not in row:
            errors.append(f"missing key: {k}")

    if "image_paths" in row and "image_path" in row:
        warnings.append("both image_path and image_paths are present; prefer one")

    # Resolve images
    image_paths: List[str] = []
    if isinstance(row.get("image_paths"), list):
        image_paths = [str(p) for p in row.get("image_paths")]
    elif "image_path" in row:
        image_paths = [str(row.get("image_path"))]

    if not image_paths:
        errors.append("no image_path(s) provided")
    else:
        for p in image_paths:
            resolved = p
            if not os.path.isabs(resolved):
                resolved = os.path.normpath(os.path.join(base_dir, resolved))
            if not os.path.exists(resolved):
                errors.append(f"image not found: {p}")
                continue
            if pil_image is not None:
                try:
                    with pil_image.open(resolved) as im:
                        im.verify()
                except Exception as e:
                    errors.append(f"image unreadable: {p} ({e})")

    # Criteria
    criteria = row.get("criteria")
    if not isinstance(criteria, list) or not criteria:
        errors.append("criteria must be a non-empty list")
    else:
        for idx, c in enumerate(criteria):
            if not isinstance(c, dict):
                errors.append(f"criteria[{idx}] must be object")
                continue
            for ck in ("id", "name", "max_points"):
                if ck not in c:
                    errors.append(f"criteria[{idx}] missing {ck}")
            if "max_points" in c and not isinstance(c.get("max_points"), (int, float)):
                errors.append(f"criteria[{idx}].max_points must be number")

    # grade_json
    gj = row.get("grade_json")
    if not isinstance(gj, str) or not gj.strip():
        errors.append("grade_json must be a non-empty string containing JSON")
    else:
        try:
            parsed = json.loads(gj)
            if not isinstance(parsed, dict):
                warnings.append("grade_json parses but is not an object")
        except Exception as e:
            errors.append(f"grade_json is not valid JSON string: {e}")

    # Basic strings
    for sk in ("prompt", "rubric"):
        if sk in row and not isinstance(row.get(sk), str):
            errors.append(f"{sk} must be string")

    return errors, warnings


def main() -> int:
    ap = argparse.ArgumentParser(description="Validate BrainInk Janus-Pro VLM dataset JSONL + images.")
    ap.add_argument("jsonl", help="Path to train.jsonl/eval.jsonl")
    ap.add_argument(
        "--base-dir",
        default=".",
        help="Base directory for resolving relative image paths (default: current directory)",
    )
    ap.add_argument(
        "--max-print",
        type=int,
        default=20,
        help="Max number of per-row errors/warnings to print (default: 20)",
    )
    args = ap.parse_args()

    rows = _read_jsonl(args.jsonl)
    if not rows:
        raise SystemExit("No rows found in JSONL.")

    base_dir = os.path.abspath(args.base_dir)
    pil_image = _try_import_pil()

    total = 0
    bad = 0
    warn_rows = 0
    mismatch_true = 0

    printed = 0
    for i, row in enumerate(rows, start=1):
        total += 1
        errors, warnings = _validate_row(row, base_dir=base_dir, pil_image=pil_image)

        # mismatch stats (if your grade_json includes it)
        try:
            parsed = json.loads(row.get("grade_json", "{}"))
            if isinstance(parsed, dict) and parsed.get("mismatch") is True:
                mismatch_true += 1
        except Exception:
            pass

        if errors:
            bad += 1
        if warnings:
            warn_rows += 1

        if (errors or warnings) and printed < args.max_print:
            printed += 1
            rid = row.get("id")
            prefix = f"row {i}" + (f" (id={rid})" if rid else "")
            if errors:
                print(prefix + " ERRORS:")
                for e in errors:
                    print("  -", e)
            if warnings:
                print(prefix + " WARNINGS:")
                for w in warnings:
                    print("  -", w)

    print("\n=== Summary ===")
    print("Rows:", total)
    print("Bad rows:", bad)
    print("Rows with warnings:", warn_rows)
    if total:
        print("Mismatch=true rate (from grade_json):", round(mismatch_true / total, 4))

    if bad:
        raise SystemExit(2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
