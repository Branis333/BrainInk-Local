import argparse
import json
import os
from typing import Any, Dict, List


def make_example(example_id: str, image_path: str) -> Dict[str, Any]:
    prompt = "Plot y = 2x + 1 for x in [-2,2]. Label intercept."
    rubric = (
        "Total 10 points. "
        "(1) Correct slope (0-3). "
        "(2) Correct intercept (0-3). "
        "(3) Line drawn correctly (0-3). "
        "(4) Axes labeled (0-1)."
    )
    criteria: List[Dict[str, Any]] = [
        {"id": "slope", "name": "Correct slope", "max_points": 3},
        {"id": "intercept", "name": "Correct intercept", "max_points": 3},
        {"id": "line", "name": "Line drawn correctly", "max_points": 3},
        {"id": "axes", "name": "Axes labeled", "max_points": 1},
    ]

    # grade_json is a STRING containing JSON (this is what you train the model to output)
    grade_obj = {
        "mismatch": False,
        "mismatch_reason": "",
        "total_points": 10,
        "points_awarded": 0,
        "criteria": [
            {
                "id": "slope",
                "points": 0,
                "max_points": 3,
                "evidence": "",
            }
        ],
        "feedback": "",
    }

    return {
        "id": example_id,
        "image_path": image_path,
        "prompt": prompt,
        "rubric": rubric,
        "criteria": criteria,
        "grade_json": json.dumps(grade_obj, ensure_ascii=False),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Create a starter VLM train.jsonl template for BrainInk.")
    ap.add_argument("--out", default="data/train.jsonl", help="Output JSONL path")
    ap.add_argument("--n", type=int, default=5, help="How many template rows")
    ap.add_argument("--image-dir", default="data/images", help="Where images will live")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    os.makedirs(args.image_dir, exist_ok=True)

    with open(args.out, "w", encoding="utf-8") as f:
        for i in range(args.n):
            ex_id = f"u_{i:06d}"
            img_path = os.path.join(args.image_dir, ex_id + ".png").replace("\\", "/")
            row = make_example(ex_id, img_path)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print("Wrote:", args.out)
    print("Images should go under:", args.image_dir)
    print("Next: fill in each row's grade_json with your gold labels.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
