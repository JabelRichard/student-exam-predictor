
from __future__ import annotations

import argparse
import io
from pathlib import Path

from analyzer import GENERATED_CSV, parse_students
from ml import MODEL_PATH, best_model, data_fingerprint, save_models, students_to_frame, train_models

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, default=GENERATED_CSV, help="student CSV to train on")
    parser.add_argument("--out", type=Path, default=MODEL_PATH, help="where to save the models")
    args = parser.parse_args()

    if not args.data.exists():
        raise SystemExit(f"{args.data} not found. Run `python generate_data.py` first.")

    data = args.data.read_bytes()
    students, problems = parse_students(io.StringIO(data.decode("utf-8-sig")))
    for problem in problems:
        print(f"Skipped a row. {problem}")

    models = train_models(students_to_frame(students))
    save_models(models, data_fingerprint(data), args.out)

    print(f"Trained on {len(students)} students from {args.data.name}\n")
    print(f"{'Model':<28} {'Error (training)':>17} {'Error (unseen)':>15}")
    print("-" * 62)
    for model in models.values():
        print(f"{model.name:<28} {model.train_mae:>17.1f} {model.cv_mae:>15.1f}")
    print(f"\nBest on unseen students: {best_model(models).name}")
    print(f"Saved to {args.out}")
