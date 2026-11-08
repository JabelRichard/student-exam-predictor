
from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

from analyzer import GENERATED_CSV, REQUIRED_COLUMNS

FIRST_NAMES = [
    "Aisha", "Ben", "Carlos", "Dana", "Elena", "Femi", "Grace", "Hiro", "Isla", "Jonas",
    "Kemi", "Liam", "Maya", "Noah", "Olu", "Priya", "Quinn", "Ravi", "Sara", "Tomas",
    "Uma", "Victor", "Wen", "Ximena", "Yusuf", "Zara", "Amir", "Bella", "Chidi", "Diego",
    "Esther", "Farah", "Gabriel", "Hana", "Ivan", "Jade", "Kofi", "Lena", "Mateo", "Nia",
]
LAST_INITIALS = "ABCDEFGHJKLMNOPRSTW"


def clamp(value: float) -> int:
    return max(0, min(100, round(value)))


def generate(rows: int = 150, seed: int = 42) -> list[dict[str, str | int]]:
    
    rng = random.Random(seed)

    all_names = [f"{first} {initial}." for first in FIRST_NAMES for initial in LAST_INITIALS]
    if rows > len(all_names):
        raise ValueError(f"Can generate at most {len(all_names)} unique students")
    names = rng.sample(all_names, rows)

    students = []
    for name in names:
        attendance = clamp(rng.gauss(75, 15))
        homework = clamp(0.7 * attendance + rng.gauss(10, 14))
        exams = clamp(18 + 0.55 * homework + 0.2 * attendance + rng.gauss(0, 7))
        students.append({
            "name": name,
            "homework": homework,
            "attendance": attendance,
            "exams": exams,
        })
    return students


def save(students: list[dict[str, str | int]], path: Path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=REQUIRED_COLUMNS)
        writer.writeheader()
        writer.writerows(students)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rows", type=int, default=150, help="number of students (default 150)")
    parser.add_argument("--seed", type=int, default=42, help="random seed (default 42)")
    parser.add_argument("--out", type=Path, default=GENERATED_CSV, help="output CSV file")
    args = parser.parse_args()

    students = generate(args.rows, args.seed)
    save(students, args.out)
    print(f"Saved {len(students)} students to {args.out}")
