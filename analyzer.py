
from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable

# Paths are built from this file's folder, so the scripts work no matter

PROJECT_DIR = Path(__file__).resolve().parent
SAMPLE_CSV = PROJECT_DIR / "students.csv"
GENERATED_CSV = PROJECT_DIR / "students_generated.csv"
REPORT_FILE = PROJECT_DIR / "report.txt"

REQUIRED_COLUMNS = ("name", "homework", "attendance", "exams")
PASS_MARK = 50


class Student:
    def __init__(self, name: str, homework: int, attendance: int, exams: int) -> None:
        self.name = name
        self.homework = homework
        self.attendance = attendance
        self.exams = exams

    def __repr__(self) -> str:
        return (
            f"Student({self.name!r}, homework={self.homework}, "
            f"attendance={self.attendance}, exams={self.exams})"
        )

    def letter_grade(self) -> str:
        if self.exams >= 80:
            return "A"
        elif self.exams >= 70:
            return "B"
        elif self.exams >= 60:
            return "C"
        elif self.exams >= PASS_MARK:
            return "D"
        return "F"

    def passed(self) -> bool:
        return self.exams >= PASS_MARK

    def summary(self) -> str:
        return (
            f"{self.name} | HW: {self.homework} | Att: {self.attendance}% | "
            f"Exam: {self.exams} ({self.letter_grade()}) | Passed: {self.passed()}"
        )


# ---------- Loading ----------

def _parse_score(value: str | None, column: str) -> int:
    """Turn one cell into a 0-100 whole number, or raise ValueError saying what's wrong."""
    text = (value or "").strip()
    try:
        score = int(text)
    except ValueError:
        raise ValueError(f"{column} '{text}' is not a whole number") from None
    if not 0 <= score <= 100:
        raise ValueError(f"{column} {score} is not between 0 and 100")
    return score


def parse_students(lines: Iterable[str]) -> tuple[list[Student], list[str]]:
    """Read students from CSV text.

    Works with an open file or any iterable of lines (e.g. an uploaded file).
    Returns (students, problems): rows that can't be read are skipped and a
    message for each one is added to `problems`, instead of crashing.
    Raises ValueError if a required column is missing entirely.
    """
    reader = csv.DictReader(lines, skipinitialspace=True)
    columns = [c.strip().lower() for c in (reader.fieldnames or [])]
    missing = [c for c in REQUIRED_COLUMNS if c not in columns]
    if missing:
        raise ValueError(f"Missing column(s): {', '.join(missing)}")
    reader.fieldnames = columns

    students: list[Student] = []
    problems: list[str] = []
    
    for line_number, row in enumerate(reader, start=2):
        try:
            name = (row["name"] or "").strip()
            if not name:
                raise ValueError("name is empty")
            students.append(Student(
                name,
                _parse_score(row["homework"], "homework"),
                _parse_score(row["attendance"], "attendance"),
                _parse_score(row["exams"], "exams"),
            ))
        except ValueError as error:
            problems.append(f"Line {line_number}: {error}")
    return students, problems


def load_students(filename: str | Path = SAMPLE_CSV) -> list[Student]:
    """Load students from a CSV file, printing a warning for any bad rows."""
    try:
        with open(filename, newline="", encoding="utf-8-sig") as f:
            students, problems = parse_students(f)
    except FileNotFoundError:
        print(f"Could not find file: {filename}")
        return []
    except ValueError as error:
        print(f"Could not read {filename}: {error}")
        return []

    for problem in problems:
        print(f"Skipped a row in {Path(filename).name}. {problem}")
    return students


# ---------- Analysis ----------

def print_student(student: Student) -> None:
    print(f"{student.name:<10} {student.homework:<10} {student.attendance:<12} "
          f"{student.exams:<8} {student.letter_grade():<6}")


def print_all(students: list[Student]) -> None:
    print(f"{'Name':<10} {'Homework':<10} {'Attendance':<12} {'Exams':<8} {'Grade':<6}")
    print("-" * 50)
    for student in students:
        print_student(student)


def find_student(students: list[Student], name: str) -> Student | None:
    for student in students:
        if student.name.lower() == name.lower():
            return student
    return None


def average(students: list[Student], field: str) -> float:
    if not students:
        return 0
    return sum(getattr(student, field) for student in students) / len(students)


def top_student(students: list[Student], field: str) -> Student | None:
    top = None
    for student in students:
        if top is None or getattr(student, field) > getattr(top, field):
            top = student
    return top


def grade_counts(students: list[Student]) -> dict[str, int]:
    counts = {"A": 0, "B": 0, "C": 0, "D": 0, "F": 0}
    for student in students:
        counts[student.letter_grade()] += 1
    return counts


def save_report(students: list[Student], filename: str | Path = REPORT_FILE) -> None:
    top = top_student(students, "exams")
    if top is None:
        raise ValueError("Can't write a report with no students.")

    with open(filename, "w", encoding="utf-8") as f:
        f.write("CLASS REPORT\n")
        f.write("=" * 50 + "\n")
        f.write(f"Top student (exams): {top.name} with {top.exams}\n")
        f.write(f"Average homework score: {average(students, 'homework'):.1f}\n")
        f.write(f"Average attendance score: {average(students, 'attendance'):.1f}\n")
        f.write(f"Average exams score: {average(students, 'exams'):.1f}\n")
        f.write("Grade counts for exams:\n")
        for grade, count in grade_counts(students).items():
            f.write(f"{grade}: {count}\n")

        f.write("\n")
        f.write(f"{'Name':<10} {'Homework':<10} {'Attendance':<12} {'Exams':<8} {'Grade':<6}\n")
        f.write("-" * 50 + "\n")
        for student in students:
            f.write(f"{student.name:<10} {student.homework:<10} {student.attendance:<12} "
                    f"{student.exams:<8} {student.letter_grade():<6}\n")

    print(f"Report saved to {filename}")


# ---------- Main program ----------
if __name__ == "__main__":
    students = load_students(SAMPLE_CSV)

    if students:
        print_all(students)
        print()

        kevin = find_student(students, "kevin")
        if kevin:
            print(kevin.summary())
        print()

        print(f"Average homework score: {average(students, 'homework'):.1f}")
        print(f"Average attendance score: {average(students, 'attendance'):.1f}")
        print(f"Average exams score: {average(students, 'exams'):.1f}")

        top = top_student(students, "exams")
        print(f"Top student (exams): {top.name} with {top.exams}")
        print("Grade counts:", grade_counts(students))

        save_report(students)
    else:
        print("No data to analyze.")
