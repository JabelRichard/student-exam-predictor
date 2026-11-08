import io

import pytest

from analyzer import (
    SAMPLE_CSV,
    Student,
    average,
    find_student,
    grade_counts,
    load_students,
    parse_students,
    save_report,
    top_student,
)


@pytest.mark.parametrize(
    ("exam", "grade", "passed"),
    [(100, "A", True), (80, "A", True), (79, "B", True), (70, "B", True), (69, "C", True),
     (60, "C", True), (59, "D", True), (50, "D", True), (49, "F", False), (0, "F", False)],
)
def test_letter_grade_and_passed(exam, grade, passed):
    student = Student("x", 0, 0, exam)
    assert student.letter_grade() == grade
    assert student.passed() is passed


def test_parse_handles_spaces_and_header_case():
    text = "Name, Homework, Attendance, Exams\nJack, 34, 45, 54\n"
    students, problems = parse_students(io.StringIO(text))
    assert problems == []
    assert len(students) == 1
    jack = students[0]
    assert (jack.name, jack.homework, jack.attendance, jack.exams) == ("Jack", 34, 45, 54)


def test_parse_skips_bad_rows_with_a_message():
    text = (
        "name,homework,attendance,exams\n"
        "Good,50,60,70\n"
        "Words,abc,60,70\n"
        "TooHigh,50,160,70\n"
        ",50,60,70\n"
    )
    students, problems = parse_students(io.StringIO(text))
    assert [s.name for s in students] == ["Good"]
    assert len(problems) == 3
    assert "Line 3" in problems[0] and "abc" in problems[0]
    assert "160" in problems[1]
    assert "name is empty" in problems[2]


def test_parse_rejects_missing_columns():
    with pytest.raises(ValueError, match="exams"):
        parse_students(io.StringIO("name,homework,attendance\nA,1,2\n"))


def test_load_sample_file_from_any_folder(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)   # run from a different folder on purpose
    students = load_students(SAMPLE_CSV)
    assert len(students) == 10


def test_load_missing_file_returns_empty(tmp_path, capsys):
    assert load_students(tmp_path / "nope.csv") == []
    assert "Could not find file" in capsys.readouterr().out


@pytest.fixture
def three_students():
    return [Student("Ann", 80, 90, 85), Student("Bo", 60, 70, 45), Student("Cy", 70, 80, 72)]


def test_helpers(three_students):
    assert average(three_students, "exams") == pytest.approx((85 + 45 + 72) / 3)
    assert average([], "exams") == 0
    assert top_student(three_students, "exams").name == "Ann"
    assert top_student([], "exams") is None
    assert find_student(three_students, "bo").name == "Bo"
    assert find_student(three_students, "zed") is None
    assert grade_counts(three_students) == {"A": 1, "B": 1, "C": 0, "D": 0, "F": 1}


def test_save_report(tmp_path, three_students):
    report = tmp_path / "report.txt"
    save_report(three_students, report)
    text = report.read_text(encoding="utf-8")
    assert "Top student (exams): Ann with 85" in text
    assert "F: 1" in text


def test_save_report_needs_students(tmp_path):
    with pytest.raises(ValueError):
        save_report([], tmp_path / "report.txt")
