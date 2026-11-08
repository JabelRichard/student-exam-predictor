import pytest
from sklearn.linear_model import LinearRegression

from analyzer import SAMPLE_CSV, load_students
from model import LinearModel, mean_absolute_error


def test_learns_a_perfect_line():
    model = LinearModel()
    model.fit([0, 1, 2, 3], [5, 7, 9, 11])     # y = 2x + 5
    assert model.slope == pytest.approx(2)
    assert model.intercept == pytest.approx(5)
    assert model.predict(10) == pytest.approx(25)
    assert mean_absolute_error(model, [0, 1, 2, 3], [5, 7, 9, 11]) == pytest.approx(0)


def test_matches_scikit_learn():
    students = load_students(SAMPLE_CSV)
    xs = [s.homework for s in students]
    ys = [s.exams for s in students]

    mine = LinearModel()
    mine.fit(xs, ys)
    theirs = LinearRegression().fit([[x] for x in xs], ys)

    assert mine.slope == pytest.approx(theirs.coef_[0])
    assert mine.intercept == pytest.approx(theirs.intercept_)


def test_rejects_data_with_no_slope():
    with pytest.raises(ValueError):
        LinearModel().fit([5, 5, 5], [1, 2, 3])


def test_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        LinearModel().fit([1, 2], [1])
