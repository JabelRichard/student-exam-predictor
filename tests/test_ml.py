import pytest

from analyzer import GENERATED_CSV, SAMPLE_CSV, Student, load_students
from generate_data import generate
from ml import (
    BASELINE,
    MAIN_MODEL,
    MODEL_FEATURES,
    best_model,
    data_fingerprint,
    load_models,
    predict,
    save_models,
    students_to_frame,
    train_models,
)


@pytest.fixture(scope="module")
def generated_models():
    students = [Student(**row) for row in generate(rows=150, seed=42)]
    return train_models(students_to_frame(students))


def test_generate_is_repeatable_and_valid():
    rows = generate(rows=200, seed=1)
    assert rows == generate(rows=200, seed=1)
    assert rows != generate(rows=200, seed=2)
    assert len({r["name"] for r in rows}) == 200
    for r in rows:
        for column in ("homework", "attendance", "exams"):
            assert 0 <= r[column] <= 100


def test_generate_has_some_failing_students():
    rows = generate(rows=150, seed=42)
    assert any(r["exams"] < 50 for r in rows)


def test_trains_every_model(generated_models):
    assert set(generated_models) == set(MODEL_FEATURES)
    for model in generated_models.values():
        assert len(model.cv_predictions) == 150


def test_models_beat_the_baseline(generated_models):
    baseline = generated_models[BASELINE].cv_mae
    assert generated_models[MAIN_MODEL].cv_mae < baseline - 2
    assert best_model(generated_models).name != BASELINE


def test_predictions_stay_in_range(generated_models):
    main = generated_models[MAIN_MODEL]
    assert 0 <= predict(main, 0, 0) <= 100
    assert 0 <= predict(main, 100, 100) <= 100
    assert predict(main, 90, 90) > predict(main, 30, 30)


def test_small_dataset_uses_honest_error():
    # On only 10 students, error on unseen students is clearly worse than on training ones
    models = train_models(students_to_frame(load_students(SAMPLE_CSV)))
    assert models[MAIN_MODEL].cv_mae > models[MAIN_MODEL].train_mae


def test_save_and_load(tmp_path, generated_models):
    path = tmp_path / "model.joblib"
    save_models(generated_models, "abc123", path)

    loaded = load_models("abc123", path)
    assert loaded is not None
    assert predict(loaded[MAIN_MODEL], 65, 70) == pytest.approx(predict(generated_models[MAIN_MODEL], 65, 70))

    assert load_models("different-data", path) is None      # trained on other data -> retrain
    assert load_models("abc123", tmp_path / "missing.joblib") is None


def test_saved_model_file_matches_generated_data():
    # If this fails, the data changed: run `python train.py` again
    loaded = load_models(data_fingerprint(GENERATED_CSV.read_bytes()))
    assert loaded is not None
