

from __future__ import annotations

import hashlib
import warnings
from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd
from sklearn.base import RegressorMixin
from sklearn.dummy import DummyRegressor
from sklearn.exceptions import InconsistentVersionWarning
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold, LeaveOneOut, cross_val_predict

from analyzer import PROJECT_DIR, Student

MODEL_PATH = PROJECT_DIR / "model.joblib"

BASELINE = "Baseline (class average)"
HOMEWORK_ONLY = "Homework only"
MAIN_MODEL = "Homework + attendance"

# Which input columns each model uses
MODEL_FEATURES: dict[str, list[str]] = {
    BASELINE: ["Homework", "Attendance"],   
    HOMEWORK_ONLY: ["Homework"],
    MAIN_MODEL: ["Homework", "Attendance"],
}


@dataclass
class TrainedModel:
    name: str
    features: list[str]
    estimator: RegressorMixin
    train_mae: float              # error on the students it learned from (too optimistic)
    cv_mae: float                 # error on students it did NOT see (honest)
    cv_predictions: list[float]  


@dataclass
class SavedModels:
    data_hash: str                
    models: dict[str, TrainedModel]


def students_to_frame(students: list[Student]) -> pd.DataFrame:
    return pd.DataFrame([
        {
            "Name": s.name,
            "Homework": s.homework,
            "Attendance": s.attendance,
            "Exam": s.exams,
            "Grade": s.letter_grade(),
        }
        for s in students
    ])


def _new_estimator(name: str) -> RegressorMixin:
    return DummyRegressor(strategy="mean") if name == BASELINE else LinearRegression()


def _splitter(n_students: int) -> LeaveOneOut | KFold:
    
    if n_students < 20:
        return LeaveOneOut()
    return KFold(n_splits=5, shuffle=True, random_state=42)


def train_models(df: pd.DataFrame) -> dict[str, TrainedModel]:
   
    y = df["Exam"]
    results: dict[str, TrainedModel] = {}

    for name, features in MODEL_FEATURES.items():
        X = df[features]
        cv_predictions = cross_val_predict(_new_estimator(name), X, y, cv=_splitter(len(df)))

        # The final model learns from every student
        estimator = _new_estimator(name).fit(X, y)
        results[name] = TrainedModel(
            name=name,
            features=features,
            estimator=estimator,
            train_mae=float(mean_absolute_error(y, estimator.predict(X))),
            cv_mae=float(mean_absolute_error(y, cv_predictions)),
            cv_predictions=[float(p) for p in cv_predictions],
        )
    return results


def predict(model: TrainedModel, homework: float, attendance: float) -> float:
    """Predict one new student's exam score, kept between 0 and 100."""
    row = pd.DataFrame({"Homework": [homework], "Attendance": [attendance]})[model.features]
    return float(min(100, max(0, model.estimator.predict(row)[0])))


def best_model(models: dict[str, TrainedModel]) -> TrainedModel:
    return min(models.values(), key=lambda m: m.cv_mae)


# ---------- Saving and loading ----------

def data_fingerprint(data: bytes) -> str:
    # Normalise line endings so Windows (CRLF) and Linux (LF) checkouts of the same CSV match
    return hashlib.sha256(data.replace(b"\r\n", b"\n")).hexdigest()


def save_models(models: dict[str, TrainedModel], data_hash: str, path: Path = MODEL_PATH) -> None:
    joblib.dump(SavedModels(data_hash=data_hash, models=models), path)


def load_models(data_hash: str, path: Path = MODEL_PATH) -> dict[str, TrainedModel] | None:
    
    if not Path(path).exists():
        return None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", InconsistentVersionWarning)
            saved = joblib.load(path)
    except Exception:
        return None
    if not isinstance(saved, SavedModels) or saved.data_hash != data_hash:
        return None
    return saved.models
