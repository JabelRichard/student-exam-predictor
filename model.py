

from __future__ import annotations

from analyzer import SAMPLE_CSV, load_students


class LinearModel:
    def __init__(self) -> None:
        self.slope = 0.0
        self.intercept = 0.0

    def fit(self, xs: list[float], ys: list[float]) -> None:
       
        if len(xs) != len(ys) or not xs:
            raise ValueError("xs and ys must be non-empty and the same length")

        avg_x = sum(xs) / len(xs)
        avg_y = sum(ys) / len(ys)

        top = 0.0
        bottom = 0.0
        for i in range(len(xs)):
            x_diff = xs[i] - avg_x
            y_diff = ys[i] - avg_y
            top += x_diff * y_diff
            bottom += x_diff ** 2

        if bottom == 0:
            raise ValueError("All x values are the same, so there is no slope to learn")

        # Save what was learned ON the model, so predict() can use it
        self.slope = top / bottom
        self.intercept = avg_y - self.slope * avg_x

    def predict(self, x: float) -> float:
        """Use the learned line to make a prediction."""
        return self.slope * x + self.intercept


def mean_absolute_error(model: LinearModel, xs: list[float], ys: list[float]) -> float:
    """Average distance between the model's predictions and the real answers."""
    total_error = 0.0
    for i in range(len(xs)):
        total_error += abs(model.predict(xs[i]) - ys[i])
    return total_error / len(xs)


# ---------- Main program (only runs when you run this file directly) ----------
if __name__ == "__main__":
    students = load_students(SAMPLE_CSV)

    homework = [s.homework for s in students]
    attendance = [s.attendance for s in students]
    exams = [s.exams for s in students]

    # ---------- Model 1 homework -> exam ----------
    hw_model = LinearModel()
    hw_model.fit(homework, exams)

    print("HOMEWORK MODEL")
    print(f"Slope: {hw_model.slope:.2f}, Intercept: {hw_model.intercept:.2f}")
    print(f"Prediction for 65 homework: {hw_model.predict(65):.1f}")
    print(f"Average error: {mean_absolute_error(hw_model, homework, exams):.1f}")
    print()

    # ---------- Model 2  attendance -> exam ----------
    att_model = LinearModel()
    att_model.fit(attendance, exams)

    print("ATTENDANCE MODEL")
    print(f"Slope: {att_model.slope:.2f}, Intercept: {att_model.intercept:.2f}")
    print(f"Prediction for 65% attendance: {att_model.predict(65):.1f}")
    print(f"Average error: {mean_absolute_error(att_model, attendance, exams):.1f}")
    print()

    # ---------- Where does the attendance model go wrong? ----------
    print("Attendance model, student by student:")
    for i, student in enumerate(students):
        predicted = att_model.predict(attendance[i])
        error = abs(predicted - exams[i])
        print(f"{student.name:<10} att {attendance[i]:>3}% -> predicted {predicted:5.1f}, "
              f"real {exams[i]:>3}, off by {error:5.1f}")
