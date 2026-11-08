

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import LeaveOneOut, cross_val_predict

from analyzer import SAMPLE_CSV, load_students
from model import LinearModel

students = load_students(SAMPLE_CSV)

homework = [s.homework for s in students]
y = [s.exams for s in students]

# ---------- 1. Same model two ways homework -> exam ----------
mine = LinearModel()
mine.fit(homework, y)


X1 = [[hw] for hw in homework]
theirs = LinearRegression().fit(X1, y)

print("HOMEWORK ONLY: MY MODEL vs SCIKIT-LEARN")
print(f"Slope:     mine {mine.slope:.4f}   scikit-learn {theirs.coef_[0]:.4f}")
print(f"Intercept: mine {mine.intercept:.4f}   scikit-learn {theirs.intercept_:.4f}")
same = abs(mine.slope - theirs.coef_[0]) < 1e-9 and abs(mine.intercept - theirs.intercept_) < 1e-9
print("Same result!" if same else "They differ - check the math.")

# ---------- 2. Two inputs homework + attendance ----------
X2 = [[s.homework, s.attendance] for s in students]
model2 = LinearRegression().fit(X2, y)

print()
print("SCIKIT-LEARN, HOMEWORK + ATTENDANCE")
print(f"Homework weight: {model2.coef_[0]:.2f}")
print(f"Attendance weight: {model2.coef_[1]:.2f}")
print(f"Intercept: {model2.intercept_:.2f}")

# Error on the students it learned from looks better than it really is...
print(f"Average error (students it trained on): {mean_absolute_error(y, model2.predict(X2)):.1f}")
# ...so also test each student with a model that never saw them
honest = cross_val_predict(LinearRegression(), X2, y, cv=LeaveOneOut())
print(f"Average error (unseen students):        {mean_absolute_error(y, honest):.1f}")
print(f"Prediction for 65 homework, 70% attendance: {model2.predict([[65, 70]])[0]:.1f}")
