import io

import altair as alt
import pandas as pd
import streamlit as st

from analyzer import GENERATED_CSV, PASS_MARK, SAMPLE_CSV, Student, parse_students
from ml import BASELINE, MAIN_MODEL, data_fingerprint, load_models, predict, students_to_frame, train_models

# ---------- Page setup  ----------
st.set_page_config(
    page_title="Exam Score Predictor",
    layout="wide",
)

# Colors for each grade.
GRADE_COLORS = {
    "A": "#16A34A",
    "B": "#2563EB",
    "C": "#CA8A04",
    "D": "#EA580C",
    "F": "#DC2626",
}

# A little custom CSS for the prediction card
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    html, body, .stApp, .stApp p, .stApp li, .stApp label, .stApp button, .stApp input,
    .stApp textarea, .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6,
    .stApp [data-testid="stMarkdownContainer"], .stApp [data-testid="stMetricValue"],
    .stApp [data-testid="stMetricLabel"], .stApp [data-testid="stMetricDelta"],
    .stApp [data-testid="stCaptionContainer"], .stApp [data-baseweb="tab"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
    }
    .block-container { padding-top: 2rem; }
    .score-card {
        background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 100%);
        color: white;
        border-radius: 18px;
        padding: 28px;
        text-align: center;
        box-shadow: 0 10px 25px rgba(79, 70, 229, 0.25);
    }
    .score-card .label { font-size: 0.95rem; opacity: 0.85; letter-spacing: 0.05em; }
    .score-card .score { font-size: 4rem; font-weight: 800; line-height: 1.1; margin: 8px 0; }
    .score-card .badge {
        display: inline-block;
        padding: 4px 16px;
        border-radius: 999px;
        font-weight: 700;
        background: white;
    }
    .score-card .note { font-size: 0.85rem; opacity: 0.85; margin-top: 14px; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------- Helpers (cached so moving a slider doesn't redo the work) ----------
@st.cache_data
def read_students(data: bytes) -> tuple[pd.DataFrame, list[str]]:
    students, problems = parse_students(io.StringIO(data.decode("utf-8-sig")))
    return students_to_frame(students), problems


@st.cache_resource(show_spinner="Training the model…")
def get_models(data_hash: str, _df: pd.DataFrame):
    
    return load_models(data_hash) or train_models(_df)


# ---------- Sidebar: data + inputs ----------
with st.sidebar:
    st.header("Data")
    uploaded = st.file_uploader(
        "Upload a student CSV",
        type="csv",
        help="Columns needed: name, homework, attendance, exams",
    )
    st.download_button(
        "Download sample CSV",
        SAMPLE_CSV.read_bytes(),
        file_name="students.csv",
        mime="text/csv",
        width="stretch",
    )

    st.divider()
    st.header("New student")
    homework = st.slider("Homework score", 0, 100, 65)
    attendance = st.slider("Attendance (%)", 0, 100, 70)


# ---------- Load data ----------
if uploaded is not None:
    data = uploaded.getvalue()
    source = f"your file **{uploaded.name}**"
else:
    data = GENERATED_CSV.read_bytes()
    source = "the **sample data**"

try:
    df, problems = read_students(data)
except (UnicodeDecodeError, ValueError):
    st.error("That file doesn't look right. It needs the columns: name, homework, attendance, exams.")
    st.stop()

if problems:
    st.sidebar.warning(f"Skipped {len(problems)} row(s) with missing or invalid values.")

if len(df) < 5:
    st.warning("Add at least 5 students to train a model.")
    st.stop()


# ---------- Train the model ----------
models = get_models(data_fingerprint(data), df)
model = models[MAIN_MODEL]
error = model.cv_mae   

prediction = predict(model, homework, attendance)
predicted = Student("New student", homework, attendance, round(prediction))
grade = predicted.letter_grade()


# ---------- Header ----------
st.title("Student Exam Score Predictor")
st.caption(
    f"A machine learning model trained on {len(df)} students from {source}. "
    "Move the sliders in the sidebar to predict a new student's exam score."
)


# ---------- Tabs ----------
tab_predict, tab_class, tab_model = st.tabs(["Predict", "Class overview", "How it works"])


# ===== Tab 1 Predict =====
with tab_predict:
    left, right = st.columns([1, 2], gap="large")

    with left:
        result = "Likely to pass" if predicted.passed() else "At risk of failing"
        st.markdown(
            f"""
            <div class="score-card">
                <div class="label">PREDICTED EXAM SCORE</div>
                <div class="score">{prediction:.0f}</div>
                <span class="badge" style="color:{GRADE_COLORS[grade]}">Grade {grade}</span>
                <div class="note">{result}<br>Typical error: ±{error:.1f} points</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.write("")
        c1, c2 = st.columns(2)
        c1.metric("Homework", homework, f"{homework - df['Homework'].mean():+.0f} vs class")
        c2.metric("Attendance", f"{attendance}%", f"{attendance - df['Attendance'].mean():+.0f} vs class")

    with right:
        # (attendance fixed at the slider value)
        line_df = pd.DataFrame({"Homework": range(0, 101, 5)})
        line_df["Attendance"] = attendance
        line_df["Exam"] = model.estimator.predict(line_df[model.features]).clip(0, 100)

        axis_x = alt.X("Homework:Q", scale=alt.Scale(domain=[0, 100]), title="Homework score")
        axis_y = alt.Y("Exam:Q", scale=alt.Scale(domain=[0, 100]), title="Exam score")

        points = alt.Chart(df).mark_circle(size=110, opacity=0.75).encode(
            x=axis_x,
            y=axis_y,
            color=alt.Color(
                "Grade:N",
                scale=alt.Scale(domain=list(GRADE_COLORS), range=list(GRADE_COLORS.values())),
                legend=alt.Legend(orient="bottom"),
            ),
            tooltip=["Name", "Homework", "Attendance", "Exam", "Grade"],
        )
        line = alt.Chart(line_df).mark_line(color="#4F46E5", strokeWidth=3, strokeDash=[6, 4]).encode(
            x=axis_x, y=axis_y,
        )
        you = alt.Chart(pd.DataFrame({"Homework": [homework], "Exam": [prediction]})).mark_point(
            shape="diamond", size=400, filled=True, color="#7C3AED", stroke="white", strokeWidth=2,
        ).encode(x=axis_x, y=axis_y, tooltip=[alt.Tooltip("Exam:Q", title="Predicted", format=".1f")])

        st.subheader("Where the new student lands")
        st.altair_chart((line + points + you).properties(height=420).configure(font="Inter"), width="stretch")
        st.caption("Dots are real students. The dashed line is the model. The purple diamond is your new student.")


# ===== Tab 2 Class overview =====
with tab_class:
    pass_rate = (df["Exam"] >= PASS_MARK).mean() * 100
    top = df.loc[df["Exam"].idxmax()]

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Students", len(df))
    m2.metric("Average exam", f"{df['Exam'].mean():.1f}")
    m3.metric("Pass rate", f"{pass_rate:.0f}%")
    m4.metric("Top student", top["Name"], f"{top['Exam']} points", delta_color="off")

    table_col, chart_col = st.columns([3, 2], gap="large")

    with table_col:
        st.dataframe(
            df,
            hide_index=True,
            width="stretch",
            column_config={
                "Homework": st.column_config.ProgressColumn("Homework", min_value=0, max_value=100, format="%d"),
                "Attendance": st.column_config.ProgressColumn("Attendance", min_value=0, max_value=100, format="%d%%"),
                "Exam": st.column_config.ProgressColumn("Exam", min_value=0, max_value=100, format="%d"),
            },
        )

    with chart_col:
        counts = df["Grade"].value_counts().reindex(list(GRADE_COLORS), fill_value=0).reset_index()
        counts.columns = ["Grade", "Students"]
        bars = alt.Chart(counts).mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6).encode(
            x=alt.X("Grade:N", sort=list(GRADE_COLORS), axis=alt.Axis(labelAngle=0)),
            y=alt.Y("Students:Q", axis=alt.Axis(tickMinStep=1)),
            color=alt.Color(
                "Grade:N",
                scale=alt.Scale(domain=list(GRADE_COLORS), range=list(GRADE_COLORS.values())),
                legend=None,
            ),
            tooltip=["Grade", "Students"],
        )
        st.subheader("Grade distribution")
        st.altair_chart(bars.properties(height=300).configure(font="Inter"), width="stretch")


# ===== Tab 3 How it works =====
with tab_model:
    hw_weight, att_weight = model.estimator.coef_
    st.subheader("The formula the model learned")
    st.latex(
        rf"\text{{exam}} = {hw_weight:.2f} \times \text{{homework}}"
        rf" + {att_weight:.2f} \times \text{{attendance}} + {model.estimator.intercept_:.2f}"
    )

    w1, w2, w3 = st.columns(3)
    w1.metric("Homework weight", f"{hw_weight:.2f}")
    w2.metric("Attendance weight", f"{att_weight:.2f}")
    w3.metric("Average error", f"±{error:.1f}")

    st.markdown(
        f"""
        **What this means**

        - This is a **linear regression**: it finds the straight-line rule that best fits the students' real scores.
        - A **weight** says how much the exam score changes for each extra point of that input.
        - The **average error** is measured on students the model never saw during training, so it is an honest score.
          Just guessing the class average would be off by ±{models[BASELINE].cv_mae:.1f}.

        I first wrote a simple version of this model from scratch in plain Python, then checked that scikit-learn gives the same result.
        """
    )


# ---------- Footer ----------
st.divider()
st.caption("Built by Jabel Richard with Python, scikit-learn and Streamlit · [jabelrichard.com](https://jabelrichard.com)")
