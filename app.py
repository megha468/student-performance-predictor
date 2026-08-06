
import streamlit as st
import pandas as pd
import joblib
import os

# ------------------------------------------------------------------
# Load the trained models (same folder as this file)
# ------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_resource
def load_models():
    reg_model = joblib.load(os.path.join(BASE_DIR, "model_score_regressor.pkl"))
    clf_passfail = joblib.load(os.path.join(BASE_DIR, "model_passfail_classifier.pkl"))
    clf_grade = joblib.load(os.path.join(BASE_DIR, "model_grade_classifier.pkl"))
    le_pf = joblib.load(os.path.join(BASE_DIR, "encoder_passfail.pkl"))
    le_grade = joblib.load(os.path.join(BASE_DIR, "encoder_grade.pkl"))
    feature_cols = joblib.load(os.path.join(BASE_DIR, "feature_columns.pkl"))
    return reg_model, clf_passfail, clf_grade, le_pf, le_grade, feature_cols

reg_model, clf_passfail, clf_grade, le_pf, le_grade, feature_cols = load_models()

# ------------------------------------------------------------------
# Page setup
# ------------------------------------------------------------------
st.set_page_config(page_title="Student Performance Predictor", page_icon="🎓")
st.title("🎓 Student Performance Predictor")
st.write("Fill in the student's details below and click **Predict**.")

# ------------------------------------------------------------------
# Input form
# ------------------------------------------------------------------
with st.form("student_form"):
    col1, col2 = st.columns(2)

    with col1:
        study_hours = st.slider("Study hours per week", 0.0, 40.0, 15.0, 0.5)
        attendance = st.slider("Attendance (%)", 30.0, 100.0, 80.0, 1.0)
        previous_score = st.slider("Previous exam score", 0.0, 100.0, 65.0, 1.0)
        sleep_hours = st.slider("Average sleep (hours/night)", 3.0, 10.0, 7.0, 0.5)

    with col2:
        parental_education = st.selectbox(
            "Parental education",
            ["high_school", "bachelors", "masters", "phd"],
        )
        internet_access = st.selectbox("Internet access at home", ["yes", "no"])
        extracurricular = st.selectbox("Extracurricular activities", ["yes", "no"])

    submitted = st.form_submit_button("Predict")

# ------------------------------------------------------------------
# Prediction
# ------------------------------------------------------------------
if submitted:
    new_student = pd.DataFrame([{
        "study_hours_per_week": study_hours,
        "attendance_percent": attendance,
        "previous_score": previous_score,
        "sleep_hours": sleep_hours,
        "parental_education": parental_education,
        "internet_access": internet_access,
        "extracurricular": extracurricular,
    }])

    new_encoded = pd.get_dummies(
        new_student, columns=["parental_education", "internet_access", "extracurricular"]
    )
    new_encoded = new_encoded.reindex(columns=feature_cols, fill_value=0)

    predicted_score = reg_model.predict(new_encoded)[0]
    predicted_passfail = le_pf.inverse_transform(clf_passfail.predict(new_encoded))[0]
    predicted_grade = le_grade.inverse_transform(clf_grade.predict(new_encoded))[0]

    st.subheader("Results")
    c1, c2, c3 = st.columns(3)
    c1.metric("Predicted Score", f"{predicted_score:.1f} / 100")
    c2.metric("Pass / Fail", predicted_passfail)
    c3.metric("Grade", predicted_grade)

    if predicted_passfail == "Fail":
        st.warning("This student is predicted to fail. Consider more study time or attendance support.")
    else:
        st.success("This student is predicted to pass.")