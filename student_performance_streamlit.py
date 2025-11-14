import streamlit as st
import pandas as pd
import joblib
import random

# ------------------------
# Page Config
# ------------------------
st.set_page_config(page_title="Student Performance Studio 2.0", page_icon="🎓", layout="wide")

# Custom CSS
st.markdown(
    """
    <style>
        /* Sidebar Styling */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0f2027, #203a43, #2c5364);
            padding-top: 20px;
        }
        .sidebar-title { color: #ffffff; text-align:center; font-size:22px; font-weight:700; }
        
        /* Main Title */
        .main-title { font-size:42px; font-weight:800; color:#1e88e5; text-align:center; padding-bottom:10px; }
        
        /* Cards */
        .predict-card {
            background: #1e88e5;
            padding: 25px;
            border-radius: 15px;
            color: white;
            text-align: center;
            box-shadow: 0px 4px 12px rgba(0,0,0,0.15);
            width: 350px;
            margin:auto;
        }

        /* Footer */
        .footer { text-align:right; font-size:11px; color:#828282; padding-top:10px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------------
# Sidebar
# ------------------------
st.sidebar.markdown("<div class='sidebar-title'>📘 Navigation</div>", unsafe_allow_html=True)
options = ["Live Predictor", "Data Insights", "Leaderboard", "Feedback"]
menu = st.sidebar.radio("", options)
quotes = [
    "Learning today, leading tomorrow.",
    "Small steps every day lead to big results.",
    "Believe in your potential.",
    "Your journey matters!",
]
st.sidebar.info(random.choice(quotes))

# ------------------------
# Load Model
# ------------------------
@st.cache_resource
def load_model():
    model = joblib.load("Student_Performance_Data_Analysis_model.pkl")
    scaler = joblib.load("Student_Performance_Data_Analysis_scaler.pkl")
    selector = joblib.load("Student_Performance_Data_Analysis_selector.pkl")
    return model, scaler, selector

model, scaler, selector = load_model()

# ------------------------
# Live Predictor
# ------------------------
if menu == "Live Predictor":
    st.markdown("<div class='main-title'>🔮 Smart Score Predictor 2.0</div>", unsafe_allow_html=True)
    st.write("Enter details below and get an improved prediction experience with a modern UI ✨.")

    with st.form("predict_form"):
        col1, col2 = st.columns(2)
        with col1:
            fs = {
                'Previous_Sem_Score': st.slider('Previous Sem Score', 0.0, 100.0, 60.0),
                'Study_Hours_per_Week': st.slider('Study Hours/week', 0.0, 40.0, 12.0),
                'Attendance_Percentage': st.slider('Attendance %', 0.0, 100.0, 75.0),
                'Parental_Education': st.selectbox('Parental Education', ['High School', 'Graduate', 'Postgraduate']),
                'Library_Usage_per_Week': st.slider('Library Usage/week', 0, 10, 2),
                'Sleep_Hours': st.slider('Sleep Hours', 2.0, 12.0, 6.0)
            }
        with col2:
            fs.update({
                'Internet_Access': st.selectbox('Internet Access', ['Yes', 'No']),
                'Gender': st.selectbox('Gender', ['Male', 'Female']),
                'Family_Income': st.number_input('Family Income', 5000.0, 100000.0, 30000.0),
                'Tutoring_Classes': st.selectbox('Tutoring Classes', ['Yes', 'No']),
                'Motivation_Level': st.slider('Motivation Level', 1.0, 10.0, 7.0),
                'Test_Anxiety_Level': st.slider('Test Anxiety', 1.0, 10.0, 5.0)
            })

        predict_btn = st.form_submit_button("🎯 Predict Score")

    if predict_btn:
        df = pd.DataFrame([fs])
        df = pd.get_dummies(df)
        req_cols = scaler.feature_names_in_
        for c in req_cols:
            if c not in df.columns:
                df[c] = 0
        df = df[req_cols]

        scaled = scaler.transform(df)
        selected = selector.transform(scaled)
        pred = model.predict(selected)[0]

        c = "#00c853" if pred >= 60 else "#d32f2f"
        st.markdown(f"<div class='predict-card' style='background:{c};'><h3>Predicted Score</h3><h1>{pred:.2f}</h1></div>", unsafe_allow_html=True)
        st.balloons()

# ------------------------
# Data Insights
# ------------------------
elif menu == "Data Insights":
    st.markdown("<div class='main-title'>📊 Data Insights Dashboard</div>", unsafe_allow_html=True)
    df = pd.read_csv("students_performance_dataset.csv")
    chart_type = st.selectbox("Choose insight:", ["Correlation Heatmap", "Numeric Distribution"])

    import seaborn as sns
    import matplotlib.pyplot as plt

    if chart_type == "Correlation Heatmap":
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.heatmap(df.corr(numeric_only=True), annot=False, cmap="coolwarm", ax=ax)
        st.pyplot(fig)
    else:
        col = st.selectbox("Select Column:", df.select_dtypes(include='number').columns)
        fig, ax = plt.subplots()
        sns.histplot(df[col], kde=True, ax=ax)
        st.pyplot(fig)

# ------------------------
# Leaderboard
# ------------------------
elif menu == "Leaderboard":
    st.markdown("<div class='main-title'>🏆 Champions Leaderboard</div>", unsafe_allow_html=True)
    sample = pd.DataFrame({
        "Student": ["Aarav", "Diya", "Rahul", "Ananya", "Kiran"],
        "Score": [97, 94, 90, 88, 86]
    })
    st.dataframe(sample, use_container_width=True)
    st.success("Aim high! You could be next on the leaderboard.")

# ------------------------
# Feedback
# ------------------------
elif menu == "Feedback":
    st.markdown("<div class='main-title'>✍️ We Value Your Feedback</div>", unsafe_allow_html=True)
    with st.form("fb_form"):
        name = st.text_input("Your Name")
        msg = st.text_area("Your Feedback")
        rate = st.slider("Rate the App", 1, 5, 4)
        send = st.form_submit_button("Send Feedback")
    if send:
        st.success("Thank you for your input!")
        st.snow()

# Footer
st.markdown("<div class='footer'>Designed with ❤️ | Modern Analytics UI | 2025</div>", unsafe_allow_html=True)
