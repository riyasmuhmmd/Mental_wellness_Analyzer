import streamlit as st
import pandas as pd
import joblib
import re
import time
from datetime import datetime
import keras
from mongo import save_to_mongodb  # MongoDB utility

# Streamlit page configuration
st.set_page_config(
    page_title="Mind & Body Wellness Analyzer",
    page_icon="🌟",
    layout="wide",
)




# Load models and preprocessor
knn_model = joblib.load('KNeighborsClassifier_model.pkl')  # Academic depression prediction
rf_model = joblib.load('Kneighbour.pkl')  # Dietary habits model
preprocessor = joblib.load('preprocessor (6).pkl')
sentiment_model = joblib.load('model21.pkl')  # Mental thoughts model
vectorizer = joblib.load('token.pkl')

data_file = 'Student Depression Dataset.csv'
data = pd.read_csv(data_file)
degree_options = sorted(data['Degree'].unique())

dietary_habits_encoding = {'Healthy': 0, 'Moderate': 1, 'Normal': 2, 'Unhealthy': 3}
inverse_dietary_habits_encoding = {v: k for k, v in dietary_habits_encoding.items()}

def clean_text(text):
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\@\w+|\#', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text.lower()

def save_to_excel(data, filename="user_data.xlsx"):
    try:
        df = pd.DataFrame([data])
        try:
            existing_data = pd.read_excel(filename)
            df = pd.concat([existing_data, df], ignore_index=True)
        except FileNotFoundError:
            pass
        df.to_excel(filename, index=False)
        st.success("✅ User data saved successfully to Excel!")
    except Exception as e:
        st.error(f"An error occurred while saving to Excel: {e}")

def intro_page():
    st.markdown(
        """
        <style>
        h1:hover, h2:hover {
            background-color: #d6eaf8;
            color: #21618c;
            transform: scale(1.1);
            transition: all 0.3s ease-in-out;
            border-radius: 10px;
            padding: 5px;
        }
        h1 {
            color: #2c3e50;
            text-align: center;
        }
        </style>
        <h1>🌟 Welcome to Mind & Body Wellness Analyzer 🌟</h1>
        """, unsafe_allow_html=True)
    st.markdown(
        """<p style='text-align: center; font-size: 1.2em;'>Discover insights into your mental health, dietary habits, and overall well-being. Your journey to wellness begins here!</p>""",
        unsafe_allow_html=True
    )
    if st.button("🚀 Start Analysis"):
        st.session_state['page'] = 'prediction'

def prediction_page():
    st.markdown(
        """
        <style>
        h2:hover, h1:hover {
            background-color: #d6eaf8;
            color: #21618c;
            transform: scale(1.1);
            transition: all 0.3s ease-in-out;
            border-radius: 10px;
            padding: 5px;
        }
        h1 {
            color: #2c3e50;
            text-align: center;
        }
        </style>
        <h1>🌟 Mind & Body Wellness Analyzer 🌟</h1>
        """, unsafe_allow_html=True)

    st.header("📝 Personal Information")
    col1, col2 = st.columns(2)
    with col1:
        name = st.text_input("Full Name")
        gender = st.radio("Gender", ["Male", "Female"], horizontal=True)
        age = st.slider("Age", min_value=10, max_value=100, value=25, step=1)
        cgpa = st.slider("CGPA (0-10 scale)", min_value=0.0, max_value=10.0, value=7.5, step=0.1)
    with col2:
        degree = st.selectbox("Degree", degree_options)
        sleep_duration = st.radio("Sleep Duration", ["5-6 hours", "7-8 hours", "> 8 hours"], horizontal=True)
        work_study_hours = st.slider("Weekly Work/Study Hours", min_value=0, max_value=100, value=40, step=1)

    suicidal_thoughts = st.radio("Have you ever had suicidal thoughts?", ["Yes", "No"], horizontal=True)
    academic_pressure = st.slider("Academic Pressure (1 = Low, 5 = High)", min_value=1, max_value=5, value=3, step=1)
    financial_stress = st.slider("Financial Stress (1 = Low, 5 = High)", min_value=1, max_value=5, value=3, step=1)

    st.header("💬 Share Your Thoughts")
    user_input = st.text_area("Please describe your current feelings, experiences, or thoughts in detail.",
                              placeholder="For example, 'I'm feeling overwhelmed with my upcoming exams.'",
                              help="This text will be analyzed for mental thought insights.")

    if st.button("🔍 Analyze"):
        if not name.strip():
            st.error("Please enter your name.")
        elif not user_input.strip():
            st.error("Please enter some text for analysis.")
        else:
            try:
                with st.spinner('🔄 Analyzing your data...'):
                    time.sleep(2)

                    input_data = {
                        'Name': name,
                        'Gender': gender,
                        'Age': age,
                        'Have you ever had suicidal thoughts ?': suicidal_thoughts,
                        'Academic Pressure': academic_pressure,
                        'CGPA': cgpa,
                        'Financial Stress': financial_stress,
                        'Degree': degree,
                        'Work/Study Hours': work_study_hours,
                        'Sleep Duration': sleep_duration,
                        'User Input': user_input,
                        'Timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }

                    processed_data = preprocessor.transform(pd.DataFrame([input_data]).drop(columns=['User Input', 'Name', 'Timestamp']))

                    academic_depression_prediction = knn_model.predict(processed_data)[0]
                    dietary_habits_prediction = rf_model.predict(processed_data)[0]
                    dietary_habit_label = inverse_dietary_habits_encoding[dietary_habits_prediction]

                    cleaned_input = clean_text(user_input)
                    input_vector = vectorizer.transform([cleaned_input])
                    mental_thought_prediction = sentiment_model.predict(input_vector)[0]

                    input_data['Academic Depression Status'] = "Depressed" if academic_depression_prediction == 1 else "Not Depressed"
                    input_data['Dietary Habit'] = dietary_habit_label
                    input_data['Mental Thoughts'] = mental_thought_prediction.capitalize()

                    save_to_mongodb(input_data)
                    save_to_excel(input_data)

                    st.success("✅ Predictions completed successfully!")

                    # Consolidated recommendations
                    st.markdown(f"<h2>📘 Your Wellness Recommendations:</h2>", unsafe_allow_html=True)

                    # Academic Depression Recommendations
                    if academic_depression_prediction == 1:
                        st.write("### 🌟 Academic Depression: **Depressed**")
                        st.markdown("- **Consider professional counseling.**\n- **Practice mindfulness techniques.**\n- **Maintain a healthy study-life balance.**")
                    else:
                        st.write("### 🌟 Academic Depression: **Not Depressed**")

                    # Dietary Habit Recommendations
                    st.write(f"### 🥗 Dietary Habit: **{dietary_habit_label}**")
                    if dietary_habit_label in ["Unhealthy", "Normal"]:
                        st.markdown("- **Add more fruits and vegetables to your diet.**\n- **Avoid excessive sugar and processed food.**\n- **Drink at least 2 liters of water daily.**")
                    elif dietary_habit_label == "Healthy":
                        st.markdown("- **Keep up the great work with your healthy eating habits!**")

                    # Mental Thoughts Recommendations
                    st.write(f"### 💭 Mental Thoughts: **{mental_thought_prediction.capitalize()}**")
                    if mental_thought_prediction.lower() == "negative":
                        st.markdown("- **Practice journaling to reflect positive thoughts.**\n- **Engage in creative activities like painting or writing.**\n- **Stay physically active through light exercise or gym sessions.**")
            except Exception as e:
                st.error(f"An error occurred: {e}")

def main():
    if 'page' not in st.session_state:
        st.session_state['page'] = 'intro'

    if st.session_state['page'] == 'intro':
        intro_page()
    elif st.session_state['page'] == 'prediction':
        prediction_page()

if __name__ == "__main__":
    main()
