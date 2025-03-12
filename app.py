import streamlit as st
from modules.text_preprocessing import preprocess_text
from modules.skill_extraction import extract_skills_from_cv
from modules.chatbot import get_resume_feedback
import joblib
from pdfminer.high_level import extract_text

# Load ML Model & Vectorizer
vectorizer = joblib.load("model/tfidf_vectorizer.pkl")
model = joblib.load("model/gbc.pkl")  

# Streamlit UI
st.title("📄 CV Analyzer App")
st.sidebar.header("Select Feature")

# Select feature (Category Prediction, Skill Extraction, Chatbot)
feature = st.sidebar.radio("Choose a feature:", ["Job Category Prediction", "Skill Extraction", "AI Chatbot"])

# File Upload Section (Shared for All Features)
uploaded_file = st.file_uploader("Upload Your CV (PDF only)", type=["pdf"])

if uploaded_file:
    # Extract text from PDF
    text = extract_text(uploaded_file)

    if feature == "Job Category Prediction":
        processed_text = preprocess_text(text)
        text_vectorized = vectorizer.transform([processed_text])
        prediction = model.predict(text_vectorized)[0]

        st.subheader("Predicted Job Category:")
        st.write(f"💼 {prediction}")

    elif feature == "Skill Extraction":
        with open("temp_resume.pdf", "wb") as f:
            f.write(uploaded_file.getbuffer())

        skills = extract_skills_from_cv("temp_resume.pdf")

        st.subheader("Extracted Skills:")
        st.write(skills)

    elif feature == "AI Chatbot":
        st.subheader("💬 Resume Improvement Chatbot")
        user_input = st.text_area("Ask for resume feedback:")

        if st.button("Get Feedback"):
            feedback = get_resume_feedback(user_input)
            st.write(feedback)

