# app.py

import streamlit as st
import pandas as pd
import joblib
import os
import google.generativeai as genai
import nltk
import re # Still useful for some string operations if needed, though not directly for JSON here

# Import helper functions from utils.py
from utils import clean_resume_text, extract_info_with_gemini, extract_text

# --- Page Configuration (Set this at the very top) ---
st.set_page_config(
    page_title="Resume Analyzer AI",
    page_icon="📄",
    layout="wide"
)

# --- Download NLTK Stopwords ---
@st.cache_resource
def download_nltk_stopwords():
    try:
        nltk.data.find('corpora/stopwords')
    except nltk.downloader.DownloadError:
        nltk.download('stopwords', quiet=True)

download_nltk_stopwords()

# --- Load Models (TF-IDF, SVM, Label Encoder) ---
MODEL_DIR = "saved_models"
TFIDF_PATH = os.path.join(MODEL_DIR, "tfidf_vectorizer.joblib")
SVM_PATH = os.path.join(MODEL_DIR, "svm_model_pipeline.joblib")
LABEL_ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoder.joblib")

@st.cache_resource
def load_models():
    try:
        tfidf_vectorizer_loaded = joblib.load(TFIDF_PATH)
        svm_model_loaded = joblib.load(SVM_PATH)
        label_encoder_loaded = joblib.load(LABEL_ENCODER_PATH)
        return tfidf_vectorizer_loaded, svm_model_loaded, label_encoder_loaded, True
    except FileNotFoundError:
        st.error(f"Error: One or more model files not found in '{MODEL_DIR}'. "
                 "Ensure 'tfidf_vectorizer.joblib', 'svm_model_pipeline.joblib', "
                 "and 'label_encoder.joblib' are present.")
        return None, None, None, False
    except Exception as e:
        st.error(f"An error occurred while loading the models: {e}")
        return None, None, None, False

tfidf_vectorizer, svm_model, label_encoder, models_successfully_loaded = load_models()

# Initialize Gemini model instance in session state
if 'gemini_model_instance' not in st.session_state:
    st.session_state.gemini_model_instance = None
if 'current_key_used_for_gemini' not in st.session_state:
    st.session_state.current_key_used_for_gemini = None
if 'api_key_input_value' not in st.session_state:
    st.session_state.api_key_input_value = ""


# --- Streamlit UI Elements ---
st.title("📄 Resume Analyzer AI ✨")

# Use columns for a cleaner input layout
col_api, col_upload = st.columns([2, 3]) # API key input takes 2/5ths, uploader 3/5ths

with col_api:
    st.markdown("#### 🔑 API Configuration")
    try:
        default_api_key = st.secrets.get("GOOGLE_API_KEY", "")
    except (FileNotFoundError, AttributeError):
        default_api_key = ""

    api_key_input = st.text_input(
        "Google Gemini API Key",
        value=default_api_key if default_api_key else st.session_state.api_key_input_value,
        type="password",
        help="Your API key is used for Gemini analysis. Not stored after session."
    )
    st.session_state.api_key_input_value = api_key_input

with col_upload:
    st.markdown("#### 📤 Upload Resume")
    uploaded_file = st.file_uploader(
        "Upload your resume (.docx or .pdf):", # Main label
        type=["docx", "pdf"],
        accept_multiple_files=False,
        label_visibility="collapsed" # Hides the default "Upload your resume file here:"
    )
    st.caption("*Note: .doc files are not directly supported. Please convert to .docx or .pdf.*")


analyze_button = st.button(
    "🚀 Analyze Resume",
    type="primary",
    use_container_width=True,
    disabled=not models_successfully_loaded
)

if not models_successfully_loaded and not analyze_button: # Show only if button not pressed yet
    st.warning("⏳ Models are loading or failed to load. Please wait or check error messages. Analysis button is disabled if loading failed.")

st.markdown("---")

# Initialize variables to hold results before the button is pressed
predicted_category_svm = "N/A"
gemini_results = {
    "summary": "N/A", "experience_years": "N/A",
    "rating": "N/A", "feedback": "N/A"
}
raw_text_display = ""
cleaned_text_display = ""

# --- Main Processing Logic (when button is clicked) ---
if analyze_button:
    if not uploaded_file:
        st.warning("⚠️ Please upload a resume file first.")
        st.stop()
    if not api_key_input:
        st.warning("⚠️ Please enter your Google Gemini API Key.")
        st.stop()
    if not models_successfully_loaded:
        st.error("🚫 SVM models could not be loaded. Analysis cannot proceed.")
        st.stop()

    # Configure Gemini model
    current_key_used_for_gemini = st.session_state.get("current_key_used_for_gemini", None)
    if st.session_state.gemini_model_instance is None or api_key_input != current_key_used_for_gemini:
        try:
            with st.spinner("⚙️ Configuring Gemini AI..."):
                genai.configure(api_key=api_key_input)
                st.session_state.gemini_model_instance = genai.GenerativeModel('gemini-1.5-flash-latest')
                st.session_state.current_key_used_for_gemini = api_key_input
        except Exception as e:
            st.error(f"🚫 Error configuring Gemini API: {e}. Please check your API key.")
            st.session_state.gemini_model_instance = None
            st.stop()

    with st.spinner(f"🤖 Analyzing '{uploaded_file.name}'... This may take a moment."):
        raw_text = extract_text(uploaded_file)
        if raw_text.startswith("Error:"):
            st.error(raw_text)
            st.stop()
        if not raw_text.strip():
            st.error("⚠️ Could not extract any text from the uploaded file.")
            st.stop()
        raw_text_display = raw_text

        cleaned_text = clean_resume_text(raw_text)
        if not cleaned_text.strip():
            st.warning("⚠️ After cleaning, the resume text is empty. Results might be limited.")
        cleaned_text_display = cleaned_text

        if st.session_state.gemini_model_instance:
            if cleaned_text.strip():
                gemini_results_api = extract_info_with_gemini(cleaned_text, st.session_state.gemini_model_instance)
                gemini_results.update(gemini_results_api)
            else:
                gemini_results["summary"] = "No text to analyze with Gemini after cleaning."
                gemini_results["feedback"] = "No text to analyze with Gemini after cleaning."
        else:
            gemini_results["summary"] = "Gemini model not available."
            gemini_results["feedback"] = "Gemini analysis skipped."

        if svm_model and tfidf_vectorizer and label_encoder:
            if cleaned_text.strip():
                try:
                    prediction_text_array = svm_model.predict([cleaned_text])
                    predicted_category_svm = prediction_text_array[0].title()
                except Exception as e_svm:
                    st.error(f"Error during SVM prediction: {e_svm}")
                    predicted_category_svm = "Error during SVM prediction"
            else:
                predicted_category_svm = "No text for SVM prediction."
        else:
            predicted_category_svm = "SVM Model components not available."
    
    st.balloons()

# --- Display Results ---
if models_successfully_loaded:
    # Only show results section if analysis was attempted (button pressed and prerequisites met)
    if analyze_button and uploaded_file and api_key_input:
        st.subheader("✨ Analysis Results")
        st.markdown("---")

        res_col1, res_col2 = st.columns(2)

        with res_col1:
            st.markdown(f"##### 🎯 Predicted Category (SVM)")
            st.success(f"**{predicted_category_svm}**")
            
            rating_val = gemini_results.get('rating')
            st.markdown(f"##### ⭐ Gemini Rating (1-10)")
            if isinstance(rating_val, (int, float)):
                # Normalize rating to 0-1 for progress bar if it's 1-10
                # Ensure rating_val is not None and is a number before division
                try:
                    progress_val = float(rating_val) / 10.0
                    st.progress(min(max(progress_val, 0.0), 1.0)) # Clamp between 0 and 1
                    st.metric(label="Rating", value=f"{rating_val}/10")
                except (TypeError, ValueError): # Handle if rating_val is not convertible to float
                     st.metric(label="Rating", value="N/A")
            else:
                st.metric(label="Rating", value=str(rating_val if rating_val is not None else "N/A"))

        with res_col2:
            exp_val = gemini_results.get('experience_years')
            st.markdown(f"##### 📅 Experience (Years - Gemini)")
            st.metric(label="Years of Experience", value=str(exp_val if exp_val is not None else "N/A"))
        
        st.markdown("---")

        st.markdown("#### 📝 Gemini Summary")
        summary_text = gemini_results.get('summary', 'Summary not available or analysis failed.')
        if summary_text and not summary_text.startswith("N/A") and not summary_text.startswith("Error") and "not initialized" not in summary_text and "not available" not in summary_text:
            st.info(f"{summary_text}")
        else:
            st.warning(summary_text)

        st.markdown("#### 💡 Gemini Feedback")
        feedback_text = gemini_results.get('feedback', 'Feedback not available or analysis failed.')
        if feedback_text and not feedback_text.startswith("N/A") and not feedback_text.startswith("Error") and "not initialized" not in feedback_text and "not available" not in feedback_text:
            st.info(f"{feedback_text}")
        else:
            st.warning(feedback_text)
        
        st.markdown("---")

        if raw_text_display or cleaned_text_display:
            expander_col1, expander_col2 = st.columns(2)
            with expander_col1:
                with st.expander("📄 View Raw Extracted Text"):
                    st.text_area("Raw Text Content", raw_text_display, height=250, key="raw_text_area_display")
            with expander_col2:
                with st.expander("🧹 View Cleaned Text (used for analysis)"):
                    st.text_area("Cleaned Text Content", cleaned_text_display, height=250, key="cleaned_text_area_display")

elif not analyze_button and not models_successfully_loaded:
    pass # Error for model loading is already shown above the button by load_models()