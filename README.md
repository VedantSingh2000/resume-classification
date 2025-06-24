# 🤖 Gemini-Powered Resume Analyzer

A smart Streamlit web app that classifies resumes into job categories using a trained SVM model and provides AI-driven insights powered by **Google's Gemini Pro API**.

---

## ✨ Features

- 🎯 **Job Category Prediction**: Identifies roles such as React Developer, Workday, Peoplesoft, etc.  
- 🧠 **Gemini-AI Insights**:
  - Professional summary  
  - Estimated years of experience  
  - Resume rating (1–10)  
  - Constructive feedback  
- 📄 Supports **PDF** and **DOCX** uploads  
- 🔐 Secure API key handling via `.streamlit/secrets.toml` or manual input

---

## ⚙️ Architecture

1. **Offline Training** (`w2_gemini_summary+features.ipynb`):
   - Cleans text, applies TF-IDF  
   - Uses Gemini for early feature prototype  
   - Trains a Linear SVM classifier  
   - Saves: `svm_model_pipeline.joblib`, `tfidf_vectorizer.joblib`, `label_encoder.joblib`  

2. **Deployment** (`app.py`, `utils.py`):
   - Loads saved model and vectorizer  
   - Extracts and cleans text from uploaded resume  
   - Uses Gemini API for analysis (summary, experience, rating, feedback)  
   - Predicts job category via SVM  
   - Displays results in Streamlit UI

---

## 🧰 Tech Stack

- **Frontend**: Streamlit  
- **ML / NLP**: Scikit-learn, NLTK, Pandas  
- **LLM**: Google Gemini Pro  
- **Parsing**: PyPDF2, python-docx  
- **Persistence**: Joblib  

---
## 📁 Repository Structure
resume-classification/
├── app.py
├── utils.py
├── requirements.txt
├── w2_gemini_summary+features.ipynb
├── saved_models/
│   ├── svm_model_pipeline.joblib
│   ├── tfidf_vectorizer.joblib
│   └── label_encoder.joblib
└── .streamlit/
    └── secrets.toml

## 🚀 Quick Start

```bash
git clone https://github.com/VedantSingh2000/resume-classification.git
cd resume-classification
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
pip install -r requirements.txt






