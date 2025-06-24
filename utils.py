# utils.py

import re
import io
import json
import time
import nltk
from docx import Document  # For .docx files
from PyPDF2 import PdfReader # For .pdf files
import google.generativeai as genai
import requests
import google.api_core.exceptions
import http.client
import os # Though os might not be strictly needed now without temp files for textract

# --- NLTK Stopwords Setup ---
try:
    from nltk.corpus import stopwords
    stopwords.words('english')
    STOP_WORDS_SET = set(stopwords.words('english'))
except LookupError:
    nltk.download('stopwords', quiet=True)
    from nltk.corpus import stopwords
    STOP_WORDS_SET = set(stopwords.words('english'))
except Exception as e:
    STOP_WORDS_SET = set()


# --- Text Cleaning Function ---
def clean_resume_text(text):
    if not isinstance(text, str):
        return ""
    text_lower = text.lower()
    text_special_chars_removed = re.sub(r'[^\w\s\+\#\.]', '', text_lower)
    words = text_special_chars_removed.split()
    words = [word for word in words if word not in STOP_WORDS_SET]
    cleaned_text = " ".join(words)
    return cleaned_text.strip()

# --- Text Extraction Functions (Without textract) ---

def extract_text_from_docx(file_bytes):
    """Extracts text from DOCX file bytes."""
    try:
        doc = Document(io.BytesIO(file_bytes))
        return "\\n".join([para.text for para in doc.paragraphs if para.text.strip()])
    except Exception as e:
        return f"Error: Could not process DOCX file ({e})"

def extract_text_from_pdf(file_bytes):
    """Extracts text from PDF file bytes using PyPDF2."""
    text = ""
    try:
        reader = PdfReader(io.BytesIO(file_bytes))
        if not reader.pages: # Check if there are any pages
             return "Error: PDF has no pages or could not be read."
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\\n"
        if text.strip():
            return text.strip()
        else: # PyPDF2 might return None or empty string for image-based PDFs
            return "Error: No text found in PDF. It might be an image-based PDF."
    except Exception as e_pypdf:
        return f"Error: Could not process PDF file ({e_pypdf})"

# --- Master Text Extraction Function (Revised) ---
def extract_text(uploaded_file):
    """
    Extracts text from an uploaded file object (from Streamlit).
    Handles .docx and .pdf formats. Informs user about .doc.
    """
    file_bytes = uploaded_file.getvalue()
    file_name = uploaded_file.name
    text = ""

    if file_name.endswith(".docx"):
        text = extract_text_from_docx(file_bytes)
    elif file_name.endswith(".pdf"):
        text = extract_text_from_pdf(file_bytes)
    elif file_name.endswith(".doc"):
        return "Error: .doc files are not directly supported. Please convert to .docx or .pdf."
    else:
        return "Error: Unsupported file format. Please upload .docx or .pdf."

    # If extraction yielded an empty string but no explicit error was returned by sub-functions
    if not text.strip() and not text.startswith("Error:"):
        return "Error: No text could be extracted from the file. It might be empty or corrupted."
        
    return text

# --- Gemini Extraction Function (adapted from Notebook Cell 5) ---
# (This function remains the same as in the previous Step 2.3, ensure it's here)
def extract_info_with_gemini(resume_text, gemini_model_instance):
    if not gemini_model_instance:
        return {
            "summary": "Gemini model not initialized.",
            "experience_years": "N/A",
            "rating": "N/A",
            "feedback": "Gemini model not initialized."
        }

    MAX_CHARS = 30000
    if len(resume_text) > MAX_CHARS:
        resume_text = resume_text[:MAX_CHARS]

    if not resume_text.strip():
        return {
            "summary": "No text provided to Gemini.",
            "experience_years": "N/A",
            "rating": "N/A",
            "feedback": "No text provided."
        }

    prompt = f"""
    Analyze the following resume text and extract the specified information.
    Return the output strictly as a JSON object with the following four keys: "summary", "experience_years", "rating", "feedback".

    Constraints for each key:
    1.  "summary": A concise summary of the candidate's profile. The summary must be under 100 words.
    2.  "experience_years": Total years of professional experience as a single number (e.g., 5 or 10.5). If not explicitly stated or clearly inferable, return null.
    3.  "rating": Your assessment of the candidate's overall suitability for a general technical role, on a scale of 1 to 10 (1 being not suitable, 10 being highly suitable). Provide only the number (e.g., 7 or 8.5).
    4.  "feedback": Brief feedback (10-15 words) highlighting key strengths or areas for improvement based on the resume.

    Resume Text:
    ---
    {resume_text}
    ---

    Output JSON:
    """
    default_error_result = {
        "summary": "Error during Gemini API call.",
        "experience_years": "N/A",
        "rating": "N/A",
        "feedback": "Error during API call."
    }
    expected_keys = {"summary": None, "experience_years": None, "rating": None, "feedback": None}
    gen_config = genai.types.GenerationConfig(temperature=0.2)
    max_retries = 3
    delay_base = 3

    for attempt in range(max_retries):
        try:
            response = gemini_model_instance.generate_content(prompt, generation_config=gen_config)
            text_response_part = response.text.strip()
            match = re.search(r"```json\s*\n(.*?)\n\s*```", text_response_part, re.DOTALL | re.IGNORECASE)
            json_str = ""
            if match:
                json_str = match.group(1)
            elif text_response_part.startswith('{') and text_response_part.endswith('}'):
                json_str = text_response_part
            else:
                json_match = re.search(r"(\{[\s\S]*\})", text_response_part)
                if json_match:
                    json_str = json_match.group(1)

            if not json_str:
                if attempt < max_retries - 1:
                    time.sleep(delay_base * (2 ** attempt))
                    continue
                return {**default_error_result, "summary": "Could not parse JSON from Gemini."}

            parsed_json = json.loads(json_str)
            result = {k: parsed_json.get(k) for k in expected_keys.keys()}

            for key_to_convert in ["experience_years", "rating"]:
                if result[key_to_convert] is not None:
                    try:
                        result[key_to_convert] = float(result[key_to_convert])
                    except (ValueError, TypeError):
                        result[key_to_convert] = None
            
            time.sleep(1.2)
            return result

        except (requests.exceptions.RequestException,
                google.api_core.exceptions.GoogleAPICallError,
                http.client.RemoteDisconnected) as e:
            if attempt < max_retries - 1:
                time.sleep(delay_base * (2 ** attempt))
            else:
                return {**default_error_result, "summary": f"API Error: {type(e).__name__}"}
        
        except json.JSONDecodeError:
            if attempt < max_retries - 1:
                time.sleep(delay_base * (2 ** attempt))
                continue
            return {**default_error_result, "summary": "Invalid JSON from Gemini."}

        except Exception as e:
            is_safety_block = False
            if hasattr(e, 'response'):
                if hasattr(e.response, 'prompt_feedback') and \
                   e.response.prompt_feedback.block_reason != genai.types.BlockReason.BLOCK_REASON_UNSPECIFIED:
                    is_safety_block = True
                    return {**default_error_result, "summary": f"Blocked by Gemini: {e.response.prompt_feedback.block_reason.name}"}
                if hasattr(e.response, 'candidates'):
                    for cand in e.response.candidates:
                        if cand.finish_reason in [genai.types.FinishReason.SAFETY, genai.types.FinishReason.RECITATION]:
                            is_safety_block = True
                            return {**default_error_result, "summary": f"Blocked by Gemini: {cand.finish_reason.name}"}
            
            if not is_safety_block and attempt < max_retries - 1:
                time.sleep(delay_base * (2 ** attempt))
            elif not is_safety_block:
                return {**default_error_result, "summary": f"Unexpected Error: {type(e).__name__}"}
    
    time.sleep(1.2)
    return {**default_error_result, "summary": "Max retries reached for Gemini API."}