import os
import streamlit as st
import cv2
import numpy as np
import pytesseract
import re
import nltk
import shutil
import zipfile
from pdf2image import convert_from_path
from tempfile import NamedTemporaryFile
from PIL import Image
from pytesseract import Output

# ------------------ Tesseract Path Configuration ------------------
# If Tesseract is installed, you can uncomment the following line and set the correct path.
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# ------------------ NLTK Setup ------------------
# Ensure the 'punkt' tokenizer is available. If the file is corrupt or missing, remove it and download again.
nltk_data_path = os.path.expanduser('~/AppData/Roaming/nltk_data/tokenizers/punkt.zip')
try:
    nltk.data.find('tokenizers/punkt')
except (OSError, zipfile.BadZipFile):
    if os.path.exists(nltk_data_path):
        os.remove(nltk_data_path)
    punkt_folder = os.path.expanduser('~/AppData/Roaming/nltk_data/tokenizers/punkt')
    if os.path.exists(punkt_folder):
        shutil.rmtree(punkt_folder)
    nltk.download('punkt')


# ------------------ OCR & Preprocessing ------------------
def preprocess_image(image: np.ndarray) -> np.ndarray:
    """Convert image to grayscale, blur, threshold, and apply morphological opening for OCR enhancement."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    return opened


def extract_text_blocks(image: np.ndarray, confidence_threshold: int = 60) -> str:
    """
    Try to extract text using Tesseract OCR.
    If Tesseract is not found, fall back to EasyOCR.
    """
    try:
        data = pytesseract.image_to_data(image, output_type=Output.DICT, config="--psm 6")
        text_builder = [
            word for i, word in enumerate(data['text'])
            if word.strip() and int(data['conf'][i]) >= confidence_threshold
        ]
        return " ".join(text_builder)
    except pytesseract.pytesseract.TesseractNotFoundError:
        try:
            import easyocr
            reader = easyocr.Reader(['en'])
            results = reader.readtext(image)
            text = " ".join([res[1] for res in results])
            return text
        except ImportError:
            return "Error: Tesseract is not installed and EasyOCR fallback is unavailable."
        except Exception as e:
            return f"Error in fallback OCR: {e}"


def extract_text_from_image(image: np.ndarray) -> str:
    """Preprocess an image and extract text."""
    processed = preprocess_image(image)
    return extract_text_blocks(processed, confidence_threshold=60)


def extract_text_from_pdf(file_path: str) -> str:
    """Convert a PDF to images, then extract text from each page."""
    try:
        images = convert_from_path(file_path)
    except Exception as e:
        return f"Error converting PDF: {str(e)}"

    full_text = [
        extract_text_from_image(np.array(img.convert('RGB'))[:, :, ::-1])
        for img in images
    ]
    return "\n".join(full_text)


# ------------------ Information Extraction ------------------
def extract_person_name_nltk(text: str) -> str:
    """Extract a person's name using NLTK's Named Entity Recognition."""
    try:
        tokens = nltk.word_tokenize(text)
        pos_tags = nltk.pos_tag(tokens)
        chunks = nltk.ne_chunk(pos_tags, binary=False)
        for chunk in chunks:
            if hasattr(chunk, 'label') and chunk.label() == 'PERSON':
                return " ".join(leaf[0] for leaf in chunk.leaves())
    except Exception:
        return None
    return None


def extract_personal_info(text: str) -> dict:
    """
    Extract structured information (Name, DOB, Gender, Aadhaar-like ID, PAN)
    using a combination of NLTK and regex.
    """
    data = {'name': None, 'dob': None, 'gender': None, 'identifier': None, 'pan': None}
    data['name'] = extract_person_name_nltk(text)
    if match := re.search(r'(?i)\b(male|female|transgender)\b', text):
        data['gender'] = match.group(1).title()
    if match := re.search(r'(\d{4}\s?\d{4}\s?\d{4})', text):
        data['identifier'] = match.group(1)
    if match := re.search(r'\b([A-Z]{5}\d{4}[A-Z])\b', text):
        data['pan'] = match.group(1)
    if match := re.search(r'(\d{2}[-/]\d{2}[-/]\d{4}|\d{2}\s[A-Za-z]{3}\s\d{4}|\d{1,2}[-/]\d{1,2}[-/]\d{2,4})', text):
        data['dob'] = match.group(1)
    return data


def load_image(uploaded_file) -> np.ndarray:
    """Convert an uploaded file to an OpenCV image."""
    try:
        return cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)
    except Exception:
        return None


# ------------------ Streamlit UI ------------------
st.set_page_config(
    page_title="High-Precision OCR for Aadhaar/PAN Documents (NLTK-based)",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for styling
st.markdown(
    """
    <style>
    body {
        color: #e0e0e0; /* Light gray/almost white text */
        background-color: #121212; /* Dark gray background */
    }
    .big-font {
        font-size:20px !important;
        color:#81D4FA; /* A lighter blue color for big font */
    }
    .reportview-container .main .block-container{
        padding-top: 10px;
        padding-left: 50px;
        padding-right: 50px;
    }
    .stSidebar {
        background-color: #212121; /* Darker gray background for sidebar */
        color: #e0e0e0;
    }

    /* CSS for scrollable About section */
    .about-scrollable {
      max-height: 400px; /* Adjust as needed */
      overflow-y: auto;
      padding: 10px;
      border: 1px solid #424242; /* Darker border */
      border-radius: 5px;
      background-color: #303030; /* Slightly lighter than main background */
    }
    h1, h2, h3, h4, h5, h6 {
        color: #FFFFFF; /* White headings */
    }
     /* Improve text readability by increasing contrast */
    .streamlit-expanderContent {
        color: #bdbdbd; /* Lighter text color for readability */
    }

    </style>
    """,
    unsafe_allow_html=True,
)

# --- Animations ---
# Add a subtle animation using CSS keyframes
st.markdown(
    """
    <style>
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }

    .animated-title {
        animation: fadeIn 2s;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Animated Title ---
st.markdown("<h1 class='animated-title'>High-Precision OCR for Aadhaar/PAN Documents (NLTK-based)</h1>", unsafe_allow_html=True)

# --- Sidebar for About Section ---
with st.sidebar:
    st.header("ℹ️ About this App")
    about_content = """
        This application is designed to extract information from Aadhaar and PAN card images and PDF documents using Optical Character Recognition (OCR).

        It leverages the following technologies:

        - **Tesseract OCR:** For initial text extraction from images.
        - **EasyOCR (Fallback):** Used if Tesseract is not available or fails.
        - **NLTK (Natural Language Toolkit):** For advanced information extraction, specifically Named Entity Recognition (NER) to identify names.

        **Key Features:**

        - **Versatile Input:** Processes both images (PNG, JPG, JPEG) and PDF documents.
        - **Automated Text Extraction:** Uses OCR to extract text from documents.
        - **Intelligent Information Extraction:** Employs NLTK to identify and extract key information such as name, date of birth, gender, Aadhaar number, and PAN card number.

        **How It Works:**

        1.  Upload an image or PDF of an Aadhaar or PAN card.
        2.  The application uses OCR to convert the image into text.
        3.  NLTK is then used to identify and extract relevant information from the text.
        4.  The extracted information is displayed in a structured format.
        """
    st.write(about_content)

# --- Steps Section ---
with st.expander("🚀 How to Use"):
    st.markdown(
        """
        Follow these simple steps to extract information from your documents:

        1.  **Upload a File:** Use the file uploader below to select an image (PNG, JPG, JPEG) or a PDF document.
        2.  **View Extracted Text:** The extracted text from the document will be displayed in the "Extracted Text" section.
        3.  **See Final Extracted Information:** The application will attempt to identify and extract specific pieces of information, such as the document holder's name, date of birth, gender, Aadhaar number, and PAN card number. The extracted information will be displayed in the "Final Extracted Information" section.
        """
    )

# --- File Uploader ---
st.subheader("1. Upload Document")
uploaded_file = st.file_uploader("Choose a file", type=["png", "jpg", "jpeg", "pdf"])

if uploaded_file:
    st.write(f"File Details: *{uploaded_file.name}* ({uploaded_file.type}, {uploaded_file.size / 1024:.2f} KB)")
    extracted_text = ""

    # --- Determine File Type and Process Accordingly ---
    if uploaded_file.type == "application/pdf":
        with NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            extracted_text = extract_text_from_pdf(tmp.name)
        os.remove(tmp.name)  # Clean up the temp file
    else:
        image = load_image(uploaded_file)
        if image is not None:
            st.subheader("2. Uploaded Image")
            st.image(image, channels="BGR", caption="Uploaded Image", use_container_width=True)
            extracted_text = extract_text_from_image(image)

    # --- Display Extracted Text ---
    st.subheader("3. Extracted Text")
    st.info(extracted_text)

    # --- Extract and Display Personal Information ---
    st.subheader("4. Final Extracted Information")
    info = extract_personal_info(extracted_text)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f'<p class="big-font">*Name:* <span style="color:#81D4FA">{info["name"] or "Not found"}</span></p>',
                    unsafe_allow_html=True)
        st.markdown(f'<p class="big-font">*Gender:* <span style="color:#81D4FA">{info["gender"] or "Not found"}</span></p>',
                    unsafe_allow_html=True)

    with col2:
        st.markdown(f'<p class="big-font">*DOB:* <span style="color:#81D4FA">{info["dob"] or "Not found"}</span></p>',
                    unsafe_allow_html=True)
        st.markdown(f'<p class="big-font">*PAN:* <span style="color:#81D4FA">{info["pan"] or "Not found"}</span></p>',
                    unsafe_allow_html=True)
        st.markdown(
            f'<p class="big-font">*Identifier:* <span style="color:#81D4FA">{info["identifier"] or "Not found"}</span></p>',
            unsafe_allow_html=True)