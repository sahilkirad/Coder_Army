High-Precision OCR for Aadhaar/PAN Documents (NLTK-based)

📝Overview
---------------------------------------------------------------------------------------------------------------------
This application extracts structured information from Aadhaar and PAN card images or PDFs using Optical Character Recognition (OCR). It is built using Streamlit and leverages Tesseract OCR, EasyOCR (as fallback), NLTK Named Entity Recognition (NER), and Regex-based pattern matching to extract:

📌 Name

📆 Date of Birth (DOB)

⚧ Gender

🔢 Aadhaar Number

🆔 PAN Number


✨Features
-------------------------------------------------------------------------------------------------------------------------
📂 Supports Image & PDF Input: Upload JPG, PNG, JPEG, or PDF files.

🔍 OCR for Text Extraction: Uses Tesseract OCR (or EasyOCR as fallback).

🛠 Preprocessing for Better Accuracy: Grayscale conversion, blurring, thresholding, and noise removal.

🤖 Information Extraction: Uses NLTK for Name Extraction and Regex for DOB, Gender, Aadhaar, and PAN Number.

🎨 User-friendly Streamlit UI: Dark mode styling, sidebar, and structured results display.



📌Installation
------------------------------------------------------------------------------------------------------------------------
📦 Prerequisites

Ensure Python (>=3.8) and the required libraries are installed:

pip install streamlit numpy opencv-python pytesseract pdf2image nltk easyocr Pillow

🔧 Additional Dependencies

🖥 Tesseract OCR: Install Tesseract on your system and configure the path in the script if necessary.

📚 NLTK Data: The script ensures punkt tokenizer is available for text processing.



🚀How to Use
-----------------------------------------------------------------------------------------------------------------------
🔄 Run the Streamlit app:

streamlit run app.py

📤 Upload an Aadhaar or PAN card image/PDF.

📑 View the extracted text and structured personal details.

📂 Folder Structure
project-directory/
│-- app.py  # Main Streamlit application
│-- requirements.txt  # Dependencies
│-- README.md  # Project Documentation


⚙️ Configuration
------------------------------------------------------------------------------------------------------------------------
If Tesseract OCR is installed, update the path in app.py:

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


📜License
-------------------------------------------------------------------------------------------------------------------------
This project is open-source and free to use for educational and research purposes.

