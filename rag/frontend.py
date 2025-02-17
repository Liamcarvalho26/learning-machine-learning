# frontend.py
import streamlit as st
import PyPDF2
import requests

# Create a Streamlit app
st.title('Financial Document Analyzer')

# Create a file uploader widget
uploaded_file = st.file_uploader('Upload a financial document (PDF)')

# Function to extract text from the uploaded PDF file
def extract_text_from_pdf(file):
    pdf_file = PyPDF2.PdfReader(file)
    text = ''
    for page in range(len(pdf_file.pages)):
        text += pdf_file.pages[page].extract_text()
    return text

# Function to send the extracted text to the backend API
def send_text_to_backend(text):
    url = 'http://localhost:8000/analyze'
    payload = {'text': text}
    response = requests.post(url, json=payload)
    return response.json()

# Function to display the answer from the backend API
def display_answer(answer):
    st.write('Answer:')
    st.write(answer)

# Main logic
if uploaded_file is not None:
    text = extract_text_from_pdf(uploaded_file)
    answer = send_text_to_backend(text)
    display_answer(answer)