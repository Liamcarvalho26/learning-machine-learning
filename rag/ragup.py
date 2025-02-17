# ragllama.py
import os
import pinecone
from pinecone import Pinecone, ServerlessSpec
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import streamlit as st
import PyPDF2
import requests

# Initialize Pinecone client
pc = Pinecone(
    api_key="pcsk_28vEyT_58X5HSfVy8i6VTjjFVyE9uUUEk7DjnjEeFQAacZ9gpKp1jXFFxgnSwM8U4b8Cyx",
    environment="us-east-1"
)

# Create a Pinecone index if it doesn't exist
if 'financial' not in pc.list_indexes().names():
    pc.create_index(
        name='financial',
        dimension=384,
        metric='cosine',
        spec=ServerlessSpec(
            cloud='aws',
            region='us-east-1'
        )
    )
else:
    print("Index 'financial' already exists.")

# Load the LLaMA model and tokenizer
model = AutoModelForCausalLM.from_pretrained('decapoda-research/llama-3b')
tokenizer = AutoTokenizer.from_pretrained('decapoda-research/llama-3b')

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