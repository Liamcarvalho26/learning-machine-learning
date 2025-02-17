import pinecone
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import streamlit as st
import PyPDF2

# Initialize Pinecone client
pinecone.init(api_key='pcsk_28vEyT_58X5HSfVy8i6VTjjFVyE9uUUEk7DjnjEeFQAacZ9gpKp1jXFFxgnSwM8U4b8Cyx', environment='us-east-1')

#
# Load the LLaMA model and tokenizer
model = AutoModelForCausalLM.from_pretrained('decapoda-research/llama-3b')
tokenizer = AutoTokenizer.from_pretrained('decapoda-research/llama-3b')

# Create a Streamlit app
st.title('Financial Document Analyzer')

# Create a file uploader widget
uploaded_file = st.file_uploader('Upload a financial document')

# Extract text from the uploaded document
if uploaded_file is not None:
    pdf_file = PyPDF2.PdfFileReader(uploaded_file)
    text = ''
    for page in range(pdf_file.numPages):
        text += pdf_file.getPage(page).extractText()

    # Preprocess the extracted text
    text = text.lower()
    text = text.replace('\n', ' ')

    # Encode the text using LLaMA 3.2
    inputs = tokenizer(text, return_tensors='pt')
    outputs = model(**inputs)
    vector = outputs.last_hidden_state[:, 0, :]

    # Index the vector in Pinecone
    index_name = 'financial-documents'
    pinecone.Index(index_name).upsert(vectors=[vector], metadata=[{'text': text}])

    # Retrieve answers from Pinecone
    query = st.text_input('Enter a question or query')
    if query:
        query_vector = model.encode(query)
        results = pinecone.Index(index_name).query(vectors=[query_vector], top_k=5)
        for result in results.matches:
            st.write(result.metadata['text'])