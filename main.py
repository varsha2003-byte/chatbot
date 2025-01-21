import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Streamlit Header and Sidebar
st.header("My first Chatbot with Hugging Face")
with st.sidebar:
    st.title("Your Documents")
    file = st.file_uploader("Upload a PDF File and start asking questions, please", type="pdf")

text = ""
# Extract text from PDF
if file is not None:
    pdf_reader = PdfReader(file)
    for page in pdf_reader.pages:
        text += page.extract_text()

    # Display extracted text for debugging
    # st.write("Extracted text:", text)

    # Break it into Chunks
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n"],
        chunk_size=200,
        chunk_overlap=150,
        length_function=len,
    )
    chunks = text_splitter.split_text(text)
    # st.write("Text chunks:", chunks)

    # Load Hugging Face Model for Embeddings
    hf_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')  # Replace with your preferred model
    chunk_embeddings = hf_model.encode(chunks)

    # Get user Question
    user_question = st.text_input("Type your question here")

    # Similarity check
    if user_question:
        question_embedding = hf_model.encode([user_question])
        similarities = cosine_similarity(question_embedding, chunk_embeddings)[0]
        top_match_idx = np.argmax(similarities)
        matched_chunk = chunks[top_match_idx]

        # Display matched chunk
        #st.write("Relevant text chunk:", matched_chunk)

        # Generate response using a Hugging Face language model
        # For simplicity, here we just display the matched text chunk as the response
        st.write("Response:", matched_chunk)
