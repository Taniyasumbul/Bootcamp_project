import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
import torch
import os
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="Policy Document Q&A", layout="centered")
st.title(" Document Q&A Chatbot ")

if "db" not in st.session_state:
    st.session_state.db = None

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file is not None and st.session_state.db is None:
   
    temp_path = "temp.pdf"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.read())

    
    loader = PyPDFLoader(temp_path)
    pages = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(pages)

    # Detect device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load embeddings
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": device}
    )

    # Create FAISS vector DB and store it permanently in session
    st.session_state.db = FAISS.from_documents(docs, embedding_model)
    st.success("✅ Document processed successfully!")

# Question input
user_query = st.text_input("💬 Ask a question from the document:")

if user_query:
    if st.session_state.db is None:
        st.warning("⚠️ Please upload a PDF first.")
    else:
        # Load Gemini LLM
        llm = ChatGoogleGenerativeAI(model="models/gemini-1.5-flash", temperature=0.3)

        # Create RetrievalQA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=st.session_state.db.as_retriever(),
            chain_type="stuff"
        )

        # Get response
        response = qa_chain(user_query)

        # Display JSON response
        st.subheader("📌 JSON Response")
        st.json(response)

