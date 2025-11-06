import pickle
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

with open("chunks.pkl", "rb") as f:
    chunks = pickle.load(f)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vectorstore = FAISS.from_documents(chunks, embeddings)

vectorstore.save_local("faiss_store")

print("Vector store created and saved.")
