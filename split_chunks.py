

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import pickle
from data_extraction import documents 

document_objs = [Document(page_content=text, metadata={}) for text in documents]

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = splitter.split_documents(document_objs)

with open("chunks.pkl", "wb") as f:
    pickle.dump(chunks, f)

print(f" Done splitting into {len(chunks)} chunks.")
