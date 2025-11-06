import os
from docx import Document

def load_documents(folder_path):
    all_texts = []
    for file in os.listdir(folder_path):
        if file.endswith(".pdf"):
            with fitz.open(os.path.join(folder_path, file)) as doc:
                text = "".join(page.get_text() for page in doc)
                all_texts.append(text)
        elif file.endswith(".docx"):
            doc = Document(os.path.join(folder_path, file))
            text = "\n".join([para.text for para in doc.paragraphs])
            all_texts.append(text)
    return all_texts
documents = load_documents("C:/Users/taniy/Desktop/documents")
print(f"Total documents loaded: {len(documents)}")
print("Type of first document:", type(documents[0]))
print("First document:", documents[0])


