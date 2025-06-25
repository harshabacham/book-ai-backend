# ai_engine.py (Updated with sentence-transformers + faiss)

import os
import PyPDF2
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

class AIEngine:
    def __init__(self, data_folder="data"):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")  # Free and fast
        self.subject_index = {}  # {subject: FAISS index}
        self.subject_chunks = {}  # {subject: [text chunks]}
        self.subject_embeddings = {}  # {subject: [embeddings array]}
        self.load_data(data_folder)

    def load_data(self, folder_path):
        if not os.path.exists(folder_path):
            print(f"[WARNING] Data folder '{folder_path}' not found. Skipping load_data.")
            return

        for subject in os.listdir(folder_path):
            subject_path = os.path.join(folder_path, subject)
            if os.path.isdir(subject_path):
                chunks = []
                for file in os.listdir(subject_path):
                    if file.endswith(".pdf"):
                        pdf_path = os.path.join(subject_path, file)
                        text = self.extract_text_from_pdf(pdf_path)
                        chunks.extend(self.split_text(text))
                if not chunks:
                    continue
                self.subject_chunks[subject] = chunks
                embeddings = self.model.encode(chunks)
                self.subject_embeddings[subject] = embeddings
                index = faiss.IndexFlatL2(embeddings.shape[1])
                index.add(np.array(embeddings))
                self.subject_index[subject] = index

    def extract_text_from_pdf(self, file_path):
        text = ""
        with open(file_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        return text

    def split_text(self, text, chunk_size=300):
        sentences = text.split(". ")
        chunks, current = [], ""
        for sentence in sentences:
            if len(current) + len(sentence) < chunk_size:
                current += sentence + ". "
            else:
                chunks.append(current.strip())
                current = sentence + ". "
        if current:
            chunks.append(current.strip())
        return chunks

    def get_answer(self, question: str, subject: str = "general") -> str:
        subject = subject.lower()
        if subject not in self.subject_index:
            return f"📄 No PDF content found for subject '{subject}'."

        question_embedding = self.model.encode([question])
        D, I = self.subject_index[subject].search(np.array(question_embedding), k=1)
        best_match_index = I[0][0]
        score = D[0][0]

        if score > 1.5:  # L2 distance threshold (lower is better)
            return "🤖 Sorry, I couldn't find an answer related to your question."

        return self.subject_chunks[subject][best_match_index]
