# ai_engine.py
import os
import PyPDF2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class AIEngine:
    def __init__(self, data_folder="data"):
        self.subject_chunks = {}  # key: subject, value: chunks
        self.load_data(data_folder)

    def load_data(self, folder_path):
        if not os.path.exists(folder_path):
            print(f"[WARNING] Data folder '{folder_path}' not found.")
            return

        for subject_folder in os.listdir(folder_path):
            subject_path = os.path.join(folder_path, subject_folder)
            if os.path.isdir(subject_path):
                all_chunks = []
                for file in os.listdir(subject_path):
                    if file.endswith(".pdf"):
                        pdf_path = os.path.join(subject_path, file)
                        text = self.extract_text_from_pdf(pdf_path)
                        chunks = self.split_text(text)
                        all_chunks.extend(chunks)
                self.subject_chunks[subject_folder.lower()] = all_chunks

    def extract_text_from_pdf(self, file_path):
        text = ""
        try:
            with open(file_path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    text += page.extract_text() + "\n"
        except Exception as e:
            print(f"[ERROR] Reading {file_path}: {e}")
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

    def get_answer(self, question: str, subject: str) -> str:
        chunks = self.subject_chunks.get(subject)
        if not chunks:
            return f"📁 No content found for subject: **{subject}**."

        vectorizer = TfidfVectorizer().fit_transform([question] + chunks)
        cosine_similarities = cosine_similarity(vectorizer[0:1], vectorizer[1:]).flatten()

        top_match_index = cosine_similarities.argmax()
        top_score = cosine_similarities[top_match_index]

        if top_score < 0.1:
            return "🤖 Sorry, I couldn't find a relevant answer."

        return chunks[top_match_index]
