import os
import PyPDF2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class AIEngine:
    def __init__(self, data_folder="data"):
        self.subject_chunks = {}
        self.load_data(data_folder)

    def load_data(self, folder_path):
        if not os.path.exists(folder_path):
            print(f"[WARNING] Data folder '{folder_path}' not found. Skipping load_data.")
            return

        for subject in os.listdir(folder_path):
            subject_path = os.path.join(folder_path, subject)
            if os.path.isdir(subject_path):
                all_chunks = []
                for file in os.listdir(subject_path):
                    if file.endswith(".pdf"):
                        pdf_path = os.path.join(subject_path, file)
                        text = self.extract_text_from_pdf(pdf_path)
                        chunks = self.split_text(text)
                        all_chunks.extend(chunks)
                self.subject_chunks[subject.lower()] = all_chunks

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
        subject_chunks = self.subject_chunks.get(subject.lower())
        if not subject_chunks:
            return f"📄 No PDF content found for subject '{subject}'."

        vectorizer = TfidfVectorizer().fit_transform([question] + subject_chunks)
        cosine_similarities = cosine_similarity(vectorizer[0:1], vectorizer[1:]).flatten()
        top_match_index = cosine_similarities.argmax()
        top_score = cosine_similarities[top_match_index]

        if top_score < 0.15:
            return "🤖 Sorry, I couldn't find an answer related to your question."

        return subject_chunks[top_match_index]
