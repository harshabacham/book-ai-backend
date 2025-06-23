import os
import PyPDF2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class AIEngine:
    def __init__(self, data_folder="data"):
        self.data_folder = data_folder
        self.subject_chunks = {}  # Dictionary for subject-wise chunks
        self.load_all_subjects()

    def load_all_subjects(self):
        if not os.path.exists(self.data_folder):
            print(f"[WARNING] Data folder '{self.data_folder}' not found.")
            return

        for subject in os.listdir(self.data_folder):
            subject_path = os.path.join(self.data_folder, subject)
            if os.path.isdir(subject_path):
                chunks = []
                for file in os.listdir(subject_path):
                    if file.endswith(".pdf"):
                        pdf_path = os.path.join(subject_path, file)
                        print(f"[INFO] Loading PDF: {pdf_path}")
                        text = self.extract_text_from_pdf(pdf_path)
                        chunks.extend(self.split_text(text))
                self.subject_chunks[subject.lower()] = chunks
                print(f"[INFO] Loaded {len(chunks)} chunks for subject: {subject}")

    def extract_text_from_pdf(self, file_path):
        text = ""
        with open(file_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"
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
        subject = subject.lower()
        if subject not in self.subject_chunks or not self.subject_chunks[subject]:
            return f"📄 No PDF content found for subject '{subject}'."

        chunks = self.subject_chunks[subject]
        vectorizer = TfidfVectorizer().fit_transform([question] + chunks)
        cosine_similarities = cosine_similarity(vectorizer[0:1], vectorizer[1:]).flatten()

        top_match_index = cosine_similarities.argmax()
        top_score = cosine_similarities[top_match_index]

        print(f"[DEBUG] Top match score: {top_score:.2f}")

        if top_score < 0.1:
            return "🤖 Sorry, I couldn't find an answer related to your question."

        return chunks[top_match_index]
