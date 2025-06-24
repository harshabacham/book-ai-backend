import os
import PyPDF2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class AIEngine:
    def __init__(self, data_folder="data"):
        self.data_folder = data_folder
        self.subject_chunks = {}  # { subject_name: [chunks] }

    def load_subject(self, subject):
        folder_path = os.path.join(self.data_folder, subject)
        if not os.path.exists(folder_path):
            print(f"[WARNING] No folder found for subject '{subject}'")
            return

        chunks = []
        for file in os.listdir(folder_path):
            if file.endswith(".pdf"):
                path = os.path.join(folder_path, file)
                text = self.extract_text_from_pdf(path)
                chunks.extend(self.split_text(text))

        self.subject_chunks[subject] = chunks
        print(f"[INFO] Loaded {len(chunks)} chunks for subject '{subject}'")

    def extract_text_from_pdf(self, file_path):
        text = ""
        with open(file_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                text += page.extract_text() or ""
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
        if subject not in self.subject_chunks:
            self.load_subject(subject)

        chunks = self.subject_chunks.get(subject, [])
        if not chunks:
            return f"📄 No PDF content found for subject '{subject}'."

        vectorizer = TfidfVectorizer().fit_transform([question] + chunks)
        cosine_similarities = cosine_similarity(vectorizer[0:1], vectorizer[1:]).flatten()

        top_index = cosine_similarities.argmax()
        top_score = cosine_similarities[top_index]

        if top_score < 0.1:  # lowered threshold
            return "🤖 Sorry, I couldn't find an answer related to your question."

        return chunks[top_index]
