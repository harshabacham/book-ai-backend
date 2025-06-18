import os
import PyPDF2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class AIEngine:
    def __init__(self, data_folder="data"):
        self.documents = []
        self.chunks = []
        self.load_data(data_folder)

    def load_data(self, folder_path):
        self.documents.clear()
        self.chunks.clear()
        for file in os.listdir(folder_path):
            if file.endswith(".pdf"):
                pdf_path = os.path.join(folder_path, file)
                text = self.extract_text_from_pdf(pdf_path)
                chunks = self.split_text(text)
                self.chunks.extend(chunks)

    def extract_text_from_pdf(self, file_path):
        text = ""
        with open(file_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                text += page.extract_text() + "\n"
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

    def get_answer(self, question: str) -> str:
        if not self.chunks:
            return "📄 No PDF content found to search from."

        vectorizer = TfidfVectorizer().fit_transform([question] + self.chunks)
        cosine_similarities = cosine_similarity(vectorizer[0:1], vectorizer[1:]).flatten()

        top_match_index = cosine_similarities.argmax()
        top_score = cosine_similarities[top_match_index]

        if top_score < 0.2:
            return "🤖 Sorry, I couldn't find an answer related to your question."

        return self.chunks[top_match_index]
