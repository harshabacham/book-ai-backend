import os
import PyPDF2
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class AIEngine:
    def __init__(self, data_folder="data"):
        self.subject_chunks = {}
        self.subject_embeddings = {}
        self.subject_index = {}
        # Initialize the sentence transformer model
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.load_data(data_folder)

    def load_data(self, folder_path):
        if not os.path.exists(folder_path):
            print(f"[WARNING] Data folder '{folder_path}' not found. Skipping load_data.")
            return

        for exam in os.listdir(folder_path):
            exam_path = os.path.join(folder_path, exam)
            if os.path.isdir(exam_path):
                for subject in os.listdir(exam_path):
                    subject_path = os.path.join(exam_path, subject)
                    if os.path.isdir(subject_path):
                        chunks = []
                        for file in os.listdir(subject_path):
                            if file.endswith(".pdf"):
                                pdf_path = os.path.join(subject_path, file)
                                text = self.extract_text_from_pdf(pdf_path)
                                chunks.extend(self.split_text(text))
                        if not chunks:
                            continue
                        key = f"{exam.lower()}_{subject.lower()}"
                        self.subject_chunks[key] = chunks
                        embeddings = self.model.encode(chunks)
                        self.subject_embeddings[key] = embeddings
                        index = faiss.IndexFlatL2(embeddings.shape[1])
                        index.add(np.array(embeddings))
                        self.subject_index[key] = index
                        print(f"Loaded {len(chunks)} chunks for {key}")

    def extract_text_from_pdf(self, file_path):
        text = ""
        try:
            with open(file_path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
        except Exception as e:
            print(f"Error reading PDF {file_path}: {e}")
        return text

    def split_text(self, text, chunk_size=300):
        sentences = text.split(". ")
        chunks, current = [], ""
        for sentence in sentences:
            if len(current) + len(sentence) < chunk_size:
                current += sentence + ". "
            else:
                if current.strip():
                    chunks.append(current.strip())
                current = sentence + ". "
        if current.strip():
            chunks.append(current.strip())
        return chunks

    def get_answer(self, question: str, exam: str = None, subject: str = None) -> str:
        # Handle case where only subject is provided (for backward compatibility)
        if exam is None and subject is not None:
            # Try to find the subject in any exam
            matching_keys = [key for key in self.subject_chunks.keys() if subject.lower() in key]
            if not matching_keys:
                return f"📄 No content found for subject '{subject}'."
            key = matching_keys[0]  # Use the first match
        else:
            key = f"{exam.lower()}_{subject.lower()}" if exam and subject else subject.lower()
        
        if key not in self.subject_index:
            available_subjects = list(self.subject_chunks.keys())
            return f"📄 No content found for '{key}'. Available subjects: {', '.join(available_subjects)}"

        try:
            question_embedding = self.model.encode([question])
            D, I = self.subject_index[key].search(np.array(question_embedding), k=3)
            
            # Get top 3 most relevant chunks
            relevant_chunks = []
            for i in range(min(3, len(I[0]))):
                if D[0][i] < 1.5:  # Similarity threshold
                    relevant_chunks.append(self.subject_chunks[key][I[0][i]])
            
            if not relevant_chunks:
                return "🤖 Sorry, I couldn't find an answer related to your question."
            
            # Combine the most relevant chunks
            answer = " ".join(relevant_chunks[:2])  # Use top 2 chunks
            return answer[:1000] + "..." if len(answer) > 1000 else answer
            
        except Exception as e:
            print(f"Error processing question: {e}")
            return "🤖 Sorry, there was an error processing your question."

    def list_available_subjects(self):
        """Return list of available subjects"""
        return list(self.subject_chunks.keys())