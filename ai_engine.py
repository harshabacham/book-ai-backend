import os
import gc
import numpy as np
from typing import Dict, List, Optional
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pypdf

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AIEngine:
    def __init__(self, data_folder: str = "data"):
        """Initialize with TF-IDF instead of heavy embeddings"""
        self.vectorizers: Dict[str, TfidfVectorizer] = {}
        self.subject_texts: Dict[str, List[str]] = {}
        self.load_data(data_folder)
        logger.info("AI Engine initialized with TF-IDF")

    def load_data(self, folder_path: str) -> None:
        """Load data using memory-efficient approach"""
        if not os.path.exists(folder_path):
            logger.warning(f"Data folder '{folder_path}' not found")
            return

        try:
            for exam in os.listdir(folder_path):
                exam_path = os.path.join(folder_path, exam)
                if os.path.isdir(exam_path):
                    for subject in os.listdir(exam_path):
                        subject_path = os.path.join(exam_path, subject)
                        if os.path.isdir(subject_path):
                            self._process_subject(exam, subject, subject_path)
                            gc.collect()  # Free memory after each subject
        except Exception as e:
            logger.error(f"Error loading data: {e}")

    def _process_subject(self, exam: str, subject: str, subject_path: str) -> None:
        """Process subject documents with TF-IDF"""
        texts = []
        for file in os.listdir(subject_path):
            if file.endswith(".pdf"):
                try:
                    text = self.extract_text_from_pdf(os.path.join(subject_path, file))
                    texts.append(text[:10000])  # Limit text length
                except Exception as e:
                    logger.error(f"Error processing PDF: {e}")

        if not texts:
            return

        key = f"{exam.lower()}_{subject.lower()}"
        self.subject_texts[key] = texts
        
        # Create TF-IDF vectorizer (memory efficient)
        self.vectorizers[key] = TfidfVectorizer(stop_words='english', max_features=5000)
        try:
            self.vectorizers[key].fit_transform(texts)
            logger.info(f"Loaded {len(texts)} documents for {key}")
        except Exception as e:
            logger.error(f"Error creating vectorizer for {key}: {e}")
            self._cleanup_failed_subject(key)

    # [Keep all other methods from previous version except embedding-related code]
    # [Keep extract_text_from_pdf, split_text, etc.]

    def get_answer(self, question: str, exam: Optional[str] = None, subject: Optional[str] = None) -> str:
        """Get answer using lightweight TF-IDF approach"""
        if not question.strip():
            return "Please provide a valid question."
            
        if subject is None:
            return "Subject is required."

        try:
            key = self._get_subject_key(exam, subject)
            
            if key not in self.vectorizers:
                return self._handle_missing_subject(subject)

            # TF-IDF similarity search
            vectorizer = self.vectorizers[key]
            question_vec = vectorizer.transform([question])
            doc_vecs = vectorizer.transform(self.subject_texts[key])
            
            # Find most similar document
            similarities = cosine_similarity(question_vec, doc_vecs)
            best_match_idx = similarities.argmax()
            
            if similarities[0, best_match_idx] < 0.2:  # Similarity threshold
                return "I couldn't find a relevant answer."
                
            return self._format_answer(self.subject_texts[key][best_match_idx])
                
        except Exception as e:
            logger.error(f"Error processing question: {e}")
            return "Sorry, I encountered an error."
        finally:
            gc.collect()

    # [Keep all helper methods]

    def _get_subject_key(self, exam: Optional[str], subject: str) -> str:
        """Generate the lookup key for a subject"""
        return f"{exam.lower()}_{subject.lower()}" if exam else subject.lower()

    def _handle_missing_subject(self, subject: str) -> str:
        """Handle cases where subject isn't found"""
        available = [s.split('_')[-1] for s in self.subject_chunks.keys()]
        return (f"No content found for '{subject}'. "
                f"Available subjects: {', '.join(set(available))}")

    def _get_relevant_chunks(self, key: str, D: np.ndarray, I: np.ndarray) -> List[str]:
        """Extract relevant chunks based on similarity scores"""
        return [
            self.subject_chunks[key][i]
            for i in I[0]
            if i >= 0 and D[0][i] < 1.0  # Strict similarity threshold
        ]

    def _format_answer(self, chunks: List[str]) -> str:
        """Format the answer with length limits"""
        answer = " ".join(chunks[:2])  # Use top 2 chunks
        return (answer[:800] + "... [truncated]") if len(answer) > 800 else answer

    def list_available_subjects(self) -> List[str]:
        """List available subjects without technical keys"""
        return list(set(
            key.split('_')[-1]  # Extract just the subject name
            for key in self.subject_chunks.keys()
        ))