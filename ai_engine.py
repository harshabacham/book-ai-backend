import os
import logging
import gc
import numpy as np
from typing import Dict, List, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path
import pypdf

logging.basicConfig(level=os.getenv('LOG_LEVEL', 'INFO'))
logger = logging.getLogger(__name__)

class AIEngine:
    def __init__(self, data_folder: str = None):
        # Initialize all required attributes
        self.vectorizers: Dict[str, TfidfVectorizer] = {}
        self.subject_texts: Dict[str, List[str]] = {}
        self._text_cache: Dict[str, str] = {}
        
        # Set data path
        data_path = os.getenv('DATA_PATH', 'data')
        self.data_folder = data_folder if data_folder else data_path
        
        try:
            self.load_data(self.data_folder)
            logger.info(f"AI Engine initialized with {len(self.vectorizers)} subjects")
        except Exception as e:
            logger.error(f"AI Engine initialization failed: {e}")
            # Clear any partial initialization
            self.vectorizers = {}
            self.subject_texts = {}
            self._text_cache = {}

    def load_data(self, folder_path: str) -> None:
        """Load and process all PDF data from the specified folder"""
        path = Path(folder_path)
        if not path.exists():
            logger.warning(f"Data folder not found at {path}")
            return

        processed_count = 0
        for exam_dir in path.iterdir():
            if exam_dir.is_dir():
                for subject_dir in exam_dir.iterdir():
                    if subject_dir.is_dir():
                        if self._process_subject(exam_dir.name, subject_dir.name, subject_dir):
                            processed_count += 1
                        gc.collect()

        logger.info(f"Loaded {processed_count} subjects")

    def _process_subject(self, exam: str, subject: str, subject_path: Path) -> bool:
        """Process all PDFs for a single subject"""
        cache_key = f"{exam}_{subject}"
        if cache_key in self._text_cache:
            return True

        texts = []
        for pdf_file in subject_path.glob("*.pdf"):
            try:
                text = self._process_pdf(pdf_file)
                if text and len(text) > 100:  # Minimum 100 characters
                    texts.append(text)
            except Exception as e:
                logger.error(f"Error processing {pdf_file}: {e}")
                continue

        if len(texts) < 1:
            logger.warning(f"No valid PDFs in {subject_path}")
            return False

        key = self._generate_key(exam, subject)
        try:
            self.vectorizers[key] = TfidfVectorizer(
                stop_words='english',
                max_features=1000,
                min_df=1,
                max_df=0.95,
                ngram_range=(1, 2))
            self.vectorizers[key].fit(texts)
            self.subject_texts[key] = texts
            self._text_cache[cache_key] = key
            return True
        except Exception as e:
            logger.error(f"Vectorizer failed for {key}: {e}")
            self._cleanup_failed_subject(key)
            return False

    def _process_pdf(self, pdf_path: Path) -> Optional[str]:
        """Extract and preprocess text from a PDF file"""
        try:
            text = []
            with pdf_path.open("rb") as f:
                reader = pypdf.PdfReader(f)
                for page in reader.pages[:50]:  # Limit to first 50 pages
                    if page_text := page.extract_text():
                        text.append(self._preprocess_text(page_text))
            return " ".join(text)[:15000] if text else None
        except Exception as e:
            logger.error(f"PDF processing error {pdf_path}: {e}")
            return None

    @staticmethod
    def _preprocess_text(text: str) -> str:
        """Basic text cleaning"""
        return " ".join(text.split())  # Normalize whitespace

    def get_answer(self, question: str, subject: str, exam: Optional[str] = None) -> str:
        """Get answer to a question using similarity search"""
        if not question.strip():
            return "Please provide a valid question."
            
        key = self._generate_key(exam, subject)
        if key not in self.vectorizers:
            return self._get_missing_subject_response(subject)

        try:
            question = self._preprocess_text(question)
            question_vec = self.vectorizers[key].transform([question])
            doc_vecs = self.vectorizers[key].transform(self.subject_texts[key])
            
            similarities = cosine_similarity(question_vec, doc_vecs)
            best_idx = similarities.argmax()
            best_score = similarities[0, best_idx]
            
            if best_score < 0.2:
                return self._get_low_confidence_response(subject)
                
            return self._format_response(
                self.subject_texts[key][best_idx],
                best_score
            )
        except Exception as e:
            logger.error(f"Query processing failed: {e}")
            return "Sorry, I encountered an error processing your question."

    # ... [keep all other existing helper methods unchanged] ...

    def _generate_key(self, exam: Optional[str], subject: str) -> str:
        return f"{exam.lower()}_{subject.lower()}" if exam else subject.lower()

    def _get_missing_subject_response(self, subject: str) -> str:
        available = sorted({k.split('_')[-1] for k in self.subject_texts})
        return (
            f"No content available for '{subject}'. "
            f"Available subjects: {', '.join(available)}"
        )

    def _get_low_confidence_response(self, subject: str) -> str:
        return (
            f"I couldn't find a confident answer about {subject}. "
            "Try rephrasing or asking about a different topic."
        )

    @staticmethod
    def _format_response(text: str, score: float) -> str:
        confidence = "high" if score > 0.15 else "medium"
        snippet = text[:600] + ("..." if len(text) > 600 else "")
        return (
            f"[{confidence} confidence]\n"
            f"{snippet}\n\n"
            f"(Source relevance score: {score:.2f})"
        )

    def list_available_subjects(self) -> List[str]:
        return sorted({k.split('_')[-1] for k in self.subject_texts})

    def get_subject_stats(self) -> Dict[str, Tuple[int, int]]:
        return {
            k: (len(texts), sum(len(t) for t in texts))
            for k, texts in self.subject_texts.items()
        }

    def _cleanup_failed_subject(self, key: str) -> None:
        for resource in [self.vectorizers, self.subject_texts, self._text_cache]:
            if key in resource:
                del resource[key]
        gc.collect()
        logger.info(f"Cleaned up resources for failed subject: {key}")