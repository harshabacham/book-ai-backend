import os
import gc
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pypdf
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AIEngine:
    def __init__(self, data_folder: str = "data"):
        self.vectorizers: Dict[str, TfidfVectorizer] = {}
        self.subject_texts: Dict[str, List[str]] = {}
        self._text_cache: Dict[str, str] = {}  # Cache for processed texts
        self.load_data(data_folder)
        logger.info("AI Engine initialized with optimized TF-IDF")

    def load_data(self, folder_path: str) -> None:
        """Load and process data with memory management"""
        folder_path = Path(folder_path)
        if not folder_path.exists():
            logger.warning(f"Data folder '{folder_path}' not found")
            return

        processed_count = 0
        for exam_dir in folder_path.iterdir():
            if exam_dir.is_dir():
                for subject_dir in exam_dir.iterdir():
                    if subject_dir.is_dir():
                        processed = self._process_subject(
                            exam_dir.name, 
                            subject_dir.name, 
                            subject_dir
                        )
                        if processed:
                            processed_count += 1
                        gc.collect()

        logger.info(f"Loaded {processed_count} subjects with {sum(len(v) for v in self.subject_texts.values())} documents")

    def _process_subject(self, exam: str, subject: str, subject_path: Path) -> bool:
        """Process all PDFs in a subject directory"""
        cache_key = f"{exam}_{subject}"
        if cache_key in self._text_cache:
            return True

        texts = []
        for pdf_file in subject_path.glob("*.pdf"):
            try:
                text = self._process_pdf(pdf_file)
                if text:
                    texts.append(text)
            except Exception as e:
                logger.error(f"Error processing {pdf_file}: {e}")
                continue

        if not texts:
            logger.warning(f"No valid PDFs in {subject_path}")
            return False

        key = self._generate_key(exam, subject)
        self.subject_texts[key] = texts
        
        try:
            self.vectorizers[key] = TfidfVectorizer(
                stop_words='english',
                max_features=5000,
                max_df=0.85,
                min_df=1
            )
            self.vectorizers[key].fit(texts)  # Just fit, we'll transform later
            self._text_cache[cache_key] = key
            return True
        except Exception as e:
            logger.error(f"Vectorizer failed for {key}: {e}")
            self._cleanup_failed_subject(key)
            return False

    def _process_pdf(self, pdf_path: Path) -> Optional[str]:
        """Extract and preprocess text from PDF"""
        try:
            text = []
            with pdf_path.open("rb") as f:
                reader = pypdf.PdfReader(f)
                for page in reader.pages[:50]:  # Limit to first 50 pages
                    if page_text := page.extract_text():
                        text.append(self._preprocess_text(page_text))
            
            return " ".join(text)[:15000]  # Concatenate and limit length
        except Exception as e:
            logger.error(f"PDF read error {pdf_path}: {e}")
            return None

    @staticmethod
    def _preprocess_text(text: str) -> str:
        """Basic text cleaning"""
        return " ".join(text.split())  # Normalize whitespace

    def get_answer(self, question: str, exam: Optional[str] = None, subject: Optional[str] = None) -> str:
        """Get answer with improved relevance scoring"""
        if not question.strip():
            return "Please provide a valid question."
            
        if not subject:
            return "Subject is required."

        question = self._preprocess_text(question)
        key = self._generate_key(exam, subject)
        
        if key not in self.vectorizers:
            return self._get_missing_subject_response(subject)

        try:
            # Transform only what we need
            question_vec = self.vectorizers[key].transform([question])
            doc_vecs = self.vectorizers[key].transform(self.subject_texts[key])
            
            similarities = cosine_similarity(question_vec, doc_vecs)
            best_idx = similarities.argmax()
            best_score = similarities[0, best_idx]
            
            if best_score < 0.25:  # Adjusted threshold
                return self._get_low_confidence_response(subject)
                
            return self._format_response(
                self.subject_texts[key][best_idx],
                best_score
            )
        except Exception as e:
            logger.error(f"Query failed: {e}", exc_info=True)
            return "Sorry, I encountered an error processing your question."

    def _generate_key(self, exam: Optional[str], subject: str) -> str:
        """Generate consistent lookup key"""
        return f"{exam.lower()}_{subject.lower()}" if exam else subject.lower()

    def _get_missing_subject_response(self, subject: str) -> str:
        """Generate helpful missing subject response"""
        available = sorted({k.split('_')[-1] for k in self.subject_texts})
        return (
            f"No content available for '{subject}'. "
            f"Available subjects: {', '.join(available)}"
        )

    def _get_low_confidence_response(self, subject: str) -> str:
        """Response when no good match found"""
        return (
            f"I couldn't find a confident answer about {subject}. "
            "Try rephrasing or asking about a different topic."
        )

    @staticmethod
    def _format_response(text: str, score: float) -> str:
        """Format answer with confidence indicator"""
        confidence = "high" if score > 0.15 else "medium"
        snippet = text[:600] + ("..." if len(text) > 600 else "")
        return (
            f"[{confidence} confidence]\n"
            f"{snippet}\n\n"
            f"(Source relevance score: {score:.2f})"
        )

    def list_available_subjects(self) -> List[str]:
        """Get sorted list of available subjects"""
        return sorted({k.split('_')[-1] for k in self.subject_texts})

    def get_subject_stats(self) -> Dict[str, Tuple[int, int]]:
        """Get statistics about loaded subjects"""
        return {
            k: (len(texts), sum(len(t) for t in texts))
            for k, texts in self.subject_texts.items()
        }

    def _cleanup_failed_subject(self, key: str) -> None:
        """Clean up resources for failed subject"""
        if key in self.vectorizers:
            del self.vectorizers[key]
        if key in self.subject_texts:
            del self.subject_texts[key]
        gc.collect()