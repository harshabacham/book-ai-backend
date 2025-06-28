import os
import logging
import gc
import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path
import pypdf

logging.basicConfig(level=os.getenv('LOG_LEVEL', 'INFO'))
logger = logging.getLogger(__name__)

class AIEngine:
    def __init__(self, data_folder: str = None):
        self.vectorizers: Dict[str, TfidfVectorizer] = {}
        self.subject_texts: Dict[str, List[str]] = {}
        self._text_cache: Dict[str, str] = {}

        data_path = os.getenv('DATA_PATH', 'data')
        self.data_folder = data_folder if data_folder else data_path

        try:
            self.load_data(self.data_folder)
            logger.info(f"AI Engine initialized with {len(self.vectorizers)} subjects")
        except Exception as e:
            logger.error(f"AI Engine initialization failed: {e}")
            self.vectorizers = {}
            self.subject_texts = {}
            self._text_cache = {}

    def load_data(self, folder_path: str) -> None:
        """Load and process all PDF data from the specified folder"""
        path = Path(folder_path)
        if not path.exists():
            logger.warning(f"Data folder not found at {path}")
            return

        processed_subjects = set()

        for exam_dir in path.iterdir():
            if exam_dir.is_dir():
                for subject_dir in exam_dir.iterdir():
                    if subject_dir.is_dir():
                        key = self._generate_key(exam_dir.name, subject_dir.name)
                        if key not in processed_subjects:
                            self.subject_texts[key] = []
                        self._process_subject(key, subject_dir)
                        processed_subjects.add(key)
                        gc.collect()

        logger.info(f"Loaded {len(self.subject_texts)} subjects")

        # Fit vectorizers after all texts collected
        for key, texts in self.subject_texts.items():
            try:
                self.vectorizers[key] = TfidfVectorizer(
                    stop_words='english',
                    max_features=3000,
                    min_df=1,
                    max_df=0.9,
                    ngram_range=(1, 3)
                )
                self.vectorizers[key].fit(texts)
                logger.info(f"✅ Vectorizer created for: {key} with {len(texts)} chunks")
            except Exception as e:
                logger.error(f"⚠️ Vectorizer failed for {key}: {e}")
                self._cleanup_failed_subject(key)

    def _process_subject(self, key: str, subject_path: Path) -> None:
        """Extract all PDFs for a subject and collect texts"""
        for pdf_file in subject_path.glob("*.pdf"):
            try:
                text = self._process_pdf(pdf_file)
                if text:
                    chunks = self._split_chunks(text)
                    self.subject_texts[key].extend(chunks)
                    logger.info(f"📄 Loaded {len(chunks)} chunks from {pdf_file.name} into {key}")
            except Exception as e:
                logger.warning(f"❌ Error processing {pdf_file}: {e}")

    def _process_pdf(self, pdf_path: Path) -> Optional[str]:
        """Extract and clean text from a PDF"""
        try:
            text = []
            with pdf_path.open("rb") as f:
                reader = pypdf.PdfReader(f)
                for i, page in enumerate(reader.pages[:100]):
                    try:
                        page_text = page.extract_text()
                        if page_text:
                            cleaned = self._preprocess_text(page_text)
                            if len(cleaned) > 50:
                                text.append(cleaned)
                    except Exception as e:
                        logger.warning(f"⚠️ Page {i} error in {pdf_path.name}: {e}")
            combined = " ".join(text)[:20000]
            return combined if len(combined) > 200 else None
        except Exception as e:
            logger.error(f"PDF read failed {pdf_path.name}: {e}")
            return None

    def _split_chunks(self, text: str, chunk_size: int = 1000) -> List[str]:
        words = text.split()
        return [
            " ".join(words[i:i+chunk_size])
            for i in range(0, len(words), chunk_size)
            if len(words[i:i+chunk_size]) > 20
        ]

    @staticmethod
    def _preprocess_text(text: str) -> str:
        return " ".join(text.split())

    def get_answer(self, question: str, subject: str, exam: Optional[str] = None) -> str:
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

            logger.info(f"Q: {question}\nBest Match Score: {best_score:.4f}")
            logger.info(f"Top Result (short): {self.subject_texts[key][best_idx][:80]}...")

            if best_score < 0.2:
                return self._get_low_confidence_response(subject)

            return self._format_response(self.subject_texts[key][best_idx], best_score)
        except Exception as e:
            logger.error(f"❌ Error in get_answer: {e}")
            return "Sorry, I encountered an error processing your question."

    def _generate_key(self, exam: Optional[str], subject: str) -> str:
        return f"{exam.lower()}_{subject.lower()}" if exam else subject.lower()

    def _get_missing_subject_response(self, subject: str) -> str:
        available = sorted({k.split('_')[-1] for k in self.subject_texts})
        return f"No content available for '{subject}'. Available subjects: {', '.join(available)}"

    def _get_low_confidence_response(self, subject: str) -> str:
        return (
            f"I couldn't find a confident answer about {subject}. "
            "Try rephrasing or asking something more specific."
        )

    @staticmethod
    def _format_response(text: str, score: float) -> str:
        snippet = text[:600] + ("..." if len(text) > 600 else "")
        confidence = "High" if score > 0.4 else "Medium"
        return f"[{confidence} confidence]\n{snippet}\n\n(Source relevance: {score:.2f})"

    def list_available_subjects(self) -> List[str]:
        return sorted({k.split('_')[-1] for k in self.subject_texts})

    def _cleanup_failed_subject(self, key: str) -> None:
        for resource in [self.vectorizers, self.subject_texts, self._text_cache]:
            if key in resource:
                del resource[key]
        gc.collect()
        logger.info(f"🧹 Cleaned up resources for: {key}")
