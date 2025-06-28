import os
import logging
import gc
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
        cache_key = f"{exam}_{subject}"
        if cache_key in self._text_cache:
            return True

        texts = []
        for pdf_file in subject_path.glob("*.pdf"):
            try:
                chunks = self._process_pdf(pdf_file)
                if chunks:
                    texts.extend(chunks)
            except Exception as e:
                logger.warning(f"Error processing {pdf_file}: {e}")
                continue

        if len(texts) < 1:
            logger.warning(f"No valid text chunks in {subject_path}")
            return False

        key = self._generate_key(exam, subject)
        try:
            self.vectorizers[key] = TfidfVectorizer(
                stop_words='english',
                max_features=3000,
                min_df=1,
                max_df=0.9,
                ngram_range=(1, 2)
            )
            self.vectorizers[key].fit(texts)
            self.subject_texts[key] = texts
            self._text_cache[cache_key] = key
            logger.info(f"Successfully loaded subject: {key} with {len(texts)} chunks")
            return True
        except Exception as e:
            logger.error(f"Vectorizer failed for {key}: {e}")
            self._cleanup_failed_subject(key)
            return False

    def _process_pdf(self, pdf_path: Path) -> Optional[List[str]]:
        try:
            text = []
            with pdf_path.open("rb") as f:
                reader = pypdf.PdfReader(f)
                for page in reader.pages[:100]:
                    try:
                        if page_text := page.extract_text():
                            cleaned = self._preprocess_text(page_text)
                            if len(cleaned) > 1:
                                text.append(cleaned)
                    except Exception as page_error:
                        logger.warning(f"Page error in {pdf_path}: {page_error}")
                        continue

            if not text:
                logger.warning(f"No extractable text in {pdf_path}")
                return None

            # Chunking by ~500 words
            chunks = []
            words = []
            for t in text:
                words.extend(t.split())
                while len(words) >= 500:
                    chunk = " ".join(words[:500])
                    chunks.append(chunk)
                    words = words[500:]

            if words:
                chunks.append(" ".join(words))

            return chunks if chunks else None
        except Exception as e:
            logger.error(f"PDF processing failed {pdf_path}: {e}")
            return None

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

            logger.info(f"Q: {question}")
            logger.info(f"Best Match Score: {best_score:.4f}")
            logger.info(f"Top Result (short): {self.subject_texts[key][best_idx][:80]}")

            if best_score < 0.0001:  # Adjusted threshold for low confidence
                return self._get_low_confidence_response(subject)

            best_text = self.subject_texts[key][best_idx]
            lines = best_text.split('. ')
            preview = '. '.join(lines[:3])
            return self._format_response(preview, best_score)

        except Exception as e:
            logger.error(f"Query processing failed: {e}")
            return "Sorry, I encountered an error processing your question."

    def _generate_key(self, exam: Optional[str], subject: str) -> str:
        return f"{exam.lower()}_{subject.lower()}" if exam else subject.lower()

    def _get_missing_subject_response(self, subject: str) -> str:
        available = sorted({k.split('_')[-1] for k in self.subject_texts})
        return f"No content available for '{subject}'. Available subjects: {', '.join(available)}"

    def _get_low_confidence_response(self, subject: str) -> str:
        return (
            f"I couldn't find a confident answer about {subject}. "
            "Try rephrasing or asking about a different topic."
        )

    @staticmethod
    def _format_response(text: str, score: float) -> str:
        confidence = "high" if score > 0.4 else "medium"
        snippet = text[:600] + ("..." if len(text) > 600 else "")
        return f"[{confidence} confidence]\n{snippet}\n\n(Source score: {score:.2f})"

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
