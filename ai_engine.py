import os
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from pathlib import Path
import gc

logging.basicConfig(level=os.getenv('LOG_LEVEL', 'INFO'))
logger = logging.getLogger(__name__)

class AIEngine:
    def __init__(self, data_folder: str = None):
        # Initialize all required attributes first
        self.vectorizers = {}
        self.subject_texts = {}
        self._text_cache = {}
        
        data_path = os.getenv('DATA_PATH', 'data')
        if data_folder is None:
            data_folder = data_path
            
        try:
            self.load_data(data_folder)
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            # Clear any partially loaded data
            self.vectorizers = {}
            self.subject_texts = {}
            self._text_cache = {}

    def load_data(self, folder_path: str):
        """Load data with proper initialization"""
        path = Path(folder_path)
        if not path.exists():
            logger.warning(f"Data folder not found at {path}")
            return

        # Process each subject
        for exam_dir in path.iterdir():
            if exam_dir.is_dir():
                for subject_dir in exam_dir.iterdir():
                    if subject_dir.is_dir():
                        self._process_subject(exam_dir.name, subject_dir.name, subject_dir)

    def _process_subject(self, exam: str, subject: str, subject_path: Path):
        """Safe subject processing with fallbacks"""
        try:
            key = f"{exam.lower()}_{subject.lower()}"
            
            # Skip if already processed
            if key in self._text_cache:
                return True
                
            texts = []
            for pdf_file in subject_path.glob("*.pdf"):
                text = self._process_pdf(pdf_file)
                if text:
                    texts.append(text)
            
            if len(texts) < 1:  # Require at least 1 valid document
                logger.warning(f"No valid documents for {key}")
                return False
                
            # Initialize vectorizer
            self.vectorizers[key] = TfidfVectorizer(
                stop_words='english',
                max_features=1000,
                min_df=1,
                max_df=0.95
            )
            self.subject_texts[key] = texts
            self._text_cache[key] = True
            return True
            
        except Exception as e:
            logger.error(f"Error processing {subject}: {e}")
            self._cleanup_failed_subject(key)
            return False

    # ... [keep all other existing methods unchanged] ...

    def _process_pdf(self, pdf_path: Path) -> Optional[str]:
        """Extract text from PDF with enhanced error handling"""
        try:
            text = []
            with pdf_path.open("rb") as f:
                reader = pypdf.PdfReader(f)
                total_pages = min(len(reader.pages), 50)  # Process up to 50 pages
                
                for page_num in range(total_pages):
                    try:
                        page = reader.pages[page_num]
                        if page_text := page.extract_text():
                            text.append(self._preprocess_text(page_text))
                    except Exception as page_error:
                        logger.warning(f"Error on page {page_num} of {pdf_path}: {page_error}")
                        continue

            if not text:
                logger.warning(f"No text extracted from {pdf_path}")
                return None
                
            return " ".join(text)[:20000]

        except pypdf.PdfReadError:
            logger.error(f"Could not read PDF: {pdf_path}")
            return None
        except Exception as e:
            logger.error(f"Unexpected PDF processing error: {e}")
            return None

    @staticmethod
    def _preprocess_text(text: str) -> str:
        """Enhanced text cleaning"""
        text = " ".join(text.split())
        for artifact in ["\x0c", "\ufeff", "\u200b"]:
            text = text.replace(artifact, "")
        return text.lower()

    def get_answer(self, question: str, exam: Optional[str] = None, subject: Optional[str] = None) -> str:
        """Get answer with improved relevance and error handling"""
        if not question.strip():
            return "Please provide a valid question."
            
        if not subject:
            return "Subject is required."

        try:
            question = self._preprocess_text(question)
            key = self._generate_key(exam, subject)
            
            if key not in self.vectorizers:
                return self._get_missing_subject_response(subject)

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
            logger.error(f"Query processing failed: {e}", exc_info=True)
            return "Sorry, I encountered an error processing your question."

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