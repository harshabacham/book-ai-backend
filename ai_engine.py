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
logger = logging.getLogger("ai_engine")

class AIEngine:
    def __init__(self, data_folder: str = os.getenv("DATA_PATH", "data")):
        self.load_data(data_folder)
        """Initialize with enhanced error handling and resource management"""
        self.vectorizers: Dict[str, TfidfVectorizer] = {}
        self.subject_texts: Dict[str, List[str]] = {}
        self._text_cache: Dict[str, str] = {}
        
        # Validate data folder exists before proceeding
        self.data_folder = Path(data_folder)
        if not self.data_folder.exists():
            logger.error(f"Data folder not found at: {self.data_folder.absolute()}")
            raise FileNotFoundError(f"Data directory not found: {data_folder}")
        
        self.load_data(self.data_folder)
        logger.info(f"AI Engine initialized with {len(self.subject_texts)} subjects")

    def load_data(self, folder_path: Path) -> None:
        """Load data with improved error handling and validation"""
        processed_count = 0
        document_count = 0
        
        try:
            # Verify directory structure
            if not any(folder_path.iterdir()):
                logger.warning(f"No exam directories found in {folder_path}")
                return

            for exam_dir in folder_path.iterdir():
                if not exam_dir.is_dir():
                    continue
                    
                for subject_dir in exam_dir.iterdir():
                    if subject_dir.is_dir():
                        if self._process_subject(exam_dir.name, subject_dir.name, subject_dir):
                            processed_count += 1
                            document_count += len(self.subject_texts[f"{exam_dir.name}_{subject_dir.name}"])
                        gc.collect()  # Proactively manage memory

        except Exception as e:
            logger.error(f"Critical error loading data: {e}", exc_info=True)
            raise

        logger.info(f"Loaded {processed_count} subjects with {document_count} documents")

    def _process_subject(self, exam: str, subject: str, subject_path: Path) -> bool:
        """Process subject with robust error handling"""
        cache_key = f"{exam}_{subject}"
        key = self._generate_key(exam, subject)
        
        try:
            # Skip if already processed
            if cache_key in self._text_cache:
                return True

            # Validate PDF files exist
            pdf_files = list(subject_path.glob("*.pdf"))
            if not pdf_files:
                logger.warning(f"No PDFs found in {subject_path}")
                return False

            # Process all PDFs
            texts = []
            for pdf_file in pdf_files:
                if text := self._process_pdf(pdf_file):
                    texts.append(text)

            if not texts:
                logger.warning(f"No valid text extracted from {subject_path}")
                return False

            # Initialize vectorizer with safe defaults
            self.vectorizers[key] = TfidfVectorizer(
                stop_words='english',
                max_features=1000,
                # max_df=0.95,
                # min_df=1,
                # ngram_range=(1, 2)
            )
            
            # Fit vectorizer
            self.vectorizers[key].fit(texts)
            self.subject_texts[key] = texts
            self._text_cache[cache_key] = key
            return True

        except ValueError as e:
            logger.error(f"Vectorizer configuration error for {key}: {e}")
            self._cleanup_failed_subject(key)
            return False
        except Exception as e:
            logger.error(f"Unexpected error processing {key}: {e}")
            self._cleanup_failed_subject(key)
            return False

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