import os
import gc
import pypdf
import numpy as np
import faiss
from typing import Dict, List, Optional
from sentence_transformers import SentenceTransformer
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AIEngine:
    def __init__(self, data_folder: str = "data"):
        """Initialize with a lightweight model and lazy loading"""
        self.subject_chunks: Dict[str, List[str]] = {}
        self.subject_embeddings: Dict[str, np.ndarray] = {}
        self.subject_index: Dict[str, faiss.Index] = {}
        self.model = None  # Will be loaded on first use
        self.model_name = "paraphrase-MiniLM-L3-v2"  # 128-dim instead of 384-dim
        self.load_data(data_folder)

    def load_model(self):
        """Lazy load the model to conserve memory"""
        if self.model is None:
            logger.info(f"Loading lightweight model: {self.model_name}")
            try:
                self.model = SentenceTransformer(self.model_name, device='cpu')
                # Reduce memory usage immediately
                self.model.max_seq_length = 128  # Reduce from default 256
            except Exception as e:
                logger.error(f"Failed to load model: {e}")
                raise

    def load_data(self, folder_path: str) -> None:
        """Load and process data with memory optimizations"""
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
        """Process a single subject with memory safeguards"""
        chunks = []
        for file in os.listdir(subject_path):
            if file.endswith(".pdf"):
                pdf_path = os.path.join(subject_path, file)
                try:
                    text = self.extract_text_from_pdf(pdf_path)
                    chunks.extend(self.split_text(text))
                except Exception as e:
                    logger.error(f"Error processing {pdf_path}: {e}")

        if not chunks:
            return

        key = f"{exam.lower()}_{subject.lower()}"
        self.subject_chunks[key] = chunks

        try:
            self.load_model()
            # Process in batches to reduce memory spikes
            batch_size = 10
            embeddings = []
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i + batch_size]
                embeddings.append(self.model.encode(batch, convert_to_numpy=True))
                gc.collect()
            
            embeddings = np.concatenate(embeddings)
            self.subject_embeddings[key] = embeddings.astype(np.float16)  # Save memory

            # Create FAISS index with reduced precision
            index = faiss.IndexFlatL2(embeddings.shape[1])
            index.add(embeddings.astype(np.float32))
            self.subject_index[key] = index
            logger.info(f"Loaded {len(chunks)} chunks for {key} (embeddings: {embeddings.shape})")
        except Exception as e:
            logger.error(f"Error creating embeddings/index for {key}: {e}")
            self._cleanup_failed_subject(key)

    def _cleanup_failed_subject(self, key: str) -> None:
        """Clean up failed subject processing"""
        self.subject_chunks.pop(key, None)
        self.subject_embeddings.pop(key, None)
        gc.collect()

    def extract_text_from_pdf(self, file_path: str) -> str:
        """Extract text from PDF with error handling"""
        text = []
        try:
            with open(file_path, "rb") as f:
                reader = pypdf.PdfReader(f)
                for page in reader.pages:
                    if page_text := page.extract_text():
                        text.append(page_text)
            return "\n".join(text)
        except Exception as e:
            logger.error(f"Error reading PDF {file_path}: {e}")
            raise

    def split_text(self, text: str, chunk_size: int = 200) -> List[str]:
        """Split text into smaller chunks with overlap"""
        words = text.split()
        chunks = []
        for i in range(0, len(words), chunk_size):
            chunk = " ".join(words[max(0, i-50):i+chunk_size])  # 50-word overlap
            chunks.append(chunk)
        return chunks

    def get_answer(self, question: str, exam: Optional[str] = None, subject: Optional[str] = None) -> str:
        """Get answer with memory-efficient processing"""
        if not question.strip():
            return "Please provide a valid question."
            
        if subject is None:
            return "Subject is required."

        try:
            self.load_model()
            key = self._get_subject_key(exam, subject)
            
            if key not in self.subject_index:
                return self._handle_missing_subject(subject)

            # Process question in memory-efficient way
            question_embedding = self.model.encode([question], convert_to_numpy=True)
            question_embedding = question_embedding.astype(np.float32)
            
            # Search with limited results
            D, I = self.subject_index[key].search(question_embedding, k=2)
            relevant_chunks = self._get_relevant_chunks(key, D, I)
            
            return self._format_answer(relevant_chunks) if relevant_chunks else \
                   "I couldn't find a relevant answer in my materials."
                
        except Exception as e:
            logger.error(f"Error processing question: {e}")
            return "Sorry, I encountered an error processing your question."
        finally:
            gc.collect()

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