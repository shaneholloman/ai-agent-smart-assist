import logging
from pathlib import Path
from typing import List, Dict, Optional
import yaml

from langchain.text_splitter import RecursiveCharacterTextSplitter
from unstructured.partition.auto import partition


# Setup logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class DocumentIngestor:
    def __init__(self, config_path: str = "config/ingestion_config.yaml"):
        self.config = self._load_config(config_path)
        self.chunk_size = self.config.get("chunk_size", 500)
        self.chunk_overlap = self.config.get("chunk_overlap", 50)
        self.supported_extensions = set(self.config.get("supported_extensions", [".pdf", ".docx", ".txt", ".eml", ".html"]))
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )

    def _load_config(self, path: str) -> Dict:
        try:
            with open(path, "r") as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logger.warning(f"No config file found at {path}, using defaults.")
            return {}
    
    def _extract_text(self, filepath: Path) -> str:
        try:
            elements = partition(filename=str(filepath))
            return "\n".join([el.text for el in elements if hasattr(el, "text") and el.text])
        except Exception as e:
            logger.error(f"[Ingestor] Failed to parse {filepath.name}: {e}")
            return ""
        
    def _is_supported(self, filepath: Path) -> bool:
        return filepath.suffix.lower() in self.supported_extensions

    def process_file(self, filepath:str) -> List[Dict]:
        if not self._is_supported(filepath):
            logger.warning(f"[Ingestor] Skipping unsupported file type: {filepath.name}")
            return []
        
        logger.info(f"[Ingestor] Processing file: {filepath.name}")
        raw_text = self._extract_text(filepath)

        if not raw_text.strip():
            logger.warning(f"[Ingestor] No text extracted from {filepath.name}")
            return []

        chunks = self.text_splitter.split_text(raw_text)

        return [
            {"chunk_id": i,
             "text": chunk,
             "doc_path": str(filepath),
             "filename": filepath.name,
             "source_type": filepath.suffix.lstrip(".").lower()}

            for i, chunk in enumerate(chunks)
        ]
    
    def process_directory(self, folder_path: Path) -> List[Dict]:
        all_chunks = []
        for file_path in folder_path.glob("**/*"):
            if file_path.is_file():
                chunks = self.process_file(file_path)
                all_chunks.extend(chunks)
        logger.info(f"[Ingestor] Finished processing directory: {folder_path}")
        return all_chunks

def ingest_and_chunk(path: str) -> dict:
    # example dummy logic
    return {"status": "ingested", "path": path}
