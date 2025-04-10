import os
import json
import logging
from typing import List, Dict, Optional
from pathlib import Path

from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from pydantic import BaseModel, ValidationError

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class ChunkMetadata(BaseModel):
    chunk_id: int
    text: str
    filename: str
    source_type: str
    doc_path: str


class DocumentEmbedder:
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        persist_dir: Optional[str] = "faiss_index"
    ):
        self.persist_dir = Path(persist_dir)
        self.metadata_file = self.persist_dir / "metadata.jsonl"
        self.model_name = model_name
        self.embedding_function = HuggingFaceEmbeddings(model_name=self.model_name)
        self.vector_store = None

        if self.persist_dir.exists():
            self._load_faiss()
        else:
            self.persist_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"[Embedder] Initialized new FAISS index directory: {self.persist_dir}")

    def _load_faiss(self):
        try:
            self.vector_store = FAISS.load_local(
                folder_path=str(self.persist_dir),
                embeddings=self.embedding_function,
                allow_dangerous_deserialization=True
            )
            logger.info("[Embedder] Loaded FAISS index from disk.")
        except Exception as e:
            logger.error(f"[Embedder] Failed to load FAISS index: {e}")
            self.vector_store = None

    def _load_existing_metadata(self) -> List[Dict]:
        if self.metadata_file.exists():
            with open(self.metadata_file, "r") as f:
                return [json.loads(line) for line in f]
        return []

    def _append_metadata(self, metadata: List[Dict]):
        with open(self.metadata_file, "a") as f:
            for record in metadata:
                f.write(json.dumps(record) + "\n")

    def _deduplicate_chunks(self, new_chunks: List[ChunkMetadata], existing_metadata: List[Dict]) -> List[ChunkMetadata]:
        existing_keys = {
            (item["chunk_id"], item["filename"])
            for item in existing_metadata
        }
        filtered = [
            chunk for chunk in new_chunks
            if (chunk.chunk_id, chunk.filename) not in existing_keys
        ]
        return filtered

    def build_or_update_index(self, chunk_data: List[Dict]):
        if not chunk_data:
            raise ValueError("[Embedder] No chunks provided.")

        validated_chunks = []
        for i, item in enumerate(chunk_data):
            if not isinstance(item, dict):
                logger.warning(f"[Embedder] Skipping non-dict chunk at index {i}: {item}")
                continue
            try:
                validated_chunks.append(ChunkMetadata(**item))
            except ValidationError as e:
                logger.error(f"[Embedder] Invalid chunk at index {i}: {e}")
                raise

        existing_metadata = self._load_existing_metadata()
        new_chunks = self._deduplicate_chunks(validated_chunks, existing_metadata)

        if not new_chunks:
            logger.info("[Embedder] No new unique chunks to index.")
            return

        documents = [
            Document(
                page_content=chunk.text,
                metadata={
                    "chunk_id": chunk.chunk_id,
                    "filename": chunk.filename,
                    "source_type": chunk.source_type,
                    "doc_path": chunk.doc_path
                }
            )
            for chunk in new_chunks
        ]

        if self.vector_store:
            self.vector_store.add_documents(documents)
            logger.info(f"[Embedder] Appended {len(documents)} new documents to existing index.")
        else:
            self.vector_store = FAISS.from_documents(documents, self.embedding_function)
            logger.info(f"[Embedder] Created new FAISS index with {len(documents)} documents.")

        self.vector_store.save_local(str(self.persist_dir))
        self._append_metadata([doc.metadata for doc in documents])

    def get_retriever(self, k: int = 4):
        if self.vector_store is None:
            raise ValueError("[Embedder] Vector store not initialized.")
        return self.vector_store.as_retriever(search_kwargs={"k": k})

    def query(self, question: str, k: int = 4) -> List[Document]:
        retriever = self.get_retriever(k=k)
        docs = retriever.get_relevant_documents(question)
        logger.info(f"[Embedder] Retrieved {len(docs)} relevant documents for query.")
        return docs
