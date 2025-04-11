import os
import json
import logging
from typing import List, Dict, Optional, Any
from pathlib import Path

from sentence_transformers import SentenceTransformer  # if needed elsewhere
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from pydantic import BaseModel, Field, PrivateAttr, ValidationError
from langchain_core.retrievers import BaseRetriever

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class ChunkMetadata(BaseModel):
    chunk_id: int
    text: str
    filename: str
    source_type: str
    doc_path: str


class DocumentEmbedder(BaseRetriever, BaseModel):
    # Public fields, part of the retriever's configuration.
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    persist_dir: str = "faiss_index"

    # Private attributes that will not be part of the Pydantic model
    _persist_dir: Path = PrivateAttr()
    _metadata_file: Path = PrivateAttr()
    _embedding_function: Any = PrivateAttr()
    _vector_store: Optional[Any] = PrivateAttr(default=None)

    def __init__(self, **data):
        super().__init__(**data)
        # Convert the persist_dir (a string) into a Path object and store it as a private attribute.
        self._persist_dir = Path(self.persist_dir)
        self._metadata_file = self._persist_dir / "metadata.jsonl"
        self._embedding_function = HuggingFaceEmbeddings(model_name=self.model_name)
        
        # Create the directory if it doesn't exist; otherwise try to load the FAISS index.
        if not self._persist_dir.exists():
            self._persist_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"[Embedder] Initialized new FAISS index directory: {self._persist_dir}")
        else:
            self._load_faiss()

    def _load_faiss(self):
        try:
            self._vector_store = FAISS.load_local(
                folder_path=str(self._persist_dir),
                embeddings=self._embedding_function,
                allow_dangerous_deserialization=True
            )
            logger.info("[Embedder] Loaded FAISS index from disk.")
        except Exception as e:
            logger.error(f"[Embedder] Failed to load FAISS index: {e}")
            # The existing index file might be corrupted or incompatible.
            # Remove all files in the persist directory to force a rebuild.
            try:
                for file in self._persist_dir.iterdir():
                    file.unlink()
                logger.info("[Embedder] Removed corrupted FAISS index files from disk.")
            except Exception as remove_error:
                logger.error(f"[Embedder] Failed to remove corrupted FAISS index files: {remove_error}")
            # Set the vector store to None so that downstream queries fail fast
            # and a new index can be built by calling build_or_update_index.
            self._vector_store = None


    def _load_existing_metadata(self) -> List[Dict]:
        if self._metadata_file.exists():
            with open(self._metadata_file, "r") as f:
                return [json.loads(line) for line in f]
        return []

    def _append_metadata(self, metadata: List[Dict]):
        with open(self._metadata_file, "a") as f:
            for record in metadata:
                f.write(json.dumps(record) + "\n")

    def _deduplicate_chunks(
        self,
        new_chunks: List[ChunkMetadata],
        existing_metadata: List[Dict]
    ) -> List[ChunkMetadata]:
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

        if self._vector_store:
            self._vector_store.add_documents(documents)
            logger.info(f"[Embedder] Appended {len(documents)} new documents to existing index.")
        else:
            self._vector_store = FAISS.from_documents(documents, self._embedding_function)
            logger.info(f"[Embedder] Created new FAISS index with {len(documents)} documents.")

        self._vector_store.save_local(str(self._persist_dir))
        self._append_metadata([doc.metadata for doc in documents])

    def get_retriever(self, k: int = 4):
        if self._vector_store is None:
            raise ValueError("[Embedder] Vector store not initialized.")
        return self._vector_store.as_retriever(search_kwargs={"k": k})

    def query(self, question: str, k: int = 4) -> List[Document]:
        retriever = self.get_retriever(k=k)
        docs = retriever.get_relevant_documents(question)
        logger.info(f"[Embedder] Retrieved {len(docs)} relevant documents for query.")
        return docs

    # Required by BaseRetriever: a synchronous method accepting a string and returning documents.
    def _get_relevant_documents(self, query: str) -> List[Document]:
        retriever = self.get_retriever(k=4)
        docs = retriever.get_relevant_documents(query)
        logger.info(f"[Embedder] Retrieved {len(docs)} relevant documents for query.")
        return docs

    # Optional asynchronous version.
    async def _aget_relevant_documents(self, query: str) -> List[Document]:
        retriever = self.get_retriever(k=4)
        docs = await retriever.aget_relevant_documents(query)
        logger.info(f"[Embedder] Retrieved {len(docs)} relevant documents for query (async).")
        return docs
