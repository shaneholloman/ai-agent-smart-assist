# tests/test_vector_store.py

import unittest
from pathlib import Path
import shutil
from langchain_ai_agent.retriever.vector_store import DocumentEmbedder
from uuid import uuid4
from dotenv import load_dotenv
load_dotenv()


class TestDocumentEmbedder(unittest.TestCase):
    def setUp(self):
        # Use a temporary directory for FAISS index persistence
        self.test_index_path = Path("tests/test_faiss_index")
        self.embedder = DocumentEmbedder(persist_dir=str(self.test_index_path))

        # Sample chunked documents
        self.sample_chunks = [
            {
                "chunk_id": 0,
                "text": "LangChain helps build LLM-powered apps using composable components.",
                "filename": "doc1.txt",
                "source_type": "txt",
                "doc_path": "/fake/path/doc1.txt"
            },
            {
                "chunk_id": 1,
                "text": "FAISS is a library for efficient similarity search and clustering of dense vectors.",
                "filename": "doc1.txt",
                "source_type": "txt",
                "doc_path": "/fake/path/doc1.txt"
            }
        ]

    def tearDown(self):
        # Clean up test index directory
        if self.test_index_path.exists():
            shutil.rmtree(self.test_index_path)

    def test_build_index(self):
        self.embedder.build_or_update_index(self.sample_chunks)
        retriever = self.embedder.get_retriever()
        self.assertIsNotNone(retriever)

        # Get FAISS index shape
        index = self.embedder.vector_store.index
        self.assertEqual(index.d, 384)  # Confirm vector dimensionality
        self.assertEqual(index.ntotal, 2)  # Confirm number of stored vectors

    def test_duplicate_chunks_are_ignored(self):
        self.embedder.build_or_update_index(self.sample_chunks)
        self.embedder.build_or_update_index(self.sample_chunks)  # Same chunks again
        retriever = self.embedder.get_retriever()
        docs = retriever.get_relevant_documents("What is LangChain?")
        self.assertTrue(len(docs) <= 2)

    def test_query_returns_results(self):
        self.embedder.build_or_update_index(self.sample_chunks)
        results = self.embedder.query("How do I use LangChain?")
        self.assertGreater(len(results), 0)
        self.assertIn("LangChain", results[0].page_content)

    def test_reject_empty_chunks(self):
        with self.assertRaises(ValueError):
            self.embedder.build_or_update_index([])

    def test_invalid_chunk_schema(self):
        invalid_chunks = [{"text": "Missing keys"}]
        with self.assertRaises(Exception):
            self.embedder.build_or_update_index(invalid_chunks)


if __name__ == "__main__":
    unittest.main()
