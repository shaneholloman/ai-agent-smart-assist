# tests/test_vector_store.py

import unittest
from pathlib import Path
import shutil
from langchain_ai_agent.retriever.vector_store import DocumentEmbedder
from dotenv import load_dotenv

load_dotenv()


class TestDocumentEmbedder(unittest.TestCase):
    def setUp(self):
        # Create a temporary FAISS index directory for testing
        self.test_index_path = Path("tests/test_faiss_index")
        self.embedder = DocumentEmbedder(persist_dir=str(self.test_index_path))

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
        # Clean up the test FAISS directory
        if self.test_index_path.exists():
            shutil.rmtree(self.test_index_path)

    def test_build_index_successfully(self):
        self.embedder.build_or_update_index(self.sample_chunks)
        retriever = self.embedder.get_retriever()
        self.assertIsNotNone(retriever)

        # Access internal FAISS index
        index = self.embedder._vector_store.index
        self.assertEqual(index.d, 384)
        self.assertEqual(index.ntotal, 2)

    def test_deduplication_skips_duplicates(self):
        self.embedder.build_or_update_index(self.sample_chunks)
        self.embedder.build_or_update_index(self.sample_chunks)  # Add same again
        retriever = self.embedder.get_retriever()
        results = retriever.get_relevant_documents("LangChain vector search")
        self.assertLessEqual(len(results), 2)

    def test_query_returns_documents(self):
        self.embedder.build_or_update_index(self.sample_chunks)
        results = self.embedder.query("How does LangChain work?")
        self.assertGreater(len(results), 0)
        self.assertIn("LangChain", results[0].page_content)

    def test_build_index_empty_input(self):
        with self.assertRaises(ValueError):
            self.embedder.build_or_update_index([])

    def test_schema_validation_raises(self):
        with self.assertRaises(Exception):
            self.embedder.build_or_update_index([{"text": "Missing metadata"}])


if __name__ == "__main__":
    unittest.main()
