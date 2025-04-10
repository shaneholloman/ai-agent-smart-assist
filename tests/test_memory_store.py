# tests/test_memory_store.py

import unittest
import shutil
from pathlib import Path
from langchain_ai_agent.feedback_loop.memory_store import MemoryStore
from dotenv import load_dotenv
load_dotenv()


class TestMemoryStore(unittest.TestCase):
    def setUp(self):
        self.test_dir = Path("tests/temp_memory_store")
        self.store = MemoryStore(persist_dir=str(self.test_dir))

        # Sample experience
        self.sample_input = "Customer reported issues with logging into the dashboard."
        self.sample_output = {
            "category": "account",
            "urgency": "high",
            "route_to": "Level 2 Support"
        }
        self.task = "triage"
        self.meta = {
            "filename": "support_ticket_001.txt",
            "chunk_id": 0
        }

    def tearDown(self):
        # Cleanup the temporary test memory store
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

    def test_add_and_query_experience(self):
        # Add experience to memory
        self.store.add_experience(
            input_text=self.sample_input,
            output=self.sample_output,
            task=self.task,
            metadata=self.meta
        )

        # Query for similar items
        results = self.store.query_similar(self.sample_input, k=1)

        self.assertEqual(len(results), 1)
        self.assertIn("metadata", results[0])
        self.assertEqual(results[0]["metadata"]["task"], self.task)
        self.assertIn("output", results[0]["metadata"])

    def test_query_without_data(self):
        # Querying empty memory should return empty list
        empty_store = MemoryStore(persist_dir="tests/empty_memory_store")
        results = empty_store.query_similar("Unseen input")
        self.assertEqual(results, [])

        # Clean up manually created folder
        shutil.rmtree("tests/empty_memory_store")


if __name__ == "__main__":
    unittest.main()
