# tests/test_ingest_api.py

import os
import shutil
import unittest
from pathlib import Path
from fastapi.testclient import TestClient
from langchain_ai_agent.api.main import app
from dotenv import load_dotenv

load_dotenv()


class TestIngestAPI(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)
        self.upload_dir = Path("tmp_uploads")
        self.faiss_index_dir = Path("faiss_index/default")

        # Create a sample text file to upload
        self.sample_file_path = Path("tests/sample_ingest.txt")
        self.sample_file_path.parent.mkdir(parents=True, exist_ok=True)
        self.sample_file_path.write_text(
            "This is a test file.\nIt contains example documentation to be chunked and embedded."
        )

    def tearDown(self):
        if self.sample_file_path.exists():
            self.sample_file_path.unlink()

        if self.upload_dir.exists():
            shutil.rmtree(self.upload_dir)

        if self.faiss_index_dir.exists():
            shutil.rmtree(self.faiss_index_dir)

    def test_ingest_success(self):
        with open(self.sample_file_path, "rb") as f:
            response = self.client.post(
                "/api/ingest",
                files={"files": ("sample_ingest.txt", f, "text/plain")},
            )

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "success")
        self.assertEqual(data["namespace"], "default")
        self.assertGreater(data["num_chunks"], 0)

    def test_ingest_empty_file(self):
        empty_file_path = self.sample_file_path.parent / "empty.txt"
        empty_file_path.write_text("")

        with open(empty_file_path, "rb") as f:
            response = self.client.post(
                "/api/ingest",
                files={"files": ("empty.txt", f, "text/plain")},
            )

        self.assertEqual(response.status_code, 400)
        self.assertIn("No valid chunks", response.json()["detail"])

        empty_file_path.unlink()

    def test_ingest_missing_file(self):
        response = self.client.post("/api/ingest", files={})
        self.assertEqual(response.status_code, 422)


if __name__ == "__main__":
    unittest.main()
