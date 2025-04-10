# tests/test_kb_tool.py

import unittest
from langchain_ai_agent.agents.tools.kb_tool import kb_writer_chain
from dotenv import load_dotenv
load_dotenv()


class TestKBTool(unittest.TestCase):
    def setUp(self):
        self.sample_doc = {
            "text": (
                "This user guide explains how to set up two-factor authentication (2FA). "
                "Users can enable 2FA via the security settings page by scanning a QR code "
                "with an authenticator app. This improves account protection."
            )
        }

    def test_output_structure(self):
        output = kb_writer_chain.invoke(self.sample_doc)

        self.assertIn("qa_pairs", output)
        self.assertIsInstance(output["qa_pairs"], list)
        self.assertGreaterEqual(len(output["qa_pairs"]), 2)

        for qa in output["qa_pairs"]:
            self.assertIsInstance(qa, dict)
            self.assertIn("question", qa)
            self.assertIn("answer", qa)
            self.assertIsInstance(qa["question"], str)
            self.assertIsInstance(qa["answer"], str)
            self.assertGreater(len(qa["question"]), 5)
            self.assertGreater(len(qa["answer"]), 5)

    def test_invalid_input(self):
        with self.assertRaises(Exception):
            kb_writer_chain.invoke({"text": ""})


if __name__ == "__main__":
    unittest.main()
