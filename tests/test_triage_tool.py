# tests/test_triage_tool.py

import unittest
from langchain_ai_agent.agents.tools.triage_tool import triage_chain
from dotenv import load_dotenv
load_dotenv()


class TestTriageTool(unittest.TestCase):
    def setUp(self):
        self.sample_ticket = {
            "text": (
                "Hi, I'm having trouble logging into my account. It says my password is incorrect "
                "but I'm sure it's right. Can someone help reset it urgently?"
            )
        }

    def test_output_structure(self):
        output = triage_chain.invoke(self.sample_ticket)

        self.assertIn("category", output)
        self.assertIn("urgency", output)
        self.assertIn("route_to", output)
        self.assertIn("explanation", output)

        self.assertIsInstance(output["category"], str)
        self.assertIsInstance(output["urgency"], str)
        self.assertIsInstance(output["route_to"], str)
        self.assertIsInstance(output["explanation"], str)

    def test_invalid_input(self):
        with self.assertRaises(Exception):
            triage_chain.invoke({"text": ""})


if __name__ == "__main__":
    unittest.main()
