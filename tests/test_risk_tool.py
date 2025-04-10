# tests/test_risk_tool.py

import unittest
from langchain_ai_agent.agents.tools.risk_tool import risk_flagger_chain
from dotenv import load_dotenv
load_dotenv()


class TestRiskTool(unittest.TestCase):
    def setUp(self):
        self.sample_contract = {
            "text": (
                "The agreement may be terminated by either party with 15 days' notice. "
                "Liability is limited to the amount paid in the last billing cycle. "
                "Disputes shall be resolved by binding arbitration in the State of Delaware."
            )
        }

    def test_risk_output_structure(self):
        output = risk_flagger_chain.invoke(self.sample_contract)

        self.assertIn("risks_found", output)
        self.assertIn("explanation", output)

        self.assertIsInstance(output["risks_found"], list)
        self.assertIsInstance(output["explanation"], str)

        self.assertGreaterEqual(len(output["risks_found"]), 1)

    def test_invalid_input(self):
        with self.assertRaises(Exception):
            risk_flagger_chain.invoke({"text": ""})


if __name__ == "__main__":
    unittest.main()
