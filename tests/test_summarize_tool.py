# tests/test_summarize_tool.py

import unittest
from langchain_ai_agent.agents.tools.summarize_tool import summarizer_chain
from dotenv import load_dotenv
load_dotenv()


class TestSummarizeTool(unittest.TestCase):
    def setUp(self):
        self.sample_input = {
            "text": (
                "Team met to discuss Q2 product roadmap. John presented customer feedback analysis. "
                "Main goals include improving onboarding, adding integration with Slack, and reducing churn. "
                "Sarah will lead the onboarding revamp, due by May 15. Next check-in scheduled for next Tuesday."
            )
        }

    def test_summary_output_format(self):
        output = summarizer_chain.invoke(self.sample_input)

        # Check keys
        self.assertIn("summary", output)
        self.assertIn("bullet_points", output)

        # Check types
        self.assertIsInstance(output["summary"], str)
        self.assertIsInstance(output["bullet_points"], list)

        # Ensure content is generated
        self.assertGreater(len(output["summary"]), 20)
        self.assertGreaterEqual(len(output["bullet_points"]), 2)

    def test_empty_text(self):
        with self.assertRaises(Exception):
            summarizer_chain.invoke({"text": ""})


if __name__ == "__main__":
    unittest.main()
