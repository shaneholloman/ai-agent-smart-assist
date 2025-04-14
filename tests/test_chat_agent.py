# tests/test_chat_agent.py

import unittest
import shutil
from pathlib import Path
from langchain_ai_agent.agents.chat_agent import get_chat_agent_with_memory
from dotenv import load_dotenv
import asyncio

load_dotenv()


class TestChatAgent(unittest.TestCase):
    def setUp(self):
        self.test_dir = Path("tests/tmp_chat_agent_store")
        self.test_dir.mkdir(parents=True, exist_ok=True)
        self.persist_dir = str(self.test_dir)
        self.agent = get_chat_agent_with_memory(persist_dir=self.persist_dir)

    def tearDown(self):
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

    def test_chat_agent_returns_answer(self):
        result = asyncio.run(
            self.agent.ainvoke({"question": "What is LangChain?"})
        )
        self.assertIn("graph_output", result)
        self.assertIsInstance(result["graph_output"], str)
        self.assertGreater(len(result["graph_output"]), 0)

    def test_chat_agent_handles_empty_question(self):
        result = asyncio.run(
            self.agent.ainvoke({"question": ""})
        )
        self.assertEqual(result["graph_output"], "[call_model] Empty or missing question.")

    def test_chat_agent_message_format(self):
        result = asyncio.run(
            self.agent.ainvoke({"question": "Tell me about vector search."})
        )
        self.assertIn("messages", result)
        self.assertIsInstance(result["messages"], list)
        self.assertTrue(any("vector" in msg.content.lower() for msg in result["messages"] if hasattr(msg, "content")))


if __name__ == "__main__":
    unittest.main()
