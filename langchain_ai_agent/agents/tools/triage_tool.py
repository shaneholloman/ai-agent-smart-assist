# agents/tools/triage_tool.py

import logging
import asyncio
from typing import Dict, Any
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import Runnable, RunnableLambda
from langchain_core.output_parsers import JsonOutputParser
from langchain_google_vertexai import ChatVertexAI

# Logger setup
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Prompt Template for ticket triage
TRIAGE_PROMPT = PromptTemplate.from_template(
    """You are a support ticket triage assistant.

Given the following support message, classify it into:
- category: 'billing', 'technical', 'account', or 'general'
- urgency: 'low', 'medium', or 'high'
- route_to: based on the issue type and urgency (e.g., 'Billing Support', 'Level 2 Support', 'Account Admin')
- explanation: reason for the above decisions

Support Ticket:
{text}

Return a JSON object:
- category
- urgency
- route_to
- explanation
"""
)

# Gemini 1.5 Flash via Vertex AI
llm = ChatVertexAI(
    model_name="gemini-2.0-flash-lite",
    temperature=0.3,
    max_output_tokens=1024,
)

# JSON output parser
parser = JsonOutputParser()


async def _log_input(x: Dict[str, Any]) -> Dict[str, Any]:
    logger.info(f"[Triage] Classifying support ticket of length {len(x.get('text', ''))}")
    return x

# Validation for triage output
async def _validate_triage_output(output: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(output, dict):
        raise ValueError("Output is not a valid dictionary.")
    for key in ["category", "urgency", "route_to", "explanation"]:
        if key not in output or not isinstance(output[key], str):
            raise ValueError(f"Missing or invalid '{key}' in output.")
    return output

# Final Runnable chain with logging and validation
triage_chain: Runnable = (
    {"text": lambda x: x["text"]}
    | RunnableLambda(_log_input)
    | TRIAGE_PROMPT
    | llm
    | parser
    | RunnableLambda(_validate_triage_output)
)
