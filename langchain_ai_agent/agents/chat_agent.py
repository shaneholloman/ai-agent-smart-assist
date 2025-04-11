from langchain.chains import (
    create_history_aware_retriever,
)
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_vertexai import ChatVertexAI
from langchain_ai_agent.retriever.vector_store import DocumentEmbedder
from langchain_core.runnables import RunnableLambda, RunnableMap

# LangGraph memory imports
from langgraph.checkpoint.memory import MemorySaver
from langchain.prompts import PromptTemplate
from langgraph.graph import START, END, StateGraph
from langchain_core.messages import AIMessage, HumanMessage
from typing import TypedDict, Annotated, Sequence
import operator
import logging

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class AgentState(TypedDict):
    messages: Annotated[Sequence[HumanMessage | AIMessage], operator.add]
    question: str
    graph_output: str


def get_chat_agent_with_memory(persist_dir: str):
    """
    Creates and returns an agent that maintains persistent conversation memory.
    The agent is invoked using .ainvoke({"question": ...}, config={"configurable": {"thread_id": ...}})
    """
    embedder = DocumentEmbedder(persist_dir=persist_dir)
    retriever = embedder.get_retriever(k=10)

    llm = ChatVertexAI(
        model_name="gemini-2.0-flash-lite",
        temperature=0.3,
        max_output_tokens=1024,
    )

    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question that can be understood "
        "without the chat history. Do NOT answer the question; just "
        "reformulate it if needed and otherwise return it as is."
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ]
    )

    try:
        history_aware_retriever = create_history_aware_retriever(
            llm, retriever, contextualize_q_prompt
        )
    except Exception as e:
        logger.info(f"[Retriever] {e}")

    qa_system_prompt = (
        "You are an assistant for question-answering tasks. Use the following "
        "pieces of retrieved context to answer the question. If you don't know "
        "the answer, just say that you don't know. Use three sentences maximum and "
        "keep the answer concise."
    )

    stuff_prompt = PromptTemplate(
        template="""You are an assistant for question-answering tasks. 
        Use the following pieces of context to answer the question. If you don't know the answer, just say that you don't know.
        Context:
        {context}
        Question:
        {question}
        Answer:""",
        input_variables=["context", "question"]
    )

    try:
        combine_docs_chain = create_stuff_documents_chain(llm, stuff_prompt)
    except Exception as e:
        logger.info(f"[Chain] {e}")

    retrieval_chain = RunnableMap({
                        "context": lambda x: retriever.invoke(x["input"]),
                        "question": lambda x: x["input"]
                    }) | combine_docs_chain


    graph_builder = StateGraph(AgentState)

    def call_model(state: AgentState) -> dict:
        logger.info(f"[call_model] Full state: {state}")
        question = state.get("question", "")
        logger.info(f"[call_model] Extracted question: {question}")

        if not question:
            return {
                "messages": [AIMessage(content="[call_model] Empty or missing question.")],
                "graph_output": "[call_model] Empty or missing question."
            }

        chat_history = state.get("messages", [])

        chain_input = {
            "input": question,
            "chat_history": chat_history
        }

        chain_output = retrieval_chain.invoke(chain_input)
        answer_text = chain_output

        logger.info(f"[Agent] Response: {answer_text}")

        return {
            "messages": [AIMessage(content=answer_text or "[call_model] No answer generated.")],
            "graph_output": answer_text or "[call_model] No answer generated."
        }

    graph_builder.add_node("model", call_model)
    graph_builder.set_entry_point("model")
    graph_builder.add_edge("model", END)

    memory = MemorySaver()
    app = graph_builder.compile(checkpointer=memory)
    return app
