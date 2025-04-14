from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_google_vertexai import ChatVertexAI
from langchain_ai_agent.retriever.vector_store import DocumentEmbedder
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, RemoveMessage
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.checkpoint.memory import MemorySaver
from typing import Annotated, Sequence, Literal
import operator
import logging

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Shared persistent memory store
store = MemorySaver()

# Use MessagesState for built-in message memory support
class AgentState(MessagesState):
    summary: str | None = None
    question: str
    graph_output: str | None = None


def create_prompt():
    system_instruction = """Given a chat history and the latest user question \
        which might reference context in the chat history, formulate a standalone question \
        which can be understood without the chat history. Do NOT answer the question, \
        just reformulate it if needed and otherwise return it as is."""

    return ChatPromptTemplate.from_messages([
        ("system", system_instruction),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])

def create_doc_chains_prompt():
    return PromptTemplate(
        template="""You are an assistant for question-answering tasks. 
        Use the following pieces of context to answer the question. If you don't know the answer, just say that you don't know.
        Context:
        {context}
        Question:
        {input}
        Answer:""",
        input_variables=["context", "input"]
    )

def get_chat_agent_with_memory(persist_dir: str):
    embedder = DocumentEmbedder(persist_dir=persist_dir)
    retriever = embedder.get_retriever(k=10)

    llm = ChatVertexAI(
        model_name="gemini-2.0-flash-lite",
        temperature=0.3,
        max_output_tokens=1024,
    )

    contextualize_q_prompt = create_prompt()
    try:
        history_aware_retriever = create_history_aware_retriever(
            llm, retriever, contextualize_q_prompt
        )
    except Exception as e:
        logger.info(f"[Retriever] {e}")

    stuff_prompt = create_doc_chains_prompt()
    try:
        combine_docs_chain = create_stuff_documents_chain(llm, stuff_prompt)
    except Exception as e:
        logger.info(f"[Chain] {e}")

    retrieval_chain = create_retrieval_chain(history_aware_retriever, combine_docs_chain)

    workflow = StateGraph(AgentState)

    def call_model(state: AgentState) -> dict:
        logger.info(f"[call_model] Full state: {state}")
        question = state.get("question", "")
        summary = state.get("summary", "")

        if not question:
            return {
                "messages": [AIMessage(content="[call_model] Empty or missing question.")],
                "graph_output": "[call_model] Empty or missing question."
            }

        system_messages = [SystemMessage(content=f"Summary of conversation earlier: {summary}")]
        chat_history = system_messages + state["messages"] if summary else state["messages"]

        logger.info(f"We have this information {chat_history}")

        chain_input = {
            "input": question,
            "chat_history": chat_history
        }

        logger.info(f"[call_model] chain_input: {chain_input}")
        chain_output = retrieval_chain.invoke(chain_input)
        answer_text = chain_output['answer']

        updated_messages = state.get("messages", []) + [
            HumanMessage(content=question),
            AIMessage(content=answer_text)
        ]

        return {
            "messages": updated_messages,
            "graph_output": answer_text
        }

    def should_continue(state: AgentState) -> Literal["summarize_conversation", END]:
        if len(state.get("messages", [])) > 6:
            return "summarize_conversation"
        return END

    def summarize_conversation(state: AgentState) -> dict:
        summary = state.get("summary", "")
        prompt = (
            f"This is summary of the conversation to date: {summary}\n\n"
            "Extend the summary by taking into account the new messages above:"
            if summary
            else "Create a summary of the conversation above:"
        )
        messages = state["messages"] + [HumanMessage(content=prompt)]
        response = llm.invoke(messages)
        delete_messages = [RemoveMessage(id=m.id) for m in state["messages"][:-2]]

        return {
            "summary": response.content,
            "messages": delete_messages
        }

    # Build the graph
    workflow.add_node("conversation", call_model)
    workflow.add_node("summarize_conversation", summarize_conversation)
    workflow.set_entry_point("conversation")
    workflow.add_conditional_edges("conversation", should_continue)
    workflow.add_edge("summarize_conversation", END)

    workflow = workflow.compile(checkpointer=store)
    logger.info("[LangGraph] Compiled with MemorySaver store")
    return workflow

