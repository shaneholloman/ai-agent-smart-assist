from fastapi import APIRouter, Query, HTTPException
from langchain_ai_agent.retriever.vector_store import DocumentEmbedder

router = APIRouter()

@router.get("/api/query")
def query_kb(question: str = Query(...), namespace: str = Query("default")):
    try:
        embedder = DocumentEmbedder(persist_dir=f"faiss_index/{namespace}")
        docs = embedder.query(question)
        return {"results": [doc.page_content for doc in docs]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
