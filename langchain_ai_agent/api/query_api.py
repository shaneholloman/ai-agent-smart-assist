from fastapi import APIRouter, Query, HTTPException
from fastapi.responses import StreamingResponse
from langchain_ai_agent.agents.chat_agent import get_chat_agent_with_memory
import traceback, logging
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

router = APIRouter()

@router.get("/api/query")
async def query_kb(question: str = Query(...), namespace: str = Query("default")):
    try:
        agent = get_chat_agent_with_memory(persist_dir=f"faiss_index/{namespace}")
        result = await agent.ainvoke(
            {"question": question},
            config={"configurable": {"thread_id": "query-session"}}
        )

        answer = result.get("graph_output", "").strip()
        if not answer:
            raise HTTPException(status_code=500, detail="Agent returned no answer.")

        # ✅ Send response as JSON with a results array
        return JSONResponse(content={"results": [answer]})

    except Exception as e:
        logger.error(f"Agent execution failed: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))
