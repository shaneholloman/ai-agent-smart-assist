from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from typing import List, Optional
from pathlib import Path
import os

from langchain_ai_agent.ingestion.reader import DocumentIngestor
from langchain_ai_agent.retriever.vector_store import DocumentEmbedder

router = APIRouter()

@router.post("/api/ingest")
async def ingest_files(
    files: List[UploadFile] = File(...),
    namespace: Optional[str] = Query(default="default", description="Namespace for FAISS index")
):
    try:
        upload_dir = Path("tmp_uploads")
        upload_dir.mkdir(exist_ok=True)

        for file in files:
            file_path = upload_dir / file.filename
            with open(file_path, "wb") as f:
                f.write(await file.read())

        ingestor = DocumentIngestor()
        all_chunks = ingestor.process_directory(upload_dir)

        if not all_chunks:
            raise HTTPException(status_code=400, detail="No valid chunks extracted.")

        embedder = DocumentEmbedder(persist_dir=f"faiss_index/{namespace}")
        embedder.build_or_update_index(all_chunks)

        return {
            "status": "success",
            "num_chunks": len(all_chunks),
            "namespace": namespace
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
