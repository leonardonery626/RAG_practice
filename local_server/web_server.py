import logging

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from func.retriever_builder import RetrieverBuilder as Retriever
from func.generator_builder import Generator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

app = FastAPI(title="RAG Retriever")

# ---------------------------------------------------------------------------
# Request schemas
# ---------------------------------------------------------------------------

class RetrievalRequest(BaseModel):
    prompt: str

class GenerationRequest(BaseModel):
    prompt: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.post("/rag/retrieval")
def retrieval_endpoint(retrieval_request: RetrievalRequest):
    retriever = Retriever(prompt=retrieval_request.prompt)
    logger.info("Starting retrieval for prompt: %s", retrieval_request.prompt)
    try:
        results_list = retriever.retrieved_context()
        logger.info("Retrieval completed successfully. Retrieved %d chunks.", len(results_list))
        return results_list
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retrieval failed: {e}") from e


@app.post("/rag/generate")
def generate_endpoint(generation_request: GenerationRequest):
    generator = Generator(prompt=generation_request.prompt)
    logger.info("Starting generation for prompt: %s", generation_request.prompt)
    try:
        result = generator.run()
        return {"generated_answer": result["answer"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {e}") from e


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    uvicorn.run(
        "web_server:app",
        host="127.0.0.1",
        port=8011,
        reload=False,
    )


if __name__ == "__main__":
    main()

