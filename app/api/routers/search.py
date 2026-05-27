from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

router = APIRouter(prefix="/search", tags=["search"])


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, examples=["Αμοιβαία Αναγνώριση Αδειών Οδήγησης"])
    top_k: int = Field(default=5, ge=1, le=50)


def _run_search(engine_name: str, body: SearchRequest, request: Request) -> JSONResponse:
    engine  = request.app.state.engines[engine_name]
    results = engine.search(body.query, body.top_k)
    return JSONResponse(content={"engine": engine_name, "query": body.query, "results": results})


@router.post(
    "/mpnet",
    summary="Search with paraphrase-multilingual-mpnet-base-v2",
    response_description="Ranked list of matching chunks with metadata",
)
def search_mpnet(body: SearchRequest, request: Request) -> JSONResponse:
    return _run_search("mpnet", body, request)


@router.post(
    "/novelcore",
    summary="Search with novelcore/embeddings-model",
    response_description="Ranked list of matching chunks with metadata",
)
def search_novelcore(body: SearchRequest, request: Request) -> JSONResponse:
    return _run_search("novelcore", body, request)
