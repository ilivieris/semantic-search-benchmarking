import os
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from app.searcher import FEKSearchEngine
from app.api.routers import search

BASE_DIR = Path(__file__).parent.parent.parent  # project root

load_dotenv()

ENGINES_DIR = os.getenv("ENGINES_DIR", "engines/")
ENGINES     = ["mpnet", "novelcore"]


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading search engines …")
    app.state.engines = {
        name: FEKSearchEngine(name, engines_dir=ENGINES_DIR)
        for name in ENGINES
    }
    print("✓ All engines ready\n")
    yield


app = FastAPI(
    title="FEK Search API",
    description="Semantic search over Greek Government Gazette (ΦΕΚ) using FAISS.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

app.include_router(search.router)

# ── Static UI ─────────────────────────────────────────────────────────────────
app.mount("/ui", StaticFiles(directory=str(BASE_DIR / "ui"), html=True), name="ui")


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    return RedirectResponse(url="/ui/")


@app.get("/health", tags=["meta"])
def health() -> dict:
    engines_info = {
        name: {"ntotal": eng.index.ntotal, "model": eng.model_name}
        for name, eng in app.state.engines.items()
    }
    return {"status": "ok", "engines": engines_info}
