import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import os
from dotenv import load_dotenv
from app.builder import build_chunks, build_engine

load_dotenv()

DATA_PATH   = os.getenv("DATA_PATH", "data/")
ENGINES_DIR = os.getenv("ENGINES_DIR", "engines/")
BATCH_SIZE  = int(os.getenv("BATCH_SIZE", "64"))
SIMILARITY  = os.getenv("SIMILARITY", "cosine")  # "cosine" or "ip"
VectorDB    = os.getenv("VectorDB", "FAISS")  # "FAISS" or "QUANTA"

MODELS = [
    {
        "name":  "Orpheas",
        "model": "novelcore/Orpheas",
    },
    {
        "name":  "mpnet",
        "model": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    },            
    # {
    #     "name":  "Themida",
    #     "model": "novelcore/Themida",
    # },    
    # {
    #     "name":  "MiniLM",
    #     "model": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    # }    
]

# ── Step 1: Build chunks once (shared across all engines) ─────────────────────
print(f"\n{'='*60}")
print(f"  Building shared chunks")
print(f"  Data   : {DATA_PATH}")
print(f"  Output : {os.path.join(ENGINES_DIR, 'chunks.json')}")
print(f"  VectorDB : {VectorDB}")
print(f"{'='*60}\n")

if not os.path.exists(os.path.join(ENGINES_DIR, "chunks.json")):
    build_chunks(data_path=DATA_PATH, engines_dir=ENGINES_DIR)

# ── Step 2: Build each engine using the shared chunks ─────────────────────────
for cfg in MODELS:
    output_dir = os.path.join(ENGINES_DIR, cfg["name"])

    print(f"\n{'='*60}")
    print(f"  Engine : {cfg['name']}")
    print(f"  Model  : {cfg['model']}")
    print(f"  Output : {output_dir}")
    print(f"{'='*60}\n")

    build_engine(
        engines_dir = ENGINES_DIR,
        engine_name = cfg["name"],
        model_name  = cfg["model"],
        batch_size  = BATCH_SIZE,
        similarity  = SIMILARITY,
        vector_db   = VectorDB,
    )

print("\n✓ All engines built successfully.")
