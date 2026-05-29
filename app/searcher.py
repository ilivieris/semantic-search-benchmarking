import os
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer


class FEKSearchEngine:
    def __init__(self, engine_name: str, engines_dir: str = "engines/"):
        self.name  = engine_name
        engine_dir = os.path.join(engines_dir, engine_name)

        # Config
        with open(os.path.join(engine_dir, "config.json")) as f:
            config = json.load(f)
        self.model_name = config["model"]
        # Whether embeddings were L2-normalized at build time. The query must be
        # normalized the same way so IndexFlatIP scores stay consistent
        # (normalized → cosine, raw → inner product). Defaults to True for
        # engines built before this field existed.
        self.normalize = config.get("normalize", True)

        # Shared chunks — stored once in engines_dir root, not per-engine
        print(f"[{engine_name}] Loading chunks …")
        chunks_file = os.path.join(engines_dir, "chunks.json")
        with open(chunks_file, "r", encoding="utf-8") as f:
            raw = json.load(f)
        self.chunks: dict[int, dict] = {int(k): v for k, v in raw.items()}

        # FAISS index
        print(f"[{engine_name}] Loading FAISS index …")
        self.index = faiss.read_index(os.path.join(engine_dir, "faiss_index.bin"))

        # Embedding model
        print(f"[{engine_name}] Loading model: {self.model_name}")
        self.model = SentenceTransformer(self.model_name)

        print(f"[{engine_name}] ✓ Ready — {self.index.ntotal:,} vectors\n")

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        query_vec = self.model.encode(
            [query],
            normalize_embeddings=self.normalize,
            convert_to_numpy=True,
        ).astype(np.float32)

        scores, indices = self.index.search(query_vec, top_k)

        results = []
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), start=1):
            if idx == -1:
                break
            chunk = self.chunks[idx].copy()
            chunk["rank"]  = rank
            chunk["score"] = round(float(score), 6)
            results.append(chunk)

        return results
