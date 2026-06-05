import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer


class SearchEngine:
    def __init__(self, engine_name: str, engines_dir: str = "engines/"):
        self.name      = engine_name
        engine_dir     = os.path.join(engines_dir, engine_name)

        # Config
        with open(os.path.join(engine_dir, "config.json")) as f:
            config = json.load(f)
        self.model_name = config["model"]
        # Whether embeddings were L2-normalized at build time. The query must be
        # normalized the same way so scores stay consistent.
        # Defaults to True for engines built before this field existed.
        self.normalize  = config.get("normalize", True)
        self.vector_db  = config.get("vector_db", "FAISS").upper()

        # Shared chunks — stored once in engines_dir root, not per-engine
        print(f"[{engine_name}] Loading chunks …")
        chunks_file = os.path.join(engines_dir, "chunks.json")
        with open(chunks_file, "r", encoding="utf-8") as f:
            raw = json.load(f)
        self.chunks: dict[int, dict] = {int(k): v for k, v in raw.items()}

        # Index
        if self.vector_db == "FAISS":
            import faiss
            print(f"[{engine_name}] Loading FAISS index …")
            self.index = faiss.read_index(os.path.join(engine_dir, "faiss_index.bin"))
            print(f"[{engine_name}] ✓ Ready — {self.index.ntotal:,} vectors\n")

        elif self.vector_db == "QUANTA":
            from quanta import QuantaIndex
            print(f"[{engine_name}] Loading QUANTA index …")
            self.index = QuantaIndex.load(engine_name, index_dir=engine_dir)
            print(f"[{engine_name}] ✓ Ready\n")

        else:
            raise ValueError(f"Unknown vector_db '{self.vector_db}' in config.")

        # Embedding model
        print(f"[{engine_name}] Loading model: {self.model_name}")
        self.model = SentenceTransformer(self.model_name)

    @property
    def ntotal(self) -> int:
        if self.vector_db == "FAISS":
            return self.index.ntotal
        return len(self.chunks)

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        query_vec = self.model.encode(
            [query],
            normalize_embeddings=self.normalize,
            convert_to_numpy=True,
        ).astype(np.float32)

        results = []

        if self.vector_db == "FAISS":
            scores, indices = self.index.search(query_vec, top_k)
            for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), start=1):
                if idx == -1:
                    break
                chunk = self.chunks[idx].copy()
                chunk["rank"]  = rank
                chunk["score"] = round(float(score), 6)
                results.append(chunk)

        elif self.vector_db == "QUANTA":
            raw_results = self.index.search(query_vec[0], k=top_k)
            for rank, r in enumerate(raw_results, start=1):
                chunk = self.chunks[int(r.id)].copy()
                chunk["rank"]  = rank
                chunk["score"] = round(float(r.score), 6)
                results.append(chunk)

        return results
