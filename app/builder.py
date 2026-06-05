import os
import json
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer


def make_embedding_text(chunk: dict) -> str:
    parts = []
    if chunk.get("header"):
        parts.append(chunk["header"])
    parts.append(f"Αριθμός φύλλου: {chunk['sheet_number']}")
    parts.append(f"ΦΕΚ Id: {chunk['fek_id']}")
    if chunk.get("text"):
        parts.append(chunk["text"])
    return "\n".join(parts)


def build_chunks(data_path: str, engines_dir: str) -> str:
    """
    Walk *data_path*, parse every JSON file, and produce the shared
    ``engines_dir/chunks.json``.  Returns the path to that file.

    This step is model-agnostic and must be run only once regardless of
    how many embedding models will be built afterwards.
    """
    os.makedirs(engines_dir, exist_ok=True)

    # ── Collect filenames ─────────────────────────────────────────────────────
    filenames = []
    for root, dirs, files in os.walk(data_path):
        filenames += [os.path.join(root, file) for file in files]
    filenames.sort()
    filenames = filenames[:10]
    # ── Parse files → chunks ──────────────────────────────────────────────────
    chunks: dict[int, dict] = {}
    for filename in tqdm(filenames, desc="Processing files", unit="file"):
        with open(filename, "r", encoding="utf-8") as f:
            d = json.load(f)

        issue        = d["issue"]
        sheet_number = d["sheet_number"]
        fek_id       = d["fek_id"]
        date         = f"{d['year']}-{d['month']}"

        for kad in d["kads"]:
            if kad["header"] is not None and kad["title"] is not None:
                header = f"{kad['header']}: {kad['title']}"
            elif kad["header"] is None and kad["title"] is not None:
                header = kad["title"]
            elif kad["header"] is not None and kad["title"] is None:
                header = kad["header"] if len(kad["header"]) > 5 else None
            else:
                header = None

            base = {
                "issue":        issue,
                "sheet_number": sheet_number,
                "fek_id":       fek_id,
                "date":         date,
                "header":       header,
            }

            # "consider" section
            if "consider" in kad["sections"] and kad["sections"]["consider"] is not None:
                consider = kad["sections"]["consider"]
                if consider["is_valid"] and consider["text"] and len(consider["text"]) > 5:
                    chunk = base.copy()
                    chunk["text"] = consider["text"]
                    chunks[len(chunks)] = chunk

            # body articles
            for part in kad["sections"]["body"]:
                if not part["is_valid"]:
                    continue

                article_header = part.get("header")
                article_title  = part.get("title")

                if article_header and article_title:
                    prefix = f"{article_header}: {article_title}\n"
                elif article_title:
                    prefix = article_title + "\n"
                elif article_header and len(article_header) > 5:
                    prefix = article_header + "\n"
                else:
                    prefix = ""

                chunk = base.copy()
                chunk["text"] = prefix + part["text"]
                chunks[len(chunks)] = chunk

    print(f"✓ Built {len(chunks):,} chunks from {len(filenames):,} files")

    # ── Save shared chunks ────────────────────────────────────────────────────
    chunks_file = os.path.join(engines_dir, "chunks.json")
    with open(chunks_file, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False)
    print(f"✓ Chunks saved → {chunks_file}")

    return chunks_file


def build_engine(
    engines_dir: str,
    engine_name: str,
    model_name: str,
    batch_size: int = 64,
    similarity: str = "cosine",
    vector_db: str = "FAISS",
) -> None:
    """
    Load the shared ``engines_dir/chunks.json``, embed every chunk with
    *model_name*, build a vector index, and save results to
    ``engines_dir/<engine_name>/``.

    Requires ``build_chunks()`` to have been run first.

    *vector_db* selects the vector store backend:
      - ``"FAISS"``  → builds a FAISS IndexFlatIP (default).
      - ``"QUANTA"`` → builds a QuantaIndex with 4-bit quantization.

    *similarity* selects the metric (applies to both backends):
      - ``"cosine"``  → embeddings are L2-normalized before indexing.
      - ``"ip"`` / ``"inner_product"`` → embeddings are left un-normalized.
    The chosen metric is persisted in ``config.json`` so the searcher applies
    the matching normalization at query time.
    """
    vector_db  = vector_db.upper()
    if vector_db not in ("FAISS", "QUANTA"):
        raise ValueError(f"Unknown vector_db '{vector_db}'. Use 'FAISS' or 'QUANTA'.")

    similarity = similarity.lower()
    if similarity in ("cosine", "cos"):
        normalize = True
    elif similarity in ("ip", "inner_product", "dot"):
        normalize = False
    else:
        raise ValueError(
            f"Unknown similarity '{similarity}'. Use 'cosine' or 'ip'."
        )

    output_dir = os.path.join(engines_dir, engine_name)
    os.makedirs(output_dir, exist_ok=True)

    # ── Step 1: Load shared chunks ────────────────────────────────────────────
    chunks_file = os.path.join(engines_dir, "chunks.json")
    print(f"Loading shared chunks from {chunks_file} …")
    with open(chunks_file, "r", encoding="utf-8") as f:
        chunks: dict[int, dict] = {int(k): v for k, v in json.load(f).items()}
    print(f"✓ {len(chunks):,} chunks loaded")

    # ── Step 2: Embed ─────────────────────────────────────────────────────────
    print(f"\nLoading model: {model_name}")
    model = SentenceTransformer(model_name)

    texts = [make_embedding_text(chunks[i]) for i in range(len(chunks))]

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=normalize,
        convert_to_numpy=True,
    ).astype(np.float32)

    dim = embeddings.shape[1]
    print(f"✓ Embeddings shape: {embeddings.shape}")
    print(f"✓ Metric: {'cosine' if normalize else 'inner_product'} (normalize={normalize})")

    # ── Step 3: Build & save index ────────────────────────────────────────────
    if vector_db == "FAISS":
        import faiss
        # IndexFlatIP computes inner products. With normalized embeddings that is
        # cosine similarity; with raw embeddings it is the plain dot product.
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)

        index_file = os.path.join(output_dir, "faiss_index.bin")
        faiss.write_index(index, index_file)
        print(f"✓ FAISS index saved → {index_file}  ({index.ntotal:,} vectors, dim={dim})")

    elif vector_db == "QUANTA":
        from quanta import QuantaIndex
        ids = [str(i) for i in range(len(embeddings))]
        idx = QuantaIndex(name=engine_name, dim=dim, bit_width=4, index_dir=output_dir)
        idx.add(embeddings, ids)
        idx.save()
        print(f"✓ QUANTA index saved → {output_dir}  ({len(embeddings):,} vectors, dim={dim})")

    # ── Step 4: Save engine config ────────────────────────────────────────────
    config = {
        "model":      model_name,
        "dim":        dim,
        "num_chunks": len(chunks),
        "similarity": "cosine" if normalize else "inner_product",
        "normalize":  normalize,
        "vector_db":  vector_db,
    }
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    print(f"✓ Config saved → {os.path.join(output_dir, 'config.json')}")
