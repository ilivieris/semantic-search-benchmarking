"""
generate.py
-----------
Endpoint that picks a random chunk from the loaded corpus and uses the
OpenAI Chat API to produce a realistic evaluation query + expected answer.

Environment variables (set in .env):
    OPENAI_API_KEY   – required
    OPENAI_MODEL     – optional, default "gpt-4o-mini"
"""

import json
import os
import random

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from openai import AsyncOpenAI, OpenAIError

router = APIRouter(prefix="/generate", tags=["generate"])

# ── System prompt ─────────────────────────────────────────────────────────────
_SYSTEM_PROMPT = """\
Είσαι ειδικός στη δημιουργία ερωτήσεων για αξιολόγηση search engines.
Δίνεται ένα απόσπασμα από την Εφημερίδα της Κυβερνήσεως (ΦΕΚ).
Δημιούργησε:
1. Ένα ρεαλιστικό query (3–15 λέξεις) που ένας χρήστης θα έγραφε \
για να εντοπίσει αυτό το έγγραφο — φυσική γλώσσα, χωρίς boolean operators.
2. Μια σύντομη, ακριβή απάντηση (1–3 προτάσεις) βασισμένη \
αποκλειστικά στο δοθέν κείμενο.

Απάντησε ΜΟΝΟ σε έγκυρο JSON με τα πεδία "query" (string) και "answer" (string).\
"""

_TEXT_PREVIEW_LEN   = 2_500   # chars sent to the model
_SOURCE_PREVIEW_LEN = 400     # chars returned to the client


# ── Helpers ───────────────────────────────────────────────────────────────────

def _build_user_message(chunk: dict) -> str:
    parts: list[str] = []
    if chunk.get("header"):
        parts.append(f"Τίτλος: {chunk['header']}")
    parts.append(
        f"ΦΕΚ: {chunk.get('fek_id', 'N/A')}  |  "
        f"Φύλλο: {chunk.get('sheet_number', 'N/A')}  |  "
        f"Ημερομηνία: {chunk.get('date', 'N/A')}"
    )
    if chunk.get("text"):
        parts.append(f"\nΚείμενο:\n{chunk['text'][:_TEXT_PREVIEW_LEN]}")
    return (
        "Δημιούργησε query και απάντηση για το παρακάτω απόσπασμα:\n\n"
        + "\n".join(parts)
    )


def _get_openai_client() -> AsyncOpenAI:
    """Return an async OpenAI client, raising 503 if the key is missing."""
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise HTTPException(
            status_code=503,
            detail=(
                "OPENAI_API_KEY is not configured. "
                "Add it to your .env file and restart the server."
            ),
        )
    return AsyncOpenAI(api_key=api_key)


# ── Endpoint ──────────────────────────────────────────────────────────────────

@router.post(
    "/query",
    summary="Generate a random evaluation query + expected answer from the corpus",
    response_description=(
        "Generated query, expected answer, and source chunk metadata"
    ),
)
async def generate_query(request: Request) -> JSONResponse:
    """
    Picks a random chunk from the shared corpus, calls OpenAI to generate a
    realistic search query and the corresponding expected answer, then returns
    both along with the source chunk's metadata for traceability.
    """
    model  = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    client = _get_openai_client()

    # ── 1. Pick a random chunk ────────────────────────────────────────────────
    engines     = request.app.state.engines
    first_eng   = next(iter(engines.values()))
    chunks: dict = first_eng.chunks

    chunk_id: int = random.choice(list(chunks.keys()))
    chunk: dict   = chunks[chunk_id]

    # ── 2. Call OpenAI ────────────────────────────────────────────────────────
    try:
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user",   "content": _build_user_message(chunk)},
            ],
            response_format={"type": "json_object"},
            temperature=0.7,
            max_tokens=512,
        )
        result: dict = json.loads(response.choices[0].message.content)
    except OpenAIError as exc:
        raise HTTPException(status_code=502, detail=f"OpenAI error: {exc}") from exc
    except (json.JSONDecodeError, KeyError) as exc:
        raise HTTPException(
            status_code=500, detail=f"Failed to parse OpenAI response: {exc}"
        ) from exc

    # ── 3. Return ─────────────────────────────────────────────────────────────
    return JSONResponse(
        content={
            "query":  result.get("query",  ""),
            "answer": result.get("answer", ""),
            "source": {
                "chunk_id":     chunk_id,
                "fek_id":       chunk.get("fek_id"),
                "sheet_number": chunk.get("sheet_number"),
                "issue":        chunk.get("issue"),
                "date":         chunk.get("date"),
                "header":       chunk.get("header"),
                "text_preview": chunk.get("text", "")[:_SOURCE_PREVIEW_LEN],
            },
        }
    )
