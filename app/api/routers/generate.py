"""
generate.py
-----------
Endpoints that pick a random chunk from the loaded corpus and use the
OpenAI Chat API to produce a realistic evaluation query from it.

Two query "styles" are exposed, both returning the same response shape:
    POST /generate/query     – a natural-language question (paraphrased)
    POST /generate/keywords  – a comma-separated list of 4–6 keyphrases

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

# ── System prompts ────────────────────────────────────────────────────────────
# Task 1 — natural-language question. Methodology adapted from the dataset
# question-generation template: paraphrase the source, prefer informal terms
# and synonyms, answerable from the context alone.
_QUESTION_PROMPT = """\
Είσαι ένας Καθηγητής/Εξεταστής.
Δίνεται ένα απόσπασμα από την Εφημερίδα της Κυβερνήσεως (ΦΕΚ).
Με βάση τις πληροφορίες του αποσπάσματος και χωρίς προηγούμενη γνώση, ετοίμασε ΜΙΑ \
ερώτηση για ένα επερχόμενο διαγώνισμα/εξέταση, η οποία μπορεί να απαντηθεί διαβάζοντας \
το απόσπασμα. Περιόρισε την ερώτηση στις πληροφορίες που παρέχονται στο απόσπασμα.

# Ορισμός
Ερώτηση:
  - Σύνθεσε την ερώτηση παραφράζοντας το αρχικό κείμενο, ώστε να είναι περιεκτική και πλήρης.
  - Μην χρησιμοποιείς επίσημους όρους όπως 'Ελληνική Δημοκρατία' και προτίμησε ανεπίσημους \
όρους όπως 'Ελλάδα'. Προτίμησε συνώνυμα του αρχικού κειμένου όπου είναι δυνατόν.
  - Η ερώτηση πρέπει να είναι στα Ελληνικά.

Απάντησε ΜΟΝΟ σε έγκυρο JSON με το πεδίο "query" (string).\
"""

# Task 2 — comma-separated keyphrases. Methodology adapted from the dataset
# entity-extraction template: prominent entities taken (largely) unaltered.
_KEYWORDS_PROMPT = """\
Είσαι ένας Καθηγητής/Εξεταστής.
Δίνεται ένα απόσπασμα από την Εφημερίδα της Κυβερνήσεως (ΦΕΚ).
Με βάση τις πληροφορίες του αποσπάσματος και χωρίς προηγούμενη γνώση, δημιούργησε ένα \
query της μορφής "<όρος 1>, <όρος 2>, ..., <όρος ν>" που περιέχει λέξεις-κλειδιά / \
φράσεις-κλειδιά από το απόσπασμα.

# Ορισμός
Λέξεις-κλειδιά / Οντότητες:
  - Σχετικά εξέχουσες οντότητες μέσα στο κείμενο, λαμβανόμενες χωρίς αλλοιώσεις.
  - Από 4 έως 6 όροι, χωρισμένοι μεταξύ τους με κόμμα.
  - Η απάντηση πρέπει να είναι στα Ελληνικά.

Απάντησε ΜΟΝΟ σε έγκυρο JSON με το πεδίο "query" (string).\
"""

_TEXT_PREVIEW_LEN = 2_500   # chars sent to the model (keeps OpenAI cost low)


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
        "Δημιούργησε query για το παρακάτω απόσπασμα:\n\n"
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


async def _generate_from_corpus(request: Request, system_prompt: str) -> JSONResponse:
    """
    Pick a random chunk from the shared corpus, call OpenAI with *system_prompt*
    to generate a query, then return the query plus the source chunk's metadata
    for traceability.
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
                {"role": "system", "content": system_prompt},
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
            "query": result.get("query", ""),
            "chunk": {
                "chunk_id":     chunk_id,
                "fek_id":       chunk.get("fek_id"),
                "sheet_number": chunk.get("sheet_number"),
                "issue":        chunk.get("issue"),
                "date":         chunk.get("date"),
                "header":       chunk.get("header"),
                "text":         chunk.get("text", ""),   # full text, no truncation
            },
        }
    )


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post(
    "/query",
    summary="Generate a random natural-language evaluation question from the corpus",
    response_description="Generated question and source chunk metadata",
)
async def generate_query(request: Request) -> JSONResponse:
    """
    Picks a random chunk from the shared corpus and calls OpenAI to generate a
    realistic, paraphrased question that can be answered from the chunk, then
    returns it along with the source chunk's metadata.
    """
    return await _generate_from_corpus(request, _QUESTION_PROMPT)


@router.post(
    "/keywords",
    summary="Generate a random keyword/keyphrase evaluation query from the corpus",
    response_description="Generated comma-separated keyphrases and source chunk metadata",
)
async def generate_keywords(request: Request) -> JSONResponse:
    """
    Picks a random chunk from the shared corpus and calls OpenAI to generate a
    query made of 4–6 prominent keyphrases/entities (comma-separated) drawn from
    the chunk, then returns it along with the source chunk's metadata.
    """
    return await _generate_from_corpus(request, _KEYWORDS_PROMPT)
