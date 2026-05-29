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
Δημιούργησε ένα ρεαλιστικό query (3–15 λέξεις) που ένας χρήστης θα έγραφε \
για να εντοπίσει αυτό το συγκεκριμένο απόσπασμα — φυσική γλώσσα, χωρίς boolean operators.

ΣΗΜΑΝΤΙΚΟ: Το query πρέπει να αφορά το ΠΕΡΙΕΧΟΜΕΝΟ του συγκεκριμένου άρθρου/αποσπάσματος \
και ΟΧΙ τον γενικό τίτλο του νόμου. Αγνόησε θέματα του νόμου που δεν σχετίζονται με το \
συγκεκριμένο απόσπασμα.

Παράδειγμα:
Είσοδος:
  Τίτλος: ΝΟΜΟΣ ΥΠ' ΑΡΙΘΜ. 4964: Διατάξεις για την απλοποίηση της περιβαλλοντικής \
αδειοδότησης, θέσπιση πλαισίου για την ανάπτυξη των Υπεράκτιων Αιολικών Πάρκων, \
την αντιμετώπιση της ενεργειακής κρίσης, την προστασία του περιβάλλοντος και λοιπές διατάξεις.
  Κείμενο: Άρθρο 104: Ασφάλιση εργατών σμύριδας — καταβολή σμυριγδεργατικού δικαιώματος \
και ασφαλιστικών εισφορών για το έτος 2022 για εργασίες περισυλλογής και διαλογής σμύριδας \
στο Καμπί Απειράνθου Νάξου.

Λάθος query (αναφέρεται στον τίτλο του νόμου, όχι στο άρθρο):
  "Διατάξεις για τη θέσπιση πλαισίου για την ανάπτυξη των Αιολικών Πάρκων"

Σωστό query (αναφέρεται στο περιεχόμενο του άρθρου):
  "Ασφαλιστικές εισφορές σμυριδεργατών Νάξου για το 2022"

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
