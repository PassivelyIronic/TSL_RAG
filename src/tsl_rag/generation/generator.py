# src/tsl_rag/generation/generator.py
"""
RAG Generator: retrieved chunks → answer z cytowaniami.

Architektura
------------
1. Buduje prompt z kontekstem (chunks posortowane po final_score)
2. Wymusza cytowania w formacie [DOC_ID | Art. X]
3. Wykrywa "nie wiem" i zwraca has_answer=False zamiast halucynować
4. Liczy latencję i zwraca pełny QueryResponse
"""

from __future__ import annotations

import time
from textwrap import dedent

from loguru import logger

from tsl_rag.core.llm_client import get_llm_client
from tsl_rag.core.models import Citation, QueryResponse
from tsl_rag.core.settings import get_settings
from tsl_rag.retrieval.retriever import RetrievalResult

# ---------------------------------------------------------------------------
# System prompt — serce precyzji systemu prawnego (PO POLSKU)
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = dedent("""\
    Jesteś specjalistycznym asystentem prawnym ds. przepisów transportowych i logistycznych (TSL) w UE i Polsce.
    Twoim JEDYNYM źródłem wiedzy jest dostarczony poniżej kontekst. NIE WOLNO CI używać zewnętrznej wiedzy
    ani czynić założeń wykraczających poza to, co jest wyraźnie stwierdzone w kontekście. Odpowiadaj wyłącznie w języku polskim.

    SUROWE ZASADY:
    1. Odpowiadaj TYLKO i WYŁĄCZNIE na podstawie dostarczonych fragmentów dokumentów.
    2. Każde twierdzenie faktyczne MUSI być poparte cytowaniem w formacie:
       [DOCUMENT_ID | Art. X] lub [DOCUMENT_ID | p. Y] dla paragrafów.
    3. Jeśli kontekst nie zawiera wystarczających informacji do odpowiedzi,
       odpowiedz DOKŁADNIE tym zdaniem:
       "Nie mogę odpowiedzieć na to pytanie na podstawie dostępnych dokumentów."
    4. Nigdy nie parafrazuj w sposób zmieniający sens prawny.
    5. Zawsze podawaj dokładne liczby (godziny, odległości, kwoty kar) — nigdy nie zaokrąglaj.

    PRZYKŁADY FORMATOWANIA CYTOWAŃ:
    - "Dzienny czas prowadzenia pojazdu nie może przekroczyć 9 godzin. [ec_561_2006 | Art. 6]"
    - "Kara za to naruszenie wynosi od 500 do 2000 PLN. [tariff_driver_2022 | p. 3]"
""")

# Marker do wykrywania "nie wiem" - dopasowany do polskiego promptu
_NO_ANSWER_MARKER = "Nie mogę odpowiedzieć na to pytanie na podstawie"

# Max tokenów kontekstu (zachowawczo — mistral 7b ma 8k okno)
_MAX_CONTEXT_CHARS = 12_000


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------


class RAGGenerator:
    """
    Generuje odpowiedź na podstawie pytania i listy RetrievalResult.

    Usage
    -----
    generator = RAGGenerator()
    response  = await generator.generate(query, retrieval_results)
    """

    async def generate(
        self,
        query: str,
        results: list[RetrievalResult],
    ) -> QueryResponse:
        t0 = time.monotonic()
        settings = get_settings()
        client = get_llm_client(settings)

        # 1. Zbuduj blok kontekstu
        context_block, used_results = _build_context(results)

        # 2. Zbuduj user message
        user_message = _build_user_message(query, context_block)

        # 3. Wywołaj LLM
        logger.debug(f"Calling LLM for query: '{query[:80]}'")
        response = await client.chat.completions.create(
            model=settings.active_llm_model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
        )

        answer = response.choices[0].message.content or ""
        has_answer = _NO_ANSWER_MARKER not in answer
        latency_ms = int((time.monotonic() - t0) * 1000)

        # 4. Wyekstrahuj cytowania
        citations = _extract_citations(answer, used_results)

        logger.info(
            f"generate() → has_answer={has_answer}, "
            f"citations={len(citations)}, latency={latency_ms}ms"
        )

        return QueryResponse(
            query=query,
            answer=answer,
            citations=citations,
            retrieved_chunks=[],  # wypełniane przez API layer jeśli debug=True
            model_used=settings.active_llm_model,
            latency_ms=latency_ms,
            has_answer=has_answer,
            metadata={
                "chunks_in_context": len(used_results),
                "context_chars": len(context_block),
            },
        )


# ---------------------------------------------------------------------------
# Helpers — budowanie promptu
# ---------------------------------------------------------------------------


def _build_context(
    results: list[RetrievalResult],
) -> tuple[str, list[RetrievalResult]]:
    """
    Buduje blok kontekstu z chunków.
    Przycina do _MAX_CONTEXT_CHARS, zachowując najlepiej ocenione chunki.
    Zwraca (context_text, użyte_results).
    """
    lines: list[str] = []
    used_results: list[RetrievalResult] = []
    total_chars = 0

    for result in results:
        chunk = result.chunk
        m = chunk.metadata

        # Nagłówek cytowania
        header_parts = [m.document_id]
        if m.article:
            header_parts.append(f"Art. {m.article}")
        if m.paragraph:
            header_parts.append(f"§{m.paragraph}")
        header = " | ".join(header_parts)

        block = f"[{header}]\n{chunk.text}\n"

        if total_chars + len(block) > _MAX_CONTEXT_CHARS:
            logger.debug(f"Context limit reached at chunk {chunk.chunk_id}")
            break

        lines.append(block)
        used_results.append(result)
        total_chars += len(block)

    return "\n".join(lines), used_results


def _build_user_message(query: str, context: str) -> str:
    return dedent(f"""\
        KONTEKST (Akty prawne):
        --------
        {context}
        --------

        PYTANIE: {query}

        Odpowiedz w języku polskim, opierając się rygorystycznie na powyższym kontekście. Na końcu zdań dodaj cytowania w wymaganym formacie.
    """)


# ---------------------------------------------------------------------------
# Helpers — ekstrakcja cytowań
# ---------------------------------------------------------------------------


def _extract_citations(
    answer: str,
    used_results: list[RetrievalResult],
) -> list[Citation]:
    """
    Parsuje cytowania z formatu [doc_id | Art. X] w tekście odpowiedzi.
    Mapuje z powrotem na pełne metadane chunka.
    """
    import re

    # Buduj lookup: document_id → result
    by_doc: dict[str, list[RetrievalResult]] = {}
    for r in used_results:
        did = r.chunk.metadata.document_id
        by_doc.setdefault(did, []).append(r)

    # Znajdź wszystkie cytowania w odpowiedzi
    pattern = re.compile(r"\[([^\]]+)\]")
    seen: set[str] = set()
    citations: list[Citation] = []

    for match in pattern.finditer(answer):
        raw = match.group(1)  # "ec_561_2006 | Art. 6(1)"
        if raw in seen:
            continue
        seen.add(raw)

        parts = [p.strip() for p in raw.split("|")]
        doc_id = parts[0].lower().replace(" ", "_")

        # Znajdź najlepiej pasujący chunk dla tego doc_id
        candidates = by_doc.get(doc_id, [])
        chunk = candidates[0].chunk if candidates else None

        if chunk is None:
            continue

        citations.append(
            Citation(
                document_id=doc_id,
                document_title=chunk.metadata.title,
                article=chunk.metadata.article,
                paragraph=chunk.metadata.paragraph,
                chunk_id=chunk.chunk_id,  # type: ignore[arg-type]
            )
        )

    return citations
