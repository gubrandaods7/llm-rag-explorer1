"""retriever.py
Papel na arquitetura:
Este modulo implementa retrieval hibrido no ChromaDB: combina busca semantica
com busca por termos, aplica reranking e retorna contexto com metadados.
"""

from __future__ import annotations

import re
from collections import Counter

from embeddings import EmbeddingServiceError, generate_local_embedding
from config import (
    DEFAULT_TOP_K,
    KEYWORD_SCORE_WEIGHT,
    KEYWORD_SEARCH_MULTIPLIER,
    MAX_DISTANCE_THRESHOLD,
    SEMANTIC_SCORE_WEIGHT,
    SEMANTIC_SEARCH_MULTIPLIER,
)


class RetrievalError(Exception):
    """Erro geral de retrieval semantico."""


def _normalize_tokens(text: str) -> list[str]:
    """Normaliza texto em tokens simples para scoring lexical.

    O que faz:
    - Converte para minusculas.
    - Extrai tokens alfanumericos.

    Por que existe:
    - Busca por keyword precisa comparar termos de maneira consistente.

    Exemplo:
    >>> _normalize_tokens("MRR e ARR")
    ['mrr', 'e', 'arr']
    """
    return re.findall(r"\w+", (text or "").lower(), flags=re.UNICODE)


def _keyword_score(query: str, document: str) -> float:
    """Calcula score lexical simples baseado em sobreposicao de termos.

    O que faz:
    - Compara frequencia dos termos da pergunta no documento.
    - Retorna score normalizado entre 0 e 1.

    Por que existe:
    - Complementa a semantica com correspondencia de termos exatos.

    Exemplo:
    >>> round(_keyword_score("mrr arr", "o mrr contribui para arr"), 2)
    1.0
    """
    query_tokens = _normalize_tokens(query)
    doc_tokens = _normalize_tokens(document)

    if not query_tokens or not doc_tokens:
        return 0.0

    query_counter = Counter(query_tokens)
    doc_counter = Counter(doc_tokens)

    matched = 0
    for token, query_count in query_counter.items():
        matched += min(query_count, doc_counter.get(token, 0))

    return matched / max(1, sum(query_counter.values()))


def _pick_document_summary(collection, source_file: str) -> dict | None:
    """Busca o chunk de resumo global de um arquivo especifico.

    O que faz:
    - Consulta a colecao por metadado source_file e chunk_type=document_summary.
    - Retorna o primeiro resumo encontrado.

    Por que existe:
    - Acrescentar contexto global aumenta percepcao do "arquivo inteiro".

    Exemplo:
    >>> summary = _pick_document_summary(collection, "manual.pdf")
    """
    try:
        data = collection.get(
            where={"$and": [{"source_file": source_file}, {"chunk_type": "document_summary"}]},
            include=["documents", "metadatas"],
        )
    except Exception:
        return None

    docs = data.get("documents") or []
    metas = data.get("metadatas") or []
    if not docs:
        return None

    return {
        "text": docs[0],
        "metadata": metas[0] if metas else {"source_file": source_file, "page_number": 0},
        "distance": None,
        "semantic_score": 0.0,
        "keyword_score": 0.0,
        "combined_score": 0.0,
        "retrieval_path": "document_summary",
    }


def retrieve_relevant_chunks(question: str, collection, top_k: int = DEFAULT_TOP_K) -> list[dict]:
    """Busca chunks relevantes usando retrieval hibrido com reranking.

    O que faz:
    - Faz busca semantica top-N via vetor.
    - Faz busca keyword top-N via matching lexical.
    - Combina scores e aplica reranking hibrido.
    - Anexa resumo global do documento dominante.

    Por que existe:
    - Separa a responsabilidade de retrieval da camada de interface e do LLM.

    Exemplo:
    >>> chunks = retrieve_relevant_chunks("Como resetar?", collection, top_k=3)
    >>> isinstance(chunks, list)
    True
    """
    if not question or not question.strip():
        raise ValueError("A pergunta nao pode ser vazia.")

    try:
        query_embedding = generate_local_embedding(question, input_type="query")
    except EmbeddingServiceError as exc:
        raise RetrievalError(f"Falha ao gerar embedding da pergunta: {exc}") from exc

    semantic_n = max(top_k * SEMANTIC_SEARCH_MULTIPLIER, top_k)
    keyword_n = max(top_k * KEYWORD_SEARCH_MULTIPLIER, top_k)

    try:
        semantic_results = collection.query(
            query_embeddings=[query_embedding],
            n_results=semantic_n,
            include=["documents", "metadatas", "distances"],
        )
    except Exception as exc:  # noqa: BLE001
        raise RetrievalError(f"Falha na busca semantica do ChromaDB: {exc}") from exc

    all_docs: list[str]
    all_metas: list[dict]
    try:
        lexical_data = collection.get(include=["documents", "metadatas"])
        all_docs = lexical_data.get("documents") or []
        all_metas = lexical_data.get("metadatas") or []
    except Exception:
        all_docs = []
        all_metas = []

    semantic_documents = semantic_results.get("documents", [[]])[0]
    semantic_metadatas = semantic_results.get("metadatas", [[]])[0]
    semantic_distances = semantic_results.get("distances", [[]])[0]

    candidates: dict[str, dict] = {}

    for rank, (doc, metadata, distance) in enumerate(
        zip(semantic_documents, semantic_metadatas, semantic_distances),
        start=1,
    ):
        metadata = metadata or {}
        source = metadata.get("source_file", "desconhecido")
        page = metadata.get("page_number", "?")
        chunk_index = metadata.get("chunk_index", rank)
        chunk_type = metadata.get("chunk_type", "content")
        candidate_id = f"{source}:{page}:{chunk_index}:{chunk_type}"

        semantic_score = 1.0 / (1.0 + max(distance, 0.0)) if distance is not None else 0.0
        candidates[candidate_id] = {
            "text": doc,
            "metadata": metadata,
            "distance": distance,
            "semantic_score": semantic_score,
            "keyword_score": 0.0,
            "combined_score": 0.0,
            "retrieval_path": "semantic",
        }

    lexical_scored: list[tuple[float, str, dict]] = []
    for doc, metadata in zip(all_docs, all_metas):
        metadata = metadata or {}
        chunk_type = metadata.get("chunk_type", "content")
        if chunk_type == "document_summary":
            continue
        score = _keyword_score(question, doc)
        if score <= 0:
            continue
        lexical_scored.append((score, doc, metadata))

    lexical_scored.sort(key=lambda item: item[0], reverse=True)
    lexical_scored = lexical_scored[:keyword_n]

    for keyword_score, doc, metadata in lexical_scored:
        source = metadata.get("source_file", "desconhecido")
        page = metadata.get("page_number", "?")
        chunk_index = metadata.get("chunk_index", 0)
        chunk_type = metadata.get("chunk_type", "content")
        candidate_id = f"{source}:{page}:{chunk_index}:{chunk_type}"

        if candidate_id not in candidates:
            candidates[candidate_id] = {
                "text": doc,
                "metadata": metadata,
                "distance": None,
                "semantic_score": 0.0,
                "keyword_score": keyword_score,
                "combined_score": 0.0,
                "retrieval_path": "keyword",
            }
        else:
            candidates[candidate_id]["keyword_score"] = keyword_score
            candidates[candidate_id]["retrieval_path"] = "hybrid"

    scored_candidates: list[dict] = []
    for candidate in candidates.values():
        semantic_score = candidate["semantic_score"]
        keyword_score = candidate["keyword_score"]
        distance = candidate.get("distance")

        combined_score = (
            (SEMANTIC_SCORE_WEIGHT * semantic_score)
            + (KEYWORD_SCORE_WEIGHT * keyword_score)
        )
        candidate["combined_score"] = combined_score

        distance_ok = distance is not None and distance <= MAX_DISTANCE_THRESHOLD
        keyword_ok = keyword_score >= 0.2

        # Permitimos passar no filtro por semantica OU por keyword forte.
        if distance_ok or keyword_ok:
            scored_candidates.append(candidate)

    scored_candidates.sort(key=lambda item: item["combined_score"], reverse=True)

    selected_content = [
        item
        for item in scored_candidates
        if item.get("metadata", {}).get("chunk_type", "content") != "document_summary"
    ][:top_k]

    if not selected_content:
        return []

    source_counter = Counter(
        item.get("metadata", {}).get("source_file", "desconhecido")
        for item in selected_content
    )
    dominant_source = source_counter.most_common(1)[0][0]
    summary_chunk = _pick_document_summary(collection, dominant_source)

    if summary_chunk:
        selected_content.append(summary_chunk)

    return selected_content


def format_chunks_for_prompt(retrieved_chunks: list[dict]) -> list[str]:
    """Formata chunks recuperados para insercao no prompt do LLM.

    O que faz:
    - Prefixa cada chunk com identificacao da fonte e pagina.
    - Gera lista de strings pronta para o template do prompt.

    Por que existe:
    - Fornecer contexto estruturado melhora rastreabilidade e qualidade da resposta.

    Exemplo:
    >>> chunks = [{"text": "Passo 1", "metadata": {"source_file": "manual.pdf", "page_number": 3}}]
    >>> format_chunks_for_prompt(chunks)[0].startswith("[Fonte:")
    True
    """
    formatted: list[str] = []

    for chunk in retrieved_chunks:
        metadata = chunk.get("metadata", {})
        source = metadata.get("source_file", "desconhecido")
        page = metadata.get("page_number", "?")
        text = chunk.get("text", "")
        formatted.append(f"[Fonte: {source} | Pagina: {page}]\n{text}")

    return formatted
