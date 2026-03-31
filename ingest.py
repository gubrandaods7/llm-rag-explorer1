"""ingest.py
Papel na arquitetura:
Este modulo implementa o pipeline de ingestao RAG: PDF -> texto por pagina ->
chunks com overlap -> embeddings -> armazenamento vetorial no ChromaDB.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Callable
import re

import fitz

from chunker import approximate_token_count, chunk_pages, chunk_text
from config import (
    CHUNK_OVERLAP_TOKENS,
    CHUNK_SIZE_TOKENS,
    DOCUMENT_SUMMARY_MAX_TOKENS,
    INDEXING_STRATEGY_VERSION,
)
from embeddings import (
    EmbeddingServiceError,
    generate_local_embedding,
    get_local_embedding_backend_name,
)


class IngestionError(Exception):
    """Erro geral de ingestao de documentos."""


def extract_pdf_pages(pdf_path: Path) -> list[tuple[int, str]]:
    """Extrai texto de um PDF preservando numero da pagina.

    O que faz:
    - Abre o arquivo PDF com PyMuPDF.
    - Extrai texto pagina por pagina e retorna lista de pares (pagina, texto).

    Por que existe:
    - O metadado de pagina permite rastrear a origem de cada resposta no chat.

    Exemplo:
    >>> pages = extract_pdf_pages(Path("data/pdfs/manual.pdf"))
    >>> pages[0][0]  # numero da primeira pagina
    1
    """
    if not pdf_path.exists():
        raise IngestionError(f"Arquivo nao encontrado: {pdf_path}")

    pages: list[tuple[int, str]] = []

    try:
        with fitz.open(pdf_path) as doc:
            for page_idx, page in enumerate(doc, start=1):
                text = page.get_text("text") or ""
                # Mantemos quebras de linha para preservar estrutura de paragrafo.
                normalized = text.replace("\r\n", "\n")
                normalized = re.sub(r"[ \t]+", " ", normalized)
                normalized = re.sub(r"\n{3,}", "\n\n", normalized)
                cleaned_text = normalized.strip()
                pages.append((page_idx, cleaned_text))
    except Exception as exc:  # noqa: BLE001
        raise IngestionError(f"Nao foi possivel ler o PDF '{pdf_path.name}': {exc}") from exc

    return pages


def ingest_pdf_to_chroma(
    pdf_path: Path,
    collection,
    progress_callback: Callable[[str, int], None] | None = None,
) -> dict:
    """Processa um PDF e armazena seus chunks vetorizados no ChromaDB.

    O que faz:
    - Extrai paginas do PDF.
    - Chunka o texto com overlap.
    - Gera embedding local para cada chunk.
    - Salva documentos, embeddings e metadados no ChromaDB.

    Por que existe:
    - Encapsula todo fluxo de ingestao em uma funcao reutilizavel pela UI.

    Exemplo:
    >>> summary = ingest_pdf_to_chroma(Path("data/pdfs/manual.pdf"), collection)
    >>> summary["chunks_ingested"] > 0
    True
    """

    def notify(stage: str, percent: int) -> None:
        if progress_callback:
            progress_callback(stage, percent)

    notify("Lendo PDF...", 5)
    pages = extract_pdf_pages(pdf_path)

    non_empty_pages = [(page_number, text) for page_number, text in pages if text.strip()]
    if not non_empty_pages:
        raise IngestionError(
            f"O arquivo '{pdf_path.name}' nao possui texto extraivel. "
            "Tente um PDF com texto selecionavel (nao apenas imagem)."
        )

    notify("Gerando chunks...", 20)
    chunked = chunk_pages(
        pages=non_empty_pages,
        chunk_size_tokens=CHUNK_SIZE_TOKENS,
        chunk_overlap_tokens=CHUNK_OVERLAP_TOKENS,
    )

    if not chunked:
        raise IngestionError(
            f"Nao foi possivel gerar chunks do arquivo '{pdf_path.name}'."
        )

    notify("Criando contexto global do documento...", 27)
    full_document_text = "\n\n".join(page_text for _, page_text in non_empty_pages)
    summary_chunks = chunk_text(
        text=full_document_text,
        chunk_size_tokens=DOCUMENT_SUMMARY_MAX_TOKENS,
        chunk_overlap_tokens=0,
    )
    document_summary_text = summary_chunks[0] if summary_chunks else full_document_text

    if approximate_token_count(document_summary_text) == 0:
        document_summary_text = "Resumo indisponivel para este documento."

    # Remove dados antigos do mesmo arquivo para evitar duplicidade ao reprocessar.
    notify("Removendo versoes antigas no banco vetorial...", 30)
    collection.delete(where={"source_file": pdf_path.name})

    ids: list[str] = []
    documents: list[str] = []
    embeddings: list[list[float]] = []
    metadatas: list[dict] = []
    embedding_backend = get_local_embedding_backend_name()

    total_chunks = len(chunked)
    notify("Gerando embeddings locais...", 35)

    for index, item in enumerate(chunked):
        chunk_content = item["chunk_text"]
        page_number = item["page_number"]
        chunk_index = item["chunk_index"]

        try:
            vector = generate_local_embedding(chunk_content, input_type="passage")
        except EmbeddingServiceError as exc:
            raise IngestionError(
                f"Erro ao gerar embedding do chunk {chunk_index} "
                f"(pagina {page_number}) em '{pdf_path.name}': {exc}"
            ) from exc

        ids.append(f"{pdf_path.name}:{page_number}:{chunk_index}")
        documents.append(chunk_content)
        embeddings.append(vector)
        metadatas.append(
            {
                "source_file": pdf_path.name,
                "page_number": page_number,
                "chunk_index": chunk_index,
                "original_text": chunk_content,
                "ingested_at": datetime.now().isoformat(timespec="seconds"),
                "chunk_type": "content",
                "indexing_strategy_version": INDEXING_STRATEGY_VERSION,
                "embedding_backend": embedding_backend,
            }
        )

        progress = 35 + int(((index + 1) / total_chunks) * 55)
        notify(f"Gerando embeddings ({index + 1}/{total_chunks})...", progress)

    notify("Gerando embedding do contexto global...", 90)
    try:
        summary_embedding = generate_local_embedding(document_summary_text, input_type="passage")
    except EmbeddingServiceError as exc:
        raise IngestionError(
            f"Erro ao gerar embedding do resumo global em '{pdf_path.name}': {exc}"
        ) from exc

    ids.append(f"{pdf_path.name}:summary:0")
    documents.append(document_summary_text)
    embeddings.append(summary_embedding)
    metadatas.append(
        {
            "source_file": pdf_path.name,
            "page_number": 0,
            "chunk_index": 0,
            "original_text": document_summary_text,
            "ingested_at": datetime.now().isoformat(timespec="seconds"),
            "chunk_type": "document_summary",
            "indexing_strategy_version": INDEXING_STRATEGY_VERSION,
            "embedding_backend": embedding_backend,
        }
    )

    notify("Salvando no ChromaDB...", 94)
    collection.add(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    notify("Ingestao concluida.", 100)
    return {
        "file_name": pdf_path.name,
        "pages_with_text": len(non_empty_pages),
        "chunks_ingested": total_chunks + 1,
        "ingested_at": datetime.now().isoformat(timespec="seconds"),
        "embedding_backend": embedding_backend,
    }
