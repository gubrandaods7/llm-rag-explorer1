"""chunker.py
Papel na arquitetura:
Este modulo transforma textos longos em pequenos trechos (chunks) com overlap.
Ele prepara o conteudo para embeddings e busca semantica de forma eficiente.
"""

from __future__ import annotations

import re
from typing import Iterable

from config import CHUNK_OVERLAP_TOKENS, CHUNK_SIZE_TOKENS


def tokenize_text(text: str) -> list[str]:
    """Tokeniza texto de forma simples usando palavras e sinais basicos.

    O que faz:
    - Quebra o texto em uma lista de "tokens" aproximados via regex.

    Por que existe:
    - Precisamos de uma unidade para estimar tamanho de chunk sem depender de
      tokenizador externo.

    Exemplo:
    >>> tokenize_text("Manual v2.0: ligar e calibrar.")[:4]
    ['Manual', 'v2', '0', 'ligar']
    """
    return re.findall(r"\w+", text, flags=re.UNICODE)


def split_text_into_passages(text: str) -> list[str]:
    """Divide texto em passagens por paragrafos e sentencas preservando forma.

    O que faz:
    - Separa o texto primeiro por paragrafos.
    - Para paragrafos longos, quebra por fronteiras de sentenca.

    Por que existe:
    - Chunks por passagens preservam semantica e contexto melhor que janelas
      cegas de palavras.

    Exemplo:
    >>> split_text_into_passages("Primeira frase. Segunda frase.")
    ['Primeira frase.', 'Segunda frase.']
    """
    if not text or not text.strip():
        return []

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    passages: list[str] = []

    for paragraph in paragraphs:
        # Quebra em sentencas mantendo pontuacao de fechamento.
        sentences = re.split(r"(?<=[\.!\?])\s+", paragraph)
        for sentence in sentences:
            cleaned = sentence.strip()
            if cleaned:
                passages.append(cleaned)

    return passages


def approximate_token_count(text: str) -> int:
    """Retorna contagem aproximada de tokens de um texto.

    O que faz:
    - Calcula quantos tokens simples existem no texto.

    Por que existe:
    - Ajuda a validar se os chunks estao no tamanho esperado para retrieval.

    Exemplo:
    >>> approximate_token_count("A B C")
    3
    """
    return len(tokenize_text(text))


def chunk_text(
    text: str,
    chunk_size_tokens: int = CHUNK_SIZE_TOKENS,
    chunk_overlap_tokens: int = CHUNK_OVERLAP_TOKENS,
) -> list[str]:
    """Divide texto em chunks por passagens, respeitando limite de tokens.

    O que faz:
    - Percorre passagens (paragrafos/sentencas) e acumula ate chunk_size_tokens.
    - Reaproveita contexto usando overlap por tokens aproximados.

    Por que existe:
    - LLMs e embeddings funcionam melhor com trechos menores.
    - Overlap reduz perda de contexto em fronteiras de chunk.

    Exemplo:
    >>> chunk_text("Primeira frase. Segunda frase. Terceira frase.", chunk_size_tokens=6, chunk_overlap_tokens=2)
    ['Primeira frase. Segunda frase.', 'Segunda frase. Terceira frase.']
    """
    if not text or not text.strip():
        return []

    if chunk_overlap_tokens >= chunk_size_tokens:
        raise ValueError("chunk_overlap_tokens deve ser menor que chunk_size_tokens.")

    passages = split_text_into_passages(text)
    if not passages:
        return []

    chunks: list[str] = []
    current_passages: list[str] = []
    current_tokens = 0

    for passage in passages:
        passage_tokens = approximate_token_count(passage)
        if passage_tokens == 0:
            continue

        # Caso extremo: uma unica passagem maior que o chunk inteiro.
        if passage_tokens > chunk_size_tokens and current_passages:
            chunks.append(" ".join(current_passages).strip())
            current_passages = [passage]
            current_tokens = passage_tokens
            continue

        if current_tokens + passage_tokens <= chunk_size_tokens:
            current_passages.append(passage)
            current_tokens += passage_tokens
            continue

        if current_passages:
            chunks.append(" ".join(current_passages).strip())

        # Por que fazemos overlap? Porque se um conceito esta na fronteira entre
        # dois chunks, sem overlap ele seria cortado ao meio e poderia nao ser
        # recuperado na busca semantica.
        overlap_passages: list[str] = []
        overlap_token_count = 0
        for old_passage in reversed(current_passages):
            old_tokens = approximate_token_count(old_passage)
            if overlap_token_count + old_tokens > chunk_overlap_tokens:
                break
            overlap_passages.insert(0, old_passage)
            overlap_token_count += old_tokens

        current_passages = overlap_passages + [passage]
        current_tokens = overlap_token_count + passage_tokens

    if current_passages:
        chunks.append(" ".join(current_passages).strip())

    return chunks


def chunk_pages(
    pages: Iterable[tuple[int, str]],
    chunk_size_tokens: int = CHUNK_SIZE_TOKENS,
    chunk_overlap_tokens: int = CHUNK_OVERLAP_TOKENS,
) -> list[dict]:
    """Gera chunks por pagina preservando metadado de origem.

    O que faz:
    - Recebe pares (numero_da_pagina, texto_da_pagina).
    - Aplica chunk_text em cada pagina e devolve lista estruturada.

    Por que existe:
    - Preservar pagina permite mostrar fontes consultadas no chat.

    Exemplo:
    >>> pages = [(1, "A B C D E F"), (2, "G H I")]
    >>> result = chunk_pages(pages, chunk_size_tokens=4, chunk_overlap_tokens=1)
    >>> result[0]["page_number"]
    1
    """
    chunked: list[dict] = []

    for page_number, page_text in pages:
        page_chunks = chunk_text(
            text=page_text,
            chunk_size_tokens=chunk_size_tokens,
            chunk_overlap_tokens=chunk_overlap_tokens,
        )
        for chunk_index, chunk in enumerate(page_chunks):
            chunked.append(
                {
                    "page_number": page_number,
                    "chunk_index": chunk_index,
                    "chunk_text": chunk,
                }
            )

    return chunked
