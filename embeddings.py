"""embeddings.py
Papel na arquitetura:
Este modulo concentra a geracao de embeddings locais (sem LLM remoto). Assim,
o Gemini fica responsavel apenas pela resposta final ao usuario.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from chromadb.utils import embedding_functions

from config import LOCAL_EMBEDDING_MODEL_NAME


class EmbeddingServiceError(Exception):
    """Erro para falhas na geracao de embeddings locais."""


@lru_cache(maxsize=1)
def get_local_embedding_function():
    """Retorna funcao de embedding local padrao do ChromaDB.

    O que faz:
    - Inicializa uma unica instancia da funcao de embedding local.
    - Reaproveita a instancia para reduzir custo de inicializacao.

    Por que existe:
    - Evita chamar API externa para embeddings.
    - Mantem consistencia entre ingestao e consulta vetorial.

    Exemplo:
    >>> emb_fn = get_local_embedding_function()
    >>> vectors = emb_fn(["texto de exemplo"])
    """
    try:
        # Preferimos um modelo multilingue para melhorar perguntas em portugues
        # tecnico. Se o ambiente nao tiver dependencias necessarias, usamos
        # fallback automatico para o embedding padrao do Chroma.
        return embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=LOCAL_EMBEDDING_MODEL_NAME
        )
    except Exception:
        return embedding_functions.DefaultEmbeddingFunction()


@lru_cache(maxsize=1)
def get_local_embedding_backend_name() -> str:
    """Informa qual backend de embedding local esta em uso.

    O que faz:
    - Detecta se estamos usando SentenceTransformer multilingue ou fallback.

    Por que existe:
    - Transparencia para debug e para orientar reindexacao quando necessario.

    Exemplo:
    >>> backend = get_local_embedding_backend_name()
    >>> isinstance(backend, str)
    True
    """
    try:
        emb_fn = get_local_embedding_function()
        class_name = emb_fn.__class__.__name__
        if class_name == "SentenceTransformerEmbeddingFunction":
            return f"sentence-transformers:{LOCAL_EMBEDDING_MODEL_NAME}"
        return f"chromadb-default:{class_name}"
    except Exception:
        return "unknown"


def _prepare_embedding_input(text: str, input_type: Literal["query", "passage"]) -> str:
    """Prepara texto para modelos que exigem prefixo instrucional.

    O que faz:
    - Prefixa query/passage para familia E5 quando necessario.

    Por que existe:
    - Modelos E5 performam melhor com instrucoes explicitas no texto.

    Exemplo:
    >>> _prepare_embedding_input("MRR", "query")
    'query: MRR'
    """
    backend = get_local_embedding_backend_name().lower()
    if "e5" in backend:
        return f"{input_type}: {text}"
    return text


def generate_local_embedding(text: str, input_type: Literal["query", "passage"] = "passage") -> list[float]:
    """Gera embedding local para um texto unico.

    O que faz:
    - Chama a funcao de embedding local com uma lista de um item.
    - Retorna o vetor correspondente ao texto informado.

    Por que existe:
    - Facilita uso no pipeline existente, que processa chunk por chunk.

    Exemplo:
    >>> vec = generate_local_embedding("manual de calibracao", input_type="passage")
    >>> len(vec) > 0
    True
    """
    if not text or not text.strip():
        raise ValueError("Nao e possivel gerar embedding de texto vazio.")

    try:
        emb_fn = get_local_embedding_function()
        prepared_text = _prepare_embedding_input(text, input_type)
        vectors = emb_fn([prepared_text])
        if vectors is None or len(vectors) == 0:
            raise EmbeddingServiceError("Embedding local retornou vetor vazio.")

        first_vector = vectors[0]
        if first_vector is None:
            raise EmbeddingServiceError("Embedding local retornou vetor vazio.")

        # Alguns providers retornam ndarray; convertemos para lista Python para
        # manter compatibilidade com APIs que esperam serializacao JSON.
        if hasattr(first_vector, "tolist"):
            first_vector = first_vector.tolist()

        if len(first_vector) == 0:
            raise EmbeddingServiceError("Embedding local retornou vetor vazio.")

        return first_vector
    except Exception as exc:  # noqa: BLE001
        raise EmbeddingServiceError(f"Falha ao gerar embedding local: {exc}") from exc
