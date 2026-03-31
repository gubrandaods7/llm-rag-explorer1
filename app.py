"""app.py
Papel na arquitetura:
Este modulo e a interface principal em Streamlit. Ele orquestra ingestao,
retrieval e geracao de respostas, exibindo fontes e controles para o usuario.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import chromadb
import streamlit as st

from config import (
    AVAILABLE_GEMINI_CHAT_MODELS,
    CHROMA_COLLECTION_NAME,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_K,
    GEMINI_CHAT_MODEL,
    GOOGLE_API_KEY_ENV_VAR,
    INDEXING_STRATEGY_VERSION,
    MAX_TEMPERATURE,
    MAX_TOP_K,
    MIN_TEMPERATURE,
    MIN_TOP_K,
    PDF_DIR,
    VECTORSTORE_DIR,
)
from embeddings import get_local_embedding_backend_name
from ingest import IngestionError, ingest_pdf_to_chroma
from llm import LLMConfigurationError, LLMServiceError, load_google_api_key, generate_chat_answer
from retriever import RetrievalError, format_chunks_for_prompt, retrieve_relevant_chunks


@st.cache_resource
def get_chroma_client() -> chromadb.PersistentClient:
    """Retorna cliente persistente do ChromaDB com cache de recurso do Streamlit.

    O que faz:
    - Inicializa um PersistentClient apontando para ./vectorstore.
    - Reutiliza a mesma conexao entre reruns para performance.

    Por que existe:
    - Evita recriar cliente a cada interacao do app, reduzindo latencia.

    Exemplo:
    >>> client = get_chroma_client()
    >>> isinstance(client, chromadb.PersistentClient)
    True
    """
    VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(VECTORSTORE_DIR))


@st.cache_resource
def get_or_create_collection() -> Any:
    """Retorna a colecao vetorial usada pelo RAG, criando se necessario.

    O que faz:
    - Usa o cliente do Chroma cacheado.
    - Cria ou recupera a colecao principal de manuais.

    Por que existe:
    - Centraliza o acesso ao banco vetorial em um unico ponto reutilizavel.

    Exemplo:
    >>> collection = get_or_create_collection()
    >>> hasattr(collection, "query")
    True
    """
    client = get_chroma_client()
    return client.get_or_create_collection(name=CHROMA_COLLECTION_NAME)


def save_uploaded_pdf(uploaded_file) -> Path:
    """Salva PDF enviado pelo usuario em data/pdfs.

    O que faz:
    - Garante que a pasta de PDFs existe.
    - Persiste o conteudo do upload em disco local.

    Por que existe:
    - A ingestao trabalha com caminhos de arquivo, nao apenas bytes em memoria.

    Exemplo:
    >>> # path = save_uploaded_pdf(uploaded_file)
    >>> # print(path.name)
    """
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    target_path = PDF_DIR / uploaded_file.name
    target_path.write_bytes(uploaded_file.getbuffer())
    return target_path


def list_processed_documents(collection) -> list[dict]:
    """Lista documentos processados a partir dos metadados da colecao.

    O que faz:
    - Le metadados no ChromaDB.
    - Agrupa por arquivo e retorna data de ingestao mais recente.

    Por que existe:
    - Permite mostrar na sidebar quais PDFs ja foram indexados.

    Exemplo:
    >>> docs = list_processed_documents(collection)
    >>> isinstance(docs, list)
    True
    """
    try:
        data = collection.get(include=["metadatas"])
    except Exception:
        return []

    metadatas = data.get("metadatas") or []
    grouped: dict[str, str] = {}
    for metadata in metadatas:
        if not metadata:
            continue
        source = metadata.get("source_file")
        ingested_at = metadata.get("ingested_at", "data desconhecida")
        if not source:
            continue
        latest = grouped.get(source)
        if latest is None or ingested_at > latest:
            grouped[source] = ingested_at

    return [
        {"file_name": file_name, "ingested_at": ingested_at}
        for file_name, ingested_at in sorted(grouped.items(), key=lambda item: item[0].lower())
    ]


def clear_vector_database() -> None:
    """Limpa completamente a base vetorial local e recria a colecao.

    O que faz:
    - Remove a colecao atual no ChromaDB.
    - Limpa cache de recurso para forcar nova instancia na proxima execucao.

    Por que existe:
    - Facilita reiniciar experimentos sem residuos de indexacoes anteriores.

    Exemplo:
    >>> clear_vector_database()
    """
    client = get_chroma_client()
    try:
        client.delete_collection(name=CHROMA_COLLECTION_NAME)
    except Exception:
        # Se a colecao nao existir, seguimos sem interromper a UX.
        pass
    st.cache_resource.clear()


def ensure_session_state() -> None:
    """Inicializa estrutura de estado do chat na sessao Streamlit.

    O que faz:
    - Cria st.session_state.messages se ainda nao existir.

    Por que existe:
    - Mantem historico do chat entre reruns automaticos do Streamlit.

    Exemplo:
    >>> ensure_session_state()
    >>> assert "messages" in st.session_state
    """
    if "messages" not in st.session_state:
        st.session_state.messages = []


def render_sources_expander(sources: list[dict], show_debug: bool = False) -> None:
    """Renderiza fontes usadas na resposta em um expander.

    O que faz:
    - Exibe cada chunk recuperado com arquivo e pagina de origem.

    Por que existe:
    - Transparencia: usuario pode auditar de onde veio a resposta.

    Exemplo:
    >>> # render_sources_expander(sources)
    """
    with st.expander("📄 Fontes consultadas"):
        for index, source in enumerate(sources, start=1):
            metadata = source.get("metadata", {})
            file_name = metadata.get("source_file", "desconhecido")
            page_number = metadata.get("page_number", "?")
            text = source.get("text", "")
            st.markdown(f"**{index}. Arquivo:** {file_name} | **Pagina:** {page_number}")

            if show_debug:
                st.caption(
                    " | ".join(
                        [
                            f"dist={source.get('distance')}",
                            f"semantic={round(source.get('semantic_score', 0.0), 4)}",
                            f"keyword={round(source.get('keyword_score', 0.0), 4)}",
                            f"combined={round(source.get('combined_score', 0.0), 4)}",
                            f"path={source.get('retrieval_path', 'n/a')}",
                        ]
                    )
                )

            st.caption(text)
            st.divider()


def inspect_indexing_strategy(collection) -> tuple[bool, list[str]]:
    """Verifica se a base atual foi indexada com a estrategia esperada.

    O que faz:
    - Le metadados da colecao.
    - Retorna se ha mismatch e quais versoes foram encontradas.

    Por que existe:
    - Misturar estrategias antigas e novas prejudica retrieval.

    Exemplo:
    >>> has_mismatch, versions = inspect_indexing_strategy(collection)
    >>> isinstance(has_mismatch, bool)
    True
    """
    try:
        data = collection.get(include=["metadatas"])
    except Exception:
        return False, []

    metadatas = data.get("metadatas") or []
    versions = sorted(
        {
            (metadata or {}).get("indexing_strategy_version", "desconhecida")
            for metadata in metadatas
        }
    )
    if not versions:
        return False, []

    has_mismatch = (len(versions) > 1) or (versions[0] != INDEXING_STRATEGY_VERSION)
    return has_mismatch, versions


def main() -> None:
    """Executa a aplicacao Streamlit com fluxo completo de RAG.

    O que faz:
    - Renderiza sidebar de ingestao/configuracao.
    - Renderiza chat principal e integra retrieval + LLM.

    Por que existe:
    - E o ponto de entrada da aplicacao para o usuario final.

    Exemplo:
    >>> # streamlit run app.py
    """
    st.set_page_config(page_title="RAG de Manuais", page_icon="📚", layout="wide")
    st.title("📚 RAG de Manuais Tecnicos")
    st.caption("Faça upload de PDFs, processe os documentos e pergunte com base no contexto indexado.")

    ensure_session_state()

    api_key = load_google_api_key()
    if not api_key:
        st.error(
            "API key do Gemini nao configurada. Crie um arquivo .env na raiz do projeto com:\n\n"
            f"{GOOGLE_API_KEY_ENV_VAR}=SUA_CHAVE_AQUI"
        )
        st.info("Consulte o README para o passo a passo de criacao da chave no Google AI Studio.")

    with st.sidebar:
        st.header("Configuração")

        uploaded_files = st.file_uploader(
            "Upload de PDFs",
            type=["pdf"],
            accept_multiple_files=True,
            help="Envie um ou mais manuais em PDF.",
        )

        selected_chat_model = st.selectbox(
            "Modelo LLM (resposta)",
            options=AVAILABLE_GEMINI_CHAT_MODELS,
            index=(
                AVAILABLE_GEMINI_CHAT_MODELS.index(GEMINI_CHAT_MODEL)
                if GEMINI_CHAT_MODEL in AVAILABLE_GEMINI_CHAT_MODELS
                else 0
            ),
            help="Escolha qual modelo Gemini sera usado somente na resposta final do chat.",
        )

        top_k = st.slider(
            "Numero de trechos para buscar",
            min_value=MIN_TOP_K,
            max_value=MAX_TOP_K,
            value=DEFAULT_TOP_K,
            step=1,
        )

        temperature = st.slider(
            "Temperature",
            min_value=MIN_TEMPERATURE,
            max_value=MAX_TEMPERATURE,
            value=DEFAULT_TEMPERATURE,
            step=0.1,
        )

        show_retrieval_debug = st.checkbox(
            "Mostrar debug de retrieval",
            value=False,
            help="Exibe distancias e scores hibridos para diagnosticar qualidade da busca.",
        )

        st.caption(
            f"Embedding local ativo: {get_local_embedding_backend_name()} | "
            f"Estrategia: {INDEXING_STRATEGY_VERSION}"
        )

        process_clicked = st.button("Processar documentos", use_container_width=True)
        clear_clicked = st.button("Limpar base vetorial", use_container_width=True, type="secondary")

        st.subheader("Documentos processados")
        collection = get_or_create_collection()
        processed_docs = list_processed_documents(collection)
        has_mismatch, versions = inspect_indexing_strategy(collection)
        if has_mismatch:
            st.warning(
                "Base indexada com estrategia diferente da atual. "
                "Recomendado limpar a base vetorial e reprocessar os documentos.\n\n"
                f"Versoes encontradas: {', '.join(versions)}\n"
                f"Versao atual: {INDEXING_STRATEGY_VERSION}"
            )

        if not processed_docs:
            st.caption("Nenhum documento processado ainda.")
        else:
            for doc in processed_docs:
                st.markdown(f"- **{doc['file_name']}** ({doc['ingested_at']})")

    if clear_clicked:
        clear_vector_database()
        st.success("Base vetorial limpa com sucesso.")
        st.rerun()

    if process_clicked:
        if not api_key:
            st.warning("Configure sua API key antes de processar documentos.")
        elif not uploaded_files:
            st.warning("Envie ao menos um PDF para iniciar a ingestao.")
        else:
            collection = get_or_create_collection()
            for uploaded_file in uploaded_files:
                progress = st.progress(0, text=f"Iniciando ingestao de {uploaded_file.name}...")

                def progress_callback(stage: str, percent: int) -> None:
                    progress.progress(percent, text=f"{uploaded_file.name}: {stage}")

                try:
                    saved_path = save_uploaded_pdf(uploaded_file)
                    summary = ingest_pdf_to_chroma(
                        pdf_path=saved_path,
                        collection=collection,
                        progress_callback=progress_callback,
                    )
                    progress.progress(100, text=f"{uploaded_file.name}: ingestao concluida.")
                    st.success(
                        f"{summary['file_name']} processado: "
                        f"{summary['pages_with_text']} paginas com texto, "
                        f"{summary['chunks_ingested']} chunks indexados."
                    )
                except IngestionError as exc:
                    st.error(str(exc))
                except LLMConfigurationError as exc:
                    st.error(str(exc))
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Erro inesperado na ingestao de {uploaded_file.name}: {exc}")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and message.get("sources"):
                render_sources_expander(message["sources"], show_debug=show_retrieval_debug)

    user_question = st.chat_input("Digite sua pergunta sobre os manuais processados...")

    if user_question:
        st.session_state.messages.append({"role": "user", "content": user_question})
        with st.chat_message("user"):
            st.markdown(user_question)

        with st.chat_message("assistant"):
            try:
                collection = get_or_create_collection()
                with st.spinner("Buscando contexto relevante..."):
                    retrieved = retrieve_relevant_chunks(
                        question=user_question,
                        collection=collection,
                        top_k=top_k,
                    )

                if not retrieved:
                    answer = (
                        "Nao encontrei trechos relevantes na base para responder com seguranca. "
                        "Tente reformular a pergunta ou processar mais documentos."
                    )
                    st.markdown(answer)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": answer, "sources": []}
                    )
                else:
                    prompt_chunks = format_chunks_for_prompt(retrieved)
                    with st.spinner("Gerando resposta com Gemini..."):
                        answer = generate_chat_answer(
                            question=user_question,
                            context_chunks=prompt_chunks,
                            temperature=temperature,
                            chat_model=selected_chat_model,
                        )

                    st.markdown(answer)
                    render_sources_expander(retrieved, show_debug=show_retrieval_debug)
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": answer,
                            "sources": retrieved,
                        }
                    )
            except (RetrievalError, LLMServiceError, LLMConfigurationError) as exc:
                st.error(f"Erro durante consulta: {exc}")
            except Exception as exc:  # noqa: BLE001
                st.error(f"Erro inesperado durante consulta: {exc}")


if __name__ == "__main__":
    main()
