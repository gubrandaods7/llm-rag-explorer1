"""llm.py
Papel na arquitetura:
Este modulo isola a comunicacao com o Google Gemini para geracao de resposta
final do usuario. Embeddings ficam em modulo local separado.
"""

from __future__ import annotations

import os

import google.generativeai as genai
from dotenv import load_dotenv

from config import (
    GEMINI_CHAT_MODEL,
    GOOGLE_API_KEY_ENV_VAR,
)


class LLMConfigurationError(Exception):
    """Erro para problemas de configuracao do cliente LLM."""


class LLMServiceError(Exception):
    """Erro para falhas de chamada na API do Gemini."""


def load_google_api_key() -> str | None:
    """Carrega a API key do Gemini a partir do arquivo .env e do ambiente.

    O que faz:
    - Executa o carregamento de variaveis do arquivo .env.
    - Retorna o valor de GOOGLE_API_KEY (ou None se nao existir).

    Por que existe:
    - Centraliza o ponto de leitura da chave e evita duplicacao no projeto.
    - Facilita mostrar mensagens claras quando a chave nao esta configurada.

    Exemplo:
    >>> key = load_google_api_key()
    >>> if key:
    ...     print("API key encontrada")
    """
    load_dotenv()
    return os.getenv(GOOGLE_API_KEY_ENV_VAR)


def configure_gemini_client() -> None:
    """Configura o SDK do Gemini usando a API key do ambiente.

    O que faz:
    - Le a API key via load_google_api_key().
    - Inicializa o cliente global do SDK com genai.configure().

    Por que existe:
    - Garante que qualquer chamada ao Gemini aconteca com credenciais validas.
    - Falha cedo com mensagem clara, evitando erros mais dificeis de depurar.

    Exemplo:
    >>> configure_gemini_client()
    >>> answer = generate_chat_answer("Teste?", ["Contexto de exemplo."])
    """
    api_key = load_google_api_key()
    if not api_key:
        raise LLMConfigurationError(
            "API key nao encontrada. Configure GOOGLE_API_KEY no arquivo .env."
        )

    genai.configure(api_key=api_key)


def build_rag_prompt(question: str, context_chunks: list[str]) -> str:
    """Monta o prompt RAG restritivo que instrui o modelo a usar somente contexto.

    O que faz:
    - Junta os chunks recuperados em um bloco de contexto.
    - Formata o template de prompt orientado a resposta fiel ao contexto.

    Por que existe:
    - Prompt padronizado reduz alucinacoes e garante comportamento consistente.

    Exemplo:
    >>> prompt = build_rag_prompt("Qual a voltagem?", ["Trecho 1", "Trecho 2"])
    >>> "Contexto:" in prompt
    True
    """
    joined_chunks = "\n\n".join(context_chunks)
    return (
        "Voce e um assistente que responde perguntas com base em manuais tecnicos.\n"
        "Use APENAS as informacoes do contexto abaixo para responder.\n"
        "Se a resposta nao estiver no contexto, diga que nao encontrou a informacao.\n\n"
        f"Contexto:\n{joined_chunks}\n\n"
        f"Pergunta: {question}"
    )


def generate_chat_answer(
    question: str,
    context_chunks: list[str],
    temperature: float = 0.3,
    chat_model: str | None = None,
) -> str:
    """Gera resposta final do chat usando Gemini e contexto recuperado.

    O que faz:
    - Constroi prompt RAG a partir da pergunta e dos chunks.
    - Chama o modelo de chat escolhido para produzir resposta em linguagem natural.

    Por que existe:
    - Encapsula a chamada de texto generativo em uma interface simples para a UI.

    Exemplo:
    >>> answer = generate_chat_answer("Como calibrar?", ["Passo 1: ..."], chat_model="gemini-2.5-flash-lite")
    >>> isinstance(answer, str)
    True
    """
    if not question or not question.strip():
        raise ValueError("A pergunta nao pode estar vazia.")

    if not context_chunks:
        return "Nao encontrei trechos relevantes para responder com seguranca."

    configure_gemini_client()

    prompt = build_rag_prompt(question=question, context_chunks=context_chunks)

    model_to_use = (chat_model or GEMINI_CHAT_MODEL).strip()

    try:
        model = genai.GenerativeModel(model_to_use)
        response = model.generate_content(
            prompt,
            generation_config={"temperature": temperature},
        )
        text = getattr(response, "text", None)
        if not text:
            raise LLMServiceError("A resposta do Gemini veio vazia.")
        return text.strip()
    except Exception as exc:  # noqa: BLE001
        raise LLMServiceError(
            f"Falha ao gerar resposta no Gemini com o modelo '{model_to_use}': {exc}"
        ) from exc
