"""config.py
Papel na arquitetura:
Este modulo centraliza configuracoes do projeto RAG para evitar valores hardcoded
espalhados no codigo. Assim, alterar modelos, tamanhos de chunk e caminhos fica
simples e seguro.
"""

from pathlib import Path

# Caminhos base do projeto.
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
PDF_DIR = DATA_DIR / "pdfs"
VECTORSTORE_DIR = BASE_DIR / "vectorstore"

# Nome da colecao vetorial no ChromaDB.
CHROMA_COLLECTION_NAME = "manuais_tecnicos"

# Versao da estrategia de indexacao. Sempre que a estrategia mudar de forma
# relevante (chunking/embedding/retrieval), aumente essa versao e reindexe.
INDEXING_STRATEGY_VERSION = "v2-hybrid-multilingual"

# Modelos do Gemini.
GEMINI_CHAT_MODEL = "gemini-2.5-flash-lite"
AVAILABLE_GEMINI_CHAT_MODELS = [
	"gemini-2.5-flash-lite",
	"gemini-2.5-flash",
	"gemini-2.0-flash-lite",
	"gemini-2.5-pro",
]

# Parametros de chunking.
CHUNK_SIZE_TOKENS = 900
CHUNK_OVERLAP_TOKENS = 150

# Parametros para resumo global do documento (contexto do arquivo inteiro).
DOCUMENT_SUMMARY_MAX_TOKENS = 1200

# Parametros de retrieval e resposta.
DEFAULT_TOP_K = 5
MIN_TOP_K = 1
MAX_TOP_K = 10
SEMANTIC_SEARCH_MULTIPLIER = 4
KEYWORD_SEARCH_MULTIPLIER = 6
DEFAULT_TEMPERATURE = 0.3
MIN_TEMPERATURE = 0.0
MAX_TEMPERATURE = 1.0

# Limiar de distancia para evitar respostas com contexto irrelevante.
# No espaco de distancia de cosseno usado por padrao no Chroma, valores menores
# indicam maior similaridade.
MAX_DISTANCE_THRESHOLD = 1.8

# Pesos do ranking hibrido (semantica + keyword).
SEMANTIC_SCORE_WEIGHT = 0.65
KEYWORD_SCORE_WEIGHT = 0.35

# Configuracao de embedding local multilinguagem (melhor para PT-BR tecnico).
LOCAL_EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-small"

# Variavel de ambiente esperada no .env.
GOOGLE_API_KEY_ENV_VAR = "GOOGLE_API_KEY"
