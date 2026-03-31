# Arquitetura do Projeto

## Diagrama ASCII da arquitetura completa

+-----------------------+           +------------------------+
|      Streamlit UI     |           |      Arquivos PDF      |
| app.py                |           | data/pdfs/*.pdf        |
| - Upload              |           +-----------+------------+
| - Processar           |                       |
| - Chat                |                       v
+-----------+-----------+             +---------+------------------+
            |                         | ingest.py                  |
            | chama ingestao          | - PyMuPDF extrai texto     |
            |                         | - chunker.py segmenta       |
            |                         | - embeddings.py vetoriza    |
            |                         | - cria resumo global        |
            |                         +---------+------------------+
            |                                   |
            |                                   v
            |                         +---------+----------+
            |                         | ChromaDB local      |
            |                         | ./vectorstore       |
            |                         | colecao manuais     |
            |                         +----+-----------+----+
            |                              ^           |
            | consulta top-k               |           | guarda
            v                              |           | chunks+metadados
+-----------+-------------------+          |           |
| retriever.py                 |-----------+           |
| - embedding local pergunta   |                      |
| - busca semantica top-N      |                      |
| - busca keyword top-N        |                      |
| - reranking hibrido          |                      |
+-----------+-------------------+                      |
            |                                          |
            | contexto relevante + resumo global       |
            v                                          |
+-----------+-----------+                               |
|      llm.py           |<------------------------------+
| - prompt RAG          |
| - Gemini chat         |
+-----------+-----------+
            |
            v
+-----------+-----------+
| Resposta + Fontes     |
| + scores de debug     |
+-----------------------+

## Explicacao de cada componente
- app.py: interface principal, ingestao, chat e diagnostico de retrieval.
- ingest.py: pipeline de ingestao e indexacao dos PDFs.
- chunker.py: chunking por passagens com overlap para preservar contexto.
- embeddings.py: embeddings locais (multilingue com fallback).
- retriever.py: retrieval hibrido (semantico + keyword) com reranking.
- llm.py: Gemini somente na etapa de resposta final.
- vectorstore/: persistencia local da base vetorial.
- data/pdfs/: armazenamento dos PDFs enviados pelo usuario.

## Fluxo de dados: ingestao
1. Usuario envia um ou mais PDFs no Streamlit.
2. app.py salva os arquivos em data/pdfs/.
3. ingest.py abre cada PDF com PyMuPDF e extrai texto por pagina.
4. chunker.py divide por passagens com chunks maiores e overlap.
5. embeddings.py gera embedding local para cada chunk.
6. ingest.py cria tambem um chunk de resumo global do documento.
7. ingest.py salva embeddings, chunks e metadados no ChromaDB.

## Fluxo de dados: consulta
1. Usuario envia pergunta no chat.
2. retriever.py gera embedding local da pergunta.
3. retriever.py faz busca semantica e busca keyword em paralelo logico.
4. retriever.py combina scores e aplica reranking hibrido.
5. retriever.py adiciona resumo global do documento dominante.
6. llm.py monta prompt RAG com os chunks recuperados.
7. llm.py chama Gemini chat para gerar resposta.
8. app.py exibe resposta, fontes e (opcionalmente) debug de scores.

## Onde cada dado e armazenado
- PDFs originais: data/pdfs/
- Vetores e indice: vectorstore/
- Metadados no ChromaDB:
  - source_file
  - page_number
  - chunk_index
  - chunk_type (content/document_summary)
  - original_text
  - ingested_at
  - indexing_strategy_version
  - embedding_backend
- Historico de chat: st.session_state (memoria da sessao do app)
