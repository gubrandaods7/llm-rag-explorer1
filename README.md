# RAG de Manuais em PDF (Streamlit + Gemini + ChromaDB)

Projeto educacional de Retrieval-Augmented Generation (RAG) para responder perguntas sobre manuais tecnicos em PDF.

Arquitetura atual:
- Embeddings: locais (sem API externa)
- Retrieval: hibrido (semantico + keyword) com reranking
- LLM: Gemini apenas para resposta final ao usuario

## 1) O que e RAG (explicacao para iniciantes)
RAG significa Retrieval-Augmented Generation.

Em vez de pedir para o LLM responder "de memoria", fazemos duas etapas:
1. Retrieval: buscamos os trechos mais relevantes em uma base de conhecimento.
2. Generation: pedimos ao LLM para responder usando esses trechos como contexto.

Vantagens:
- Menos alucinacao
- Respostas com fonte rastreavel
- Melhor controle sobre o conhecimento usado

## 2) Como funciona este projeto

### Pipeline de ingestao (PDF -> base vetorial)

+---------+      +-----------+      +-----------------+      +----------------------+
|  PDF(s) | ---> | Extracao  | ---> | Chunking por    | ---> | Embeddings locais    |
| upload  |      | PyMuPDF   |      | passagens       |      | (multilingue/fallback)|
+---------+      +-----------+      +-----------------+      +----------+-----------+
                                                                       |
                                                                       v
                                                                +------+------+
                                                                | ChromaDB    |
                                                                | vectorstore |
                                                                +-------------+

### Pipeline de consulta (pergunta -> resposta)

+-----------+      +-------------------+      +-------------------------------+
| Pergunta  | ---> | Embedding query   | ---> | Retrieval hibrido             |
| usuario   |      | local             |      | (semantico + keyword + rank)  |
+-----------+      +-------------------+      +---------------+---------------+
                                                                |
                                                                v
                                                   +------------+-------------+
                                                   | Prompt RAG + Gemini chat |
                                                   | (resposta com contexto)   |
                                                   +------------+-------------+
                                                                |
                                                                v
                                                   +------------+-------------+
                                                   | Resposta + Fontes        |
                                                   | (arquivo/pagina/chunk)   |
                                                   +--------------------------+

## 3) Pre-requisitos
- Python 3.11+
- Conta Google (para gerar API key no Google AI Studio)
- Sistema com permissao para instalar dependencias Python

## 4) Passo a passo de instalacao

1. Entre na pasta do projeto:

```powershell
cd rag-manuais
```

2. Crie e ative um ambiente virtual:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

3. Instale as dependencias:

```powershell
pip install -r requirements.txt
```

4. Configure o arquivo .env:

```powershell
copy .env.example .env
```

5. Edite o .env e preencha sua chave:

```env
GOOGLE_API_KEY=SUA_CHAVE_AQUI
```

## 5) Como obter API key gratuita do Google AI Studio
1. Acesse: https://aistudio.google.com
2. Faça login com sua conta Google.
3. Abra a area de API keys.
4. Clique em Create API key.
5. Copie a chave e cole no arquivo .env.

Observacoes:
- Nao compartilhe sua chave.
- Nao faca hardcode da chave no codigo.

## 6) Como rodar o app

```powershell
streamlit run app.py
```

Depois:
1. Faça upload de um ou mais PDFs na sidebar.
2. Clique em Processar documentos.
3. Faça perguntas no chat.
4. Abra "Fontes consultadas" para auditar os chunks usados.

Importante:
- Se voce alterar a estrategia de chunking/embedding/retrieval, limpe a base vetorial e reindexe os PDFs.
- Misturar vetores antigos e novos na mesma colecao prejudica a busca.

## 7) Estrutura de pastas explicada

rag-manuais/
- app.py: UI Streamlit e orquestracao principal
- ingest.py: pipeline PDF -> chunks -> embeddings locais -> ChromaDB
- retriever.py: retrieval hibrido com reranking
- llm.py: chamadas Gemini apenas para resposta final
- embeddings.py: embeddings locais
- chunker.py: chunking por passagens com overlap
- config.py: configuracoes centralizadas
- requirements.txt: dependencias Python
- .env.example: template para API key
- README.md: guia principal
- docs/
  - ARQUITETURA.md: arquitetura e fluxos
  - CONCEITOS.md: fundamentos de RAG
  - APRENDIZADOS.md: diario de estudos
- data/
  - pdfs/: PDFs enviados para indexacao
- vectorstore/: persistencia local do ChromaDB

## 8) Glossario rapido
- Chunk: trecho de texto menor extraido de um documento grande.
- Overlap: repeticao controlada de parte do chunk anterior no proximo.
- Embedding: vetor numerico que representa significado semantico.
- Vetor: lista de numeros em um espaco matematico.
- Similaridade: medida de "proximidade" entre dois vetores.
- Retrieval: etapa de busca dos trechos mais relevantes.
- Prompt: instrucao enviada ao LLM.
- Contexto: informacao recuperada usada para responder.
- RAG: estrategia de combinar retrieval com generation.

## Tratamento de erros implementado no app
- API key ausente: instrucoes claras na tela.
- PDF sem texto extraivel: aviso amigavel durante ingestao.
- Busca sem resultado relevante: app informa que nao encontrou contexto suficiente.
- Erros de API/rede: try/except com mensagens explicitas.
- Estrategia de indexacao divergente: aviso de reindexacao no app.

## Observacao tecnica importante
- O cliente do ChromaDB e cacheado com st.cache_resource para melhorar desempenho.
- A base vetorial e persistida localmente em ./vectorstore.
- O modo "Mostrar debug de retrieval" exibe distancias e scores para diagnostico.
