# Conceitos Fundamentais de RAG

## 1) O que sao embeddings e por que sao uteis
Embeddings sao vetores numericos que representam significado semantico de um texto. Em vez de comparar apenas palavras exatas, comparamos vetores para localizar trechos "parecidos em sentido".

Neste projeto:
- Pergunta do usuario vira embedding local.
- Chunks dos PDFs tambem viram embeddings locais.
- A busca vetorial retorna os chunks mais proximos semanticamente.

## 2) O que e similaridade de cosseno
A similaridade de cosseno mede o angulo entre dois vetores:
- mais proximo de 1: vetores mais alinhados (mais similares)
- mais proximo de 0: pouco relacionados

No ChromaDB, geralmente recebemos distancia: quanto menor a distancia, maior a similaridade.

## 3) Por que nao mandamos o PDF inteiro pro LLM
Motivos principais:
- Limite de contexto do modelo.
- Custo e latencia maiores.
- Muito ruido no prompt aumenta chance de alucinacao.

Por isso usamos RAG: recuperamos apenas os melhores trechos e entregamos ao LLM.

## 4) O que e chunking e por que o overlap importa
Chunking = quebrar texto longo em partes menores.

Sem overlap:
- Conceitos na fronteira entre chunks podem ser cortados.

Com overlap:
- Parte final do chunk anterior entra no proximo.
- A chance de recuperar contexto completo aumenta.

Neste projeto, o chunking e por passagens (paragrafos/sentencas), preservando melhor significado.

## 5) O que o ChromaDB faz por baixo dos panos
ChromaDB e um banco vetorial local que:
- Armazena embeddings
- Armazena documentos e metadados
- Executa busca de vizinhos mais proximos

A base persiste em ./vectorstore, entao os dados continuam entre execucoes.

## 6) Busca por keyword vs busca semantica
Busca por keyword:
- Depende de termos exatos
- Boa para siglas e termos tecnicos (ex: MRR, ARR)

Busca semantica:
- Compara significado
- Boa para perguntas em linguagem natural

Neste projeto usamos retrieval hibrido:
- Parte semantica (embedding)
- Parte keyword (sobreposicao de termos)
- Reranking combinado para selecionar melhores chunks

## 7) Como aproximar a visao do "arquivo inteiro"
Nao enviamos o arquivo inteiro ao LLM, mas usamos duas tecnicas:
- Chunks maiores e mais coesos por passagens.
- Um chunk de resumo global do documento (document_summary), anexado ao contexto.

Isso melhora respostas de alto nivel sem estourar janela de contexto.

## 8) Por que reindexar quando estrategia muda
Se voce altera chunking, embedding ou ranking, vetores antigos ficam em outro "espaco" de representacao.
Misturar estrategias na mesma colecao reduz qualidade de retrieval.

Boa pratica:
- Limpar base vetorial
- Reprocessar todos os PDFs
