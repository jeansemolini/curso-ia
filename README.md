# 🤖 Curso IA - Especialização IA Dev Eficiente

Um projeto de especialização em IA focado no desenvolvimento eficiente com IA, incluindo fundamentos de NLP, extração de documentos, embeddings, sistemas RAG (Retrieval-Augmented Generation) e análise de documentos financeiros com busca híbrida.

## 📋 Estrutura do Projeto

```
curso-ia/
├── fundamentos/                    # Conceitos fundamentais de NLP
│   ├── tokenization-01.py             # Tokenização básica com NLTK
│   ├── tokenization-02.py             # Tokenização avançada
│   ├── tokenization-03.py             # Análise de frequência de tokens
│   └── tokenization-04.py             # BM25 para busca por ranking
│
├── docling/                        # Extração e processamento de documentos PDF
│   ├── 1-extration.py                 # Extração básica de documentos
│   ├── 2-extraction-images.py         # Extração com imagens
│   ├── 3-chunking.py                  # Divisão em chunks
│   ├── 4-hybrid-chunker.py            # Chunking híbrido
│   ├── 5-metadados.py                 # Extração de metadados com API
│   ├── 6-embeddings.py                # Geração de embeddings
│   └── 2408.09869v5.pdf               # Documento de exemplo
│
├── llm/                            # Integração com Large Language Models
│   ├── llm-01.py                      # Utilização da API Groq
│   └── llm-02.py                      # Interações avançadas com LLMs
│
├── rag/                            # Sistema de Retrieval-Augmented Generation
│   ├── rag.py                         # RAG com busca vetorial
│   └── rag-qdrant.py                  # RAG usando Qdrant
│
├── projeto/                        # Análise financeira com busca híbrida (Dense + Sparse + ColBERT)
│   ├── create_collection.py           # Criação de coleção Qdrant com vetores híbridos
│   ├── ingestion.py                   # Ingestion de filings SEC (10-K e 10-Q)
│   ├── test-query.py                  # Query com busca híbrida e RRF fusion
│   ├── app/                           # API FastAPI para processamento de eventos
│   │   ├── main.py                    # Aplicação principal FastAPI
│   │   ├── router.py                  # Orquestrador de rotas
│   │   └── endpoint.py                # Implementação de endpoints
│   ├── utils/
│   │   ├── edgar_client.py            # Cliente para fetching de EDGAR filings
│   │   └── semantic_chunker.py        # Chunking semântico com HDBSCAN
│   └── AAPL_10-K_1A_temp.md           # Análise de Risk Factors - Apple
│
├── pyproject.toml                  # Dependências do projeto
├── .env                            # Variáveis de ambiente (não commitar)
└── README.md                       # Este arquivo
```

## 🚀 Tecnologias Utilizadas

- **Processamento de Texto**: NLTK, Whoosh, BM25
- **Embeddings**: Sentence Transformers, FastEmbed, ColBERT
- **Extração de Documentos**: Docling (IBM)
- **LLMs**: Groq, OpenAI
- **Vector Database**: Qdrant (com busca híbrida)
- **Filings Financeiros**: EdgarTools (SEC EDGAR)
- **Clustering Semântico**: HDBSCAN
- **Machine Learning**: Scikit-learn
- **Parsing Estruturado**: Pydantic
- **Linguagem**: Python 3.12+

## 📦 Dependências Principais

O projeto utiliza as seguintes bibliotecas principais:

```
docling>=2.65.0                  # Extração de documentos PDF
edgartools>=5.6.4                # Acesso a SEC EDGAR filings
fastembed>=0.7.4                 # Embeddings rápidos (dense, sparse, ColBERT)
groq>=1.0.0                      # API Groq para LLMs
hdbscan>=0.8.41                  # Clustering semântico
langextract>=1.1.1               # Extração de linguagem
nltk>=3.9.2                      # NLP
openai>=2.6.1                    # API OpenAI
python-dotenv>=1.2.1             # Gerenciamento de variáveis de ambiente
qdrant-client>=1.16.2            # Vector database com busca híbrida
rank-bm25>=0.2.2                 # Algoritmo BM25
sentence-transformers>=5.2.0     # Sentence embeddings
whoosh>=2.7.4                    # Full-text search
```

Veja [pyproject.toml](pyproject.toml) para a lista completa.

## 🔧 Instalação

### Pré-requisitos
- Python 3.12 ou superior
- pip ou uv (gerenciador de pacotes)

### Passos

1. **Clone o repositório**
```bash
git clone <repository-url>
cd curso-ia
```

2. **Crie e ative um ambiente virtual**

**Opção A: Usando `uv` (recomendado)**
```bash
# Criar ambiente virtual com uv
uv venv

# Ativar ambiente virtual
source .venv/bin/activate  # macOS/Linux
# ou no Windows: .venv\Scripts\activate
```

**Opção B: Usando Python nativo**
```bash
# Criar ambiente virtual
python -m venv .venv

# Ativar ambiente virtual
source .venv/bin/activate  # macOS/Linux
# ou no Windows: .venv\Scripts\activate
```

3. **Instale as dependências**

**Com `uv` (mais rápido)**
```bash
uv sync
```

**Com pip**
```bash
pip install -e .
```

4. **Configure as variáveis de ambiente**
```bash
cp .env.example .env
# Edite .env com suas chaves de API
```

**Variáveis de ambiente necessárias:**
- `GROQ_API_KEY` - Chave da API Groq
- `OPENAI_API_KEY` - Chave da API OpenAI
- `GOOGLE_API_KEY` - Chave da API Google

### Verificar Ativação do Ambiente

```bash
# Se venv está ativado, você verá (.venv) no prompt
$ (.venv) python --version
Python 3.12.x

# Ou use
which python  # macOS/Linux
# deve apontar para: /path/to/projeto/.venv/bin/python
```

### Desativar Ambiente Virtual
```bash
deactivate
```

## 📚 Módulos Principais

### 1. **Fundamentos** (`fundamentos/`)
Introdução aos conceitos de NLP e processamento de texto:
- Tokenização com NLTK
- Análise de frequência
- Algoritmo BM25 para ranking de documentos

```bash
python fundamentos/tokenization-01.py
```

### 2. **Docling** (`docling/`)
Extração e processamento de documentos PDF usando a biblioteca Docling da IBM:
- Extração de texto e imagens
- Chunking inteligente
- Extração de metadados
- Geração de embeddings

```bash
python docling/1-extration.py
```

### 3. **LLM** (`llm/`)
Integração com modelos de linguagem:
- Utilização da API Groq
- Chamadas e streaming
- Processamento de respostas

```bash
python llm/llm-01.py
```

### 4. **RAG** (`rag/`)
Sistema de Retrieval-Augmented Generation:
- Busca vetorial com embeddings
- Integração com Qdrant
- Recuperação de contexto para LLMs

```bash
python rag/rag.py
python rag/rag-qdrant.py
```

### 5. **Projeto** (`projeto/`)
Análise de documentos financeiros da SEC com **busca híbrida** (Dense + Sparse + ColBERT):

#### Fluxo Completo:
1. **Fetching** de SEC 10-K e 10-Q filings com EdgarTools
2. **Chunking semântico** usando HDBSCAN
3. **Embeddings híbridos**:
   - **Dense**: `sentence-transformers/all-MiniLM-L6-v2` (384D)
   - **Sparse**: `Qdrant/bm25` (BM25 ranking)
   - **ColBERT**: `colbert-ir/colbertv2.0` (late interaction)
4. **Query com RRF Fusion** (Reciprocal Rank Fusion)

#### Scripts:
- `create_collection.py` - Cria coleção Qdrant com vetores híbridos
- `ingestion.py` - Faz download e ingestão de filings
- `test-query.py` - Testa queries com busca híbrida

```bash
# 1. Criar coleção (precisa executar uma única vez)
python projeto/create_collection.py

# 2. Ingestão de dados
python projeto/ingestion.py

# 3. Testar queries
python projeto/test-query.py
```

#### Exemplo de Query com Busca Híbrida:
```python
from projeto.utils.edgar_client import EdgarClient
from projeto.utils.semantic_chunker import SemanticChunker
from fastembed import TextEmbedding, SparseTextEmbedding, LateInteractionTextEmbedding
from qdrant_client import QdrantClient, models

# Query in English
query = "what are the main financial risks?"

# Gerar embeddings híbridos
dense_embed = list(dense_model.query_embed([query]))[0].tolist()
sparse_embed = list(sparse_model.query_embed([query]))[0].as_object()
colbert_embed = list(colbert_model.query_embed([query]))[0].tolist()

# Buscar com Reciprocal Rank Fusion
results = qdrant.query_points(
    collection_name="financial",
    prefetch=[
        {
            "prefetch": [
                {"query": dense_embed, "using": "dense", "limit": 10},
                {"query": sparse_embed, "using": "sparse", "limit": 10},
            ],
            "query": models.FusionQuery(fusion=models.Fusion.RRF),
            "limit": 20,
        }
    ],
    query=colbert_embed,
    using="colbert",
    limit=3,
)
```

## 🔐 Configuração de Chaves de API

O projeto utiliza variáveis de ambiente para gerenciar as chaves de API de forma segura. Crie um arquivo `.env` na raiz do projeto:

```bash
GROQ_API_KEY="sua_chave_aqui"
OPENAI_API_KEY="sua_chave_aqui"
GOOGLE_API_KEY="sua_chave_aqui"
QDRANT_URL="sua_url_aqui"
QDRANT_API_KEY="sua_chave_aqui"
```

**Variáveis de ambiente necessárias:**
- `GROQ_API_KEY` - Chave da API Groq
- `OPENAI_API_KEY` - Chave da API OpenAI
- `GOOGLE_API_KEY` - Chave da API Google
- `QDRANT_URL` - URL do servidor Qdrant
- `QDRANT_API_KEY` - Chave de autenticação do Qdrant

**Nunca faça commit de arquivos `.env` com chaves reais!**

## 📖 Exemplos de Uso

### Tokenização com BM25
```python
from rank_bm25 import BM25Okapi
import nltk

corpus = [
    "Este é um exemplo de documento",
    "Outro documento para teste",
    "Terceiro documento aqui"
]

tokenized_corpus = [doc.split() for doc in corpus]
bm25 = BM25Okapi(tokenized_corpus)

query = "example document"
scores = bm25.get_scores(query.split())
```

### Extração de Documentos PDF com Metadados
```python
import os
from dotenv import load_dotenv
import langextract as lx
from docling.document_converter import DocumentConverter

load_dotenv()

converter = DocumentConverter()
result = converter.convert("documento.pdf")
markdown = result.document.export_to_markdown()

# Extrair metadados usando Gemini
extraction_result = lx.extract(
    text_or_documents=markdown[:4000],
    prompt_description="Extraia título, autores, afiliação e URLs de repositório",
    model_id="gemini-2.5-flash",
    api_key=os.getenv("GOOGLE_API_KEY"),
)
```

### RAG com Qdrant (busca simples)
```python
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

client = QdrantClient(
    url="https://seu-url-qdrant.io",
    api_key="sua-api-key"
)
model = SentenceTransformer("all-MiniLM-L6-v2")

# Gerar embedding e buscar
query_embedding = model.encode("what is the subject of this document?")
results = client.search(
    collection_name="documents",
    query_vector=query_embedding,
    limit=5
)
```

### Chunking Semântico com HDBSCAN
```python
from projeto.utils.semantic_chunker import SemanticChunker

chunker = SemanticChunker(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    min_cluster_size=3,
    max_tokens=300
)

text = "Your text here..."
chunks = chunker.create_chunks(text)
print(f"Criados {len(chunks)} chunks semânticos")
```

### Fetching de EDGAR com EdgarTools
```python
from projeto.utils.edgar_client import EdgarClient

client = EdgarClient(email="seu_email@example.com")

# Fetch 10-K filing
data_10k = client.fetch_filing_data("AAPL", "10-K")
text_10k = client.get_combined_text(data_10k)

# Metadados extraídos
print(f"Empresa: {data_10k['metadata']['company_name']}")
print(f"Data: {data_10k['metadata']['report_date']}")
```

## 🎯 Fluxo de Uso Completo - Projeto Financeiro

### 1. **Setup Inicial**
```bash
# Instalar dependências
uv install

# Configurar variáveis de ambiente
cp .env.example .env
# Editar .env com suas chaves
```

### 2. **Criar Coleção Qdrant**
```bash
python projeto/create_collection.py
```
Cria uma coleção com esquema de vetores híbridos:
- `dense`: 384 dimensões (all-MiniLM-L6-v2)
- `sparse`: BM25 vectors
- `colbert`: 128 dimensões com MultiVectorConfig

### 3. **Ingestão de Dados**
```bash
python projeto/ingestion.py
```
Processa:
- 10-K filing (relatório anual)
- 10-Q filing (relatório trimestral)
- Extrai itens relevantes (Risk Factors, MD&A, etc)
- Faz chunking semântico
- Gera embeddings híbridos
- Upload para Qdrant

### 4. **Query com Busca Híbrida**
```bash
python projeto/test-query.py
```
Demonstra busca com RRF Fusion:
- Dense similarity search
- Sparse BM25 search
- ColBERT late interaction
- Combinação com Reciprocal Rank Fusion

### 5. **API FastAPI para Processamento de Eventos**
```bash
cd projeto/app
uv run uvicorn main:app --reload --port 8001
```

Acesse:
- **API**: http://127.0.0.1:8001/events/
- **Documentação**: http://127.0.0.1:8001/docs (Swagger UI)
- **ReDoc**: http://127.0.0.1:8001/redoc

#### Exemplo de cURL
```bash
curl -X POST http://127.0.0.1:8001/events/ \
  -H "Content-Type: application/json" \
  -d '{"event_id":"123","event_type":"user_signup","event_data":{"name":"João"}}'
```

#### Arquitetura da API
```
main.py (entrada)
    ↓
    └─→ app = FastAPI()
        app.include_router(process_router)
            ↓
        router.py (orquestrador)
            ↓
            └─→ router.include_router(endpoint.router, prefix="/events")
                ↓
            endpoint.py (implementação)
                ↓
                └─→ POST /events/ → handle_event()
```

**Estrutura de Arquivos:**
- `main.py`: Aplicação principal, registra routers
- `router.py`: Orquestra rotas, agrupa endpoints
- `endpoint.py`: Define schemas (Pydantic) e implementa endpoints

## � Conceitos Principais

### Busca Híbrida (Dense + Sparse + ColBERT)

A busca híbrida combina múltiplas estratégias para melhor relevância:

| Tipo | Modelo | Dimensões | Vantagem |
|------|--------|-----------|----------|
| **Dense** | all-MiniLM-L6-v2 | 384D | Captura semântica geral |
| **Sparse** | BM25 | Variável | Busca por palavras-chave exatas |
| **ColBERT** | colbertv2.0 | 128D (multi) | Late interaction, melhor precisão |

**RRF (Reciprocal Rank Fusion)**: Combina rankings de múltiplas buscas para resultado final.

### Chunking Semântico com HDBSCAN

Em vez de chunks de tamanho fixo:
1. Divide texto em parágrafos
2. Gera embeddings dos parágrafos
3. Usa HDBSCAN para encontrar clusters semânticos
4. Combina parágrafos do mesmo cluster
5. Respeita limite de tokens (max_tokens)

Resultado: chunks que mantêm coerência semântica!

### EdgarTools para SEC EDGAR

Acessa automaticamente:
- **10-K**: Relatório anual completo
- **10-Q**: Relatório trimestral
- **Extrai**: Item 1 (Negócio), Item 1A (Risk Factors), Item 7 (MD&A), etc.

```python
data = client.fetch_filing_data("AAPL", "10-K")
# Retorna: metadata + items estruturados
```

---

## 📝 Licença

Este projeto é parte de um curso de especialização em IA. Verifique a licença específica do curso antes de usar em produção.

## 👤 Autor

Desenvolvido durante o curso de Especialização em IA Dev Eficiente.

## 📞 Suporte e Referências

Para dúvidas e suporte, verifique a documentação das bibliotecas utilizadas:

- [Docling Documentation](https://ds4sd.github.io/docling/)
- [Qdrant Documentation](https://qdrant.tech/documentation/)
- [FastEmbed (Dense, Sparse, ColBERT)](https://github.com/qdrant/fastembed)
- [EdgarTools Documentation](https://github.com/dgunning/edgartools)
- [Sentence Transformers](https://www.sbert.net/)
- [NLTK Book](https://www.nltk.org/book/)
- [HDBSCAN](https://hdbscan.readthedocs.io/)
- [Groq API](https://groq.com/)

---

**Última atualização**: Dezembro de 2025
