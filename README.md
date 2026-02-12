# 🤖 Curso IA - Especialização IA Dev Eficiente

Um projeto completo de especialização em IA focado no desenvolvimento eficiente com tecnologias modernas de IA e Machine Learning. Inclui fundamentos de NLP, extração inteligente de documentos, embeddings avançados, sistemas RAG, processamento de agentes com memória persistente, workflows de IA, análise de documentos financeiros com busca híbrida e muito mais.

## 📋 Estrutura do Projeto

```
curso-ia/
├── fundamentos/                    # 📚 Conceitos fundamentais de NLP
│   ├── tokenization-01.py             # Tokenização básica com NLTK
│   ├── tokenization-02.py             # Tokenização avançada e análise
│   ├── tokenization-03.py             # Análise de frequência de tokens
│   └── tokenization-04.py             # BM25 para ranking de documentos
│
├── docling/                        # 📄 Extração e processamento de PDFs
│   ├── 1-extration.py                 # Extração básica com Docling
│   ├── 2-extraction-images.py         # Extração com preservação de imagens
│   ├── 3-chunking.py                  # Divisão inteligente em chunks
│   ├── 4-hybrid-chunker.py            # Chunking híbrido avançado
│   ├── 5-metadados.py                 # Extração de metadados com Gemini API
│   ├── 6-embeddings.py                # Geração de embeddings
│   ├── 2408.09869v5.pdf               # Documento PDF de exemplo
│   ├── db/                            # Base de dados local Qdrant
│   ├── images/                        # Imagens extraídas dos PDFs
│   └── test_output/                   # Saída de testes e metadados
│
├── llm/                            # 🧠 Integração com LLMs
│   ├── llm-01.py                      # Interação com API Groq
│   └── llm-02.py                      # Interações avançadas com LLMs
│
├── agents/                         # 🤖 Agentes de IA com Workflows
│   ├── exemplo-1-output-estruturado.py  # Parsing estruturado com Pydantic
│   ├── exemplo-2-openai.py             # Integração com OpenAI
│   ├── exemplo-2-tool.py               # Tool use (function calling)
│   ├── exemplo-3-retrieval.py          # Retrieval augmented generation
│   ├── memory-1.py                     # Memória simples com Mem0
│   ├── memory-2.py                     # Memória persistente avançada
│   ├── workflows-1-prompt-chaining.py  # Prompt chaining (passo a passo)
│   ├── workflows-2-routing.py          # Routing inteligente entre workflows
│   ├── workflow-3-parallelization.py   # Execução paralela de tasks
│   └── workflows-3-parallelization.py  # Paralelização avançada
│
├── rag/                            # 🔍 Retrieval-Augmented Generation
│   ├── rag.py                         # RAG com busca vetorial simples
│   ├── rag-qdrant.py                  # RAG com Qdrant
│   └── db/                            # Base de dados de documentos
│
├── projeto/                        # 💰 Análise Financeira com Busca Híbrida
│   ├── create_collection.py           # Setup: Criar coleção Qdrant híbrida
│   ├── ingestion.py                   # Ingestão de SEC EDGAR filings
│   ├── test-query.py                  # Query com busca híbrida (RRF)
│   ├── AAPL_10-K_1A_temp.md           # Exemplo: Risk Factors da Apple
│   ├── app/                           # 🚀 API FastAPI
│   │   ├── main.py                    # Inicialização FastAPI
│   │   ├── router.py                  # Orquestração de rotas
│   │   └── endpoint.py                # Implementação de endpoints
│   └── utils/
│       ├── edgar_client.py            # Cliente SEC EDGAR
│       └── semantic_chunker.py        # Chunking semântico com HDBSCAN
│
├── pyproject.toml                  # Dependências do projeto
├── uv.lock                         # Lock file (uv)
├── .env.example                    # Template de variáveis de ambiente
├── .env                            # Variáveis de ambiente (não commitar!)
├── .python-version                 # Versão Python do projeto
└── README.md                       # Este arquivo
```

---

## 🚀 Tecnologias Utilizadas

### 🧠 Modelos e LLMs
- **Groq API**: Modelos open-source de alta velocidade (Llama, Mixtral)
- **OpenAI API**: GPT-4, GPT-3.5-turbo
- **Google Gemini API**: Extração de metadados e análise

### 📊 Processamento e NLP
- **NLTK**: Tokenização e análise de texto
- **Whoosh**: Full-text search local
- **BM25**: Ranking de documentos por relevância
- **LangExtract**: Detecção de idioma e extração

### 🧬 Embeddings e Vetores
- **Sentence Transformers**: all-MiniLM-L6-v2 (384D)
- **FastEmbed**: Dense, Sparse (BM25) e ColBERT embeddings
- **ColBERT**: Late interaction embeddings (128D)

### 📄 Processamento de Documentos
- **Docling (IBM)**: Extração inteligente de PDFs com OCR
- **EdgarTools**: Acesso a SEC EDGAR filings (10-K, 10-Q)

### 🗄️ Bancos de Dados
- **Qdrant**: Vector database com busca híbrida
- **Whoosh**: Índices full-text locais

### 🤖 Agentes e Workflows
- **Mem0**: Memória persistente para agentes
- **Pydantic**: Parsing estruturado de LLM outputs
- **Tool Use**: Function calling para agentes

### 🔧 Outras Ferramentas
- **HDBSCAN**: Clustering semântico para chunking
- **scikit-learn**: Machine Learning utilities
- **yfinance**: Dados financeiros
- **FastAPI & Uvicorn**: API web
- **Python 3.12+**: Runtime

---

## 📦 Dependências Principais

| Pacote | Versão | Propósito |
|--------|--------|----------|
| `docling` | ≥2.65.0 | Extração de PDFs com IA |
| `edgartools` | ≥5.6.4 | Acesso SEC EDGAR filings |
| `fastembed` | ≥0.7.4 | Embeddings (dense, sparse, ColBERT) |
| `groq` | ≥1.0.0 | API Groq para LLMs |
| `openai` | ≥2.6.1 | API OpenAI |
| `langextract` | ≥1.1.1 | Detecção de idioma |
| `mem0ai` | latest | Memória persistente para agentes |
| `nltk` | ≥3.9.2 | NLP e tokenização |
| `qdrant-client` | ≥1.16.2 | Vector database |
| `rank-bm25` | ≥0.2.2 | BM25 ranking |
| `sentence-transformers` | ≥5.2.0 | Sentence embeddings |
| `fastapi` | ≥0.128.0 | Framework web |
| `uvicorn` | ≥0.40.0 | ASGI server |
| `pydantic` | ≥2.12.3 | Parsing estruturado |
| `hdbscan` | ≥0.8.41 | Clustering semântico |
| `scikit-learn` | ≥1.8.0 | ML utilities |
| `whoosh` | ≥2.7.4 | Full-text search |
| `yfinance` | ≥1.0 | Dados financeiros |
| `python-dotenv` | ≥1.2.1 | Variáveis de ambiente |

Veja [pyproject.toml](pyproject.toml) para a lista completa com todas as versões exatas.

---

## 📚 Módulos Principais do Projeto

### 1. **Fundamentos** (`fundamentos/`) - Conceitos Essenciais de NLP

Introdução aos conceitos fundamentais de processamento de linguagem natural:

| Arquivo | Descrição |
|---------|-----------|
| `tokenization-01.py` | Tokenização básica com NLTK |
| `tokenization-02.py` | Tokenização avançada com stemming/lemmatization |
| `tokenization-03.py` | Análise de frequência de tokens e word clouds |
| `tokenization-04.py` | Algoritmo BM25 para ranking de documentos |

**Exemplo de uso:**
```bash
python fundamentos/tokenization-01.py
python fundamentos/tokenization-04.py  # BM25 ranking
```

---

### 2. **Docling** (`docling/`) - Extração Inteligente de PDFs

Processamento completo de documentos PDF usando Docling (IBM):

| Arquivo | Descrição |
|---------|-----------|
| `1-extration.py` | Extração básica de texto de PDFs |
| `2-extraction-images.py` | Extração preservando imagens e figuras |
| `3-chunking.py` | Divisão inteligente em chunks fixos |
| `4-hybrid-chunker.py` | Chunking híbrido (tamanho + semântica) |
| `5-metadados.py` | Extração de metadados com Gemini API |
| `6-embeddings.py` | Geração de embeddings e storage em Qdrant |

**Exemplo de uso:**
```bash
python docling/1-extration.py
python docling/6-embeddings.py  # Gerar embeddings e salvar em DB
```

---

### 3. **LLM** (`llm/`) - Integração com Large Language Models

Exemplos de integração com LLMs via Groq e OpenAI:

| Arquivo | Descrição |
|---------|-----------|
| `llm-01.py` | Utilização da API Groq com streaming |
| `llm-02.py` | Interações avançadas, tool use, parsing |

**Exemplo de uso:**
```bash
python llm/llm-01.py
```

---

### 4. **Agents** (`agents/`) - Agentes de IA com Workflows e Memória

Sistema completo de agentes com suporte a workflows, memória persistente e tool use:

#### Estruturado (Output Parsing)
| Arquivo | Descrição |
|---------|-----------|
| `exemplo-1-output-estruturado.py` | Parsing estruturado com Pydantic |
| `exemplo-2-tool.py` | Function calling (tool use) |

#### Workflows (Orquestração)
| Arquivo | Descrição |
|---------|-----------|
| `workflows-1-prompt-chaining.py` | Prompt chaining (múltiplos passos) |
| `workflows-2-routing.py` | Routing inteligente entre diferentes fluxos |
| `workflow-3-parallelization.py` | Execução paralela de tasks |
| `workflows-3-parallelization.py` | Paralelização avançada |

#### Memória
| Arquivo | Descrição |
|---------|-----------|
| `memory-1.py` | Memória simples com Mem0 |
| `memory-2.py` | Memória persistente e contexto |

#### Retrieval
| Arquivo | Descrição |
|---------|-----------|
| `exemplo-3-retrieval.py` | RAG com retrieval aumentado |

**Exemplo de uso:**
```bash
python agents/exemplo-1-output-estruturado.py
python agents/workflows-1-prompt-chaining.py
python agents/memory-1.py
```

---

### 5. **RAG** (`rag/`) - Retrieval-Augmented Generation

Sistema completo de RAG com diferentes estratégias de busca:

| Arquivo | Descrição |
|---------|-----------|
| `rag.py` | RAG com busca vetorial simples (SentenceTransformers) |
| `rag-qdrant.py` | RAG com Qdrant vector database |

**Exemplo de uso:**
```bash
python rag/rag.py
python rag/rag-qdrant.py
```

---

### 6. **Projeto** (`projeto/`) - Análise Financeira com Busca Híbrida

Sistema end-to-end de análise de documentos financeiros SEC com **busca híbrida** (Dense + Sparse + ColBERT):

#### Setup Inicial
```bash
# 1. Criar coleção Qdrant com vectors híbridos
python projeto/create_collection.py
```

#### Ingestão de Dados
```bash
# 2. Fazer download e processar SEC EDGAR filings
python projeto/ingestion.py
```

#### Query e Busca
```bash
# 3. Testar queries com RRF Fusion
python projeto/test-query.py
```

#### API FastAPI
```bash
# 4. Iniciar API de processamento
cd projeto/app
uvicorn main:app --reload --port 8001

# Documentação: http://127.0.0.1:8001/docs
```

**Arquitetura:**
- `create_collection.py`: Setup inicial (executa uma única vez)
- `ingestion.py`: Fetching → Chunking → Embeddings → Upload Qdrant
- `test-query.py`: Query com múltiplas estratégias de busca
- `app/main.py`: Aplicação FastAPI
- `app/router.py`: Orquestrador de rotas
- `app/endpoint.py`: Implementação de endpoints
- `utils/edgar_client.py`: Cliente SEC EDGAR
- `utils/semantic_chunker.py`: Chunking semântico com HDBSCAN

---

## 🔧 Instalação e Setup

### Pré-requisitos
- Python 3.12 ou superior
- pip ou `uv` (gerenciador de pacotes recomendado)
- Chaves de API (Groq, OpenAI, Google Gemini)

### Passos de Instalação

#### 1️⃣ Clone o Repositório
```bash
git clone <repository-url>
cd curso-ia
```

#### 2️⃣ Crie e Ative um Ambiente Virtual

**Opção A: Usando `uv` (recomendado)** ⚡
```bash
# Instalar uv (se ainda não tiver)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Criar ambiente virtual com uv
uv venv

# Ativar ambiente virtual
source .venv/bin/activate  # macOS/Linux
# ou no Windows: .venv\Scripts\activate
```

**Opção B: Usando Python nativo**
```bash
# Criar ambiente virtual
python3.12 -m venv .venv

# Ativar ambiente virtual
source .venv/bin/activate  # macOS/Linux
# ou no Windows: .venv\Scripts\activate
```

#### 3️⃣ Instale as Dependências

**Com `uv` (mais rápido)** ⚡
```bash
uv sync
```

**Com pip**
```bash
pip install -e .
```

#### 4️⃣ Configure as Variáveis de Ambiente
```bash
# Copiar template
cp .env.example .env

# Editar com suas chaves
nano .env  # ou use seu editor de texto preferido
```

**Variáveis necessárias:**
```bash
# APIs de LLMs
GROQ_API_KEY="sua-chave-groq"
OPENAI_API_KEY="sua-chave-openai"
GOOGLE_API_KEY="sua-chave-google"

# Qdrant (opcional, se usar versão hosted)
QDRANT_URL="https://seu-qdrant.io"
QDRANT_API_KEY="sua-api-key"

# Email para SEC EDGAR (obrigatório)
SEC_EMAIL="seu-email@example.com"
```

#### 5️⃣ Verificar Instalação
```bash
# Python deve estar na versão correta
python --version  # deve ser 3.12+

# Se .venv está ativado, você verá (.venv) no prompt:
(.venv) $ python --version
Python 3.12.x

# Verificar localização do Python (deve apontar para .venv)
which python  # macOS/Linux
# /path/to/projeto/.venv/bin/python
```

### 🔄 Desativar Ambiente Virtual
```bash
deactivate
```

---

## 🎯 Guia Rápido de Uso

### 📚 Executar Exemplos Fundamentais
```bash
# Tokenização básica
python fundamentos/tokenization-01.py

# BM25 ranking (mais interessante)
python fundamentos/tokenization-04.py
```

### 📄 Processar Documentos com Docling
```bash
# Extrair texto de PDF
python docling/1-extration.py

# Extrair com imagens preservadas
python docling/2-extraction-images.py

# Gerar embeddings (salva em Qdrant)
python docling/6-embeddings.py
```

### 🧠 Usar LLMs
```bash
# Chamar Groq API
python llm/llm-01.py
```

### 🤖 Executar Agentes e Workflows
```bash
# Parsing estruturado
python agents/exemplo-1-output-estruturado.py

# Prompt chaining
python agents/workflows-1-prompt-chaining.py

# Memória persistente
python agents/memory-1.py
```

### 🔍 Sistema RAG
```bash
# RAG com busca vetorial
python rag/rag.py

# RAG com Qdrant
python rag/rag-qdrant.py
```

### 💰 Projeto Financeiro (Busca Híbrida)
```bash
# IMPORTANTE: Execute na ordem abaixo

# 1. Setup inicial (executa UMA ÚNICA VEZ)
python projeto/create_collection.py

# 2. Fazer download e processar filings
python projeto/ingestion.py

# 3. Testar queries com busca híbrida
python projeto/test-query.py

# 4. Iniciar API FastAPI
cd projeto/app
uvicorn main:app --reload --port 8001
# Documentação: http://127.0.0.1:8001/docs
```

---

## 🎓 Exemplos de Código Detalhados

### Exemplo 1: Tokenização com BM25
```python
from rank_bm25 import BM25Okapi
import nltk
from nltk.tokenize import word_tokenize

# Corpus de documentos
corpus = [
    "Machine learning é uma subárea da IA",
    "Deep learning utiliza redes neurais",
    "Processamento de linguagem natural é fascinante",
    "Embeddings transformam texto em vetores"
]

# Tokenizar corpus
tokenized_corpus = [word_tokenize(doc.lower()) for doc in corpus]

# Criar modelo BM25
bm25 = BM25Okapi(tokenized_corpus)

# Query
query = "machine learning neural networks"
tokenized_query = word_tokenize(query.lower())

# Score dos documentos
scores = bm25.get_scores(tokenized_query)

# Ranking
ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
for idx, score in ranked:
    print(f"Doc {idx}: {corpus[idx]} (score: {score:.2f})")
```

### Exemplo 2: Extração de PDF com Docling
```python
from docling.document_converter import DocumentConverter
from pathlib import Path

# Converter PDF para Markdown
converter = DocumentConverter()
result = converter.convert("documento.pdf")

# Markdown estruturado
markdown = result.document.export_to_markdown()
print(markdown)

# Extrair metadados
metadata = result.document.metadata
print(f"Título: {metadata.get('title', 'N/A')}")
```

### Exemplo 3: Chamar Groq LLM
```python
from groq import Groq
import os

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Criar chat
response = client.chat.completions.create(
    model="mixtral-8x7b-32768",
    messages=[
        {
            "role": "user",
            "content": "Explique embedding de texto em 3 linhas"
        }
    ]
)

print(response.choices[0].message.content)
```

### Exemplo 4: Memória Persistente com Mem0
```python
from mem0 import MemoryClient

client = MemoryClient()

# Adicionar memória do usuário
client.add(
    "Meu nome é João e trabalho com IA",
    user_id="joao"
)

# Recuperar memórias relevantes
memories = client.search(
    "Qual é meu nome?",
    filters={"user_id": "joao"}
)

print(memories["results"][0]["memory"])
```

### Exemplo 5: RAG com Qdrant
```python
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

# Inicializar
client = QdrantClient(":memory:")  # ou URL remota
model = SentenceTransformer("all-MiniLM-L6-v2")

# Gerar embedding para query
query = "Qual é o melhor modelo de IA para processamento de texto?"
query_embedding = model.encode(query).tolist()

# Buscar no Qdrant
results = client.search(
    collection_name="documents",
    query_vector=query_embedding,
    limit=5
)

for result in results:
    print(f"Score: {result.score:.2f} - {result.payload}")
```

### Exemplo 6: Busca Híbrida (Projeto Financeiro)
```python
from fastembed import TextEmbedding, SparseTextEmbedding, LateInteractionTextEmbedding
from qdrant_client import QdrantClient, models

# Modelos
dense_model = TextEmbedding("sentence-transformers/all-MiniLM-L6-v2")
sparse_model = SparseTextEmbedding("Qdrant/bm25")
colbert_model = LateInteractionTextEmbedding("colbert-ir/colbertv2.0")

client = QdrantClient("http://localhost:6333")

# Query
query = "Quais são os principais riscos financeiros?"

# Gerar embeddings híbridos
dense_embed = list(dense_model.query_embed([query]))[0].tolist()
sparse_embed = list(sparse_model.query_embed([query]))[0].as_object()
colbert_embed = list(colbert_model.query_embed([query]))[0].tolist()

# Buscar com RRF Fusion
results = client.query_points(
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

for point in results:
    print(f"Score: {point.score:.2f}")
    print(f"Documento: {point.payload['text'][:100]}...\n")
```

### Exemplo 7: Parsing Estruturado com Pydantic
```python
from pydantic import BaseModel, Field
from groq import Groq

class CalendarEvent(BaseModel):
    name: str = Field(description="Nome do evento")
    date: str = Field(description="Data em ISO 8601")
    participants: list[str] = Field(description="Participantes")

client = Groq()

# Extrair estruturado
response = client.chat.completions.create(
    model="mixtral-8x7b-32768",
    messages=[
        {
            "role": "user",
            "content": "Daniel e Maria vão reunir na segunda-feira para discutir IA"
        }
    ]
)

# Parse com Pydantic
event = CalendarEvent.model_validate_json(response.choices[0].message.content)
print(f"Evento: {event.name} em {event.date}")
print(f"Participantes: {', '.join(event.participants)}")
```

---

## 🔐 Configuração de Chaves de API

### Obtendo as Chaves

#### 🔵 Groq
1. Acesse [console.groq.com](https://console.groq.com)
2. Faça login ou crie conta
3. Gere API Key
4. Copie a chave

#### 🟢 OpenAI
1. Acesse [platform.openai.com](https://platform.openai.com)
2. Vá em "API keys"
3. Crie nova chave
4. Copie a chave

#### 🔴 Google Gemini
1. Acesse [aistudio.google.com](https://aistudio.google.com)
2. Clique em "API keys"
3. Crie nova chave
4. Copie a chave

#### 🟠 Qdrant (Opcional)
1. Acesse [cloud.qdrant.io](https://cloud.qdrant.io)
2. Crie cluster
3. Obtenha URL e API Key

#### ✉️ SEC Email (Obrigatório para Projeto)
Use seu email pessoal/corporativo

### Arquivo .env
```bash
# .env (NUNCA commitar este arquivo!)
GROQ_API_KEY="gsk_yQtXYZ..."
OPENAI_API_KEY="sk-proj-..."
GOOGLE_API_KEY="AIzaSyD..."

QDRANT_URL="https://seu-qdrant-xxx.qdrant.io"
QDRANT_API_KEY="ey..."

SEC_EMAIL="seu-email@example.com"
```

---

## 🏗️ Conceitos-Chave Explicados

### 1️⃣ Embeddings (Vetores de Texto)
Embeddings transformam texto em vetores numéricos que capturam semântica:

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")
embedding = model.encode("Machine learning é IA")
# Resultado: array de 384 números

# Textos similares têm embeddings similares
embedding2 = model.encode("Deep learning é aprendizado profundo")
# Distância Euclidiana ou Cosine entre vectors mede similaridade
```

### 2️⃣ Busca Híbrida (Dense + Sparse + ColBERT)

Combina múltiplas estratégias para melhor relevância:

| Tipo | Modelo | Dimensões | Uso |
|------|--------|-----------|-----|
| **Dense** | all-MiniLM-L6-v2 | 384D | Semântica geral |
| **Sparse** | BM25 | Variável | Palavras-chave exatas |
| **ColBERT** | colbertv2.0 | 128D | Late interaction precisão |

**RRF (Reciprocal Rank Fusion)**: Combina rankings de múltiplas buscas.

### 3️⃣ Chunking Semântico com HDBSCAN

Em vez de chunks de tamanho fixo:
1. Divide texto em parágrafos
2. Gera embeddings dos parágrafos
3. HDBSCAN encontra clusters semânticos
4. Combina parágrafos do mesmo cluster
5. Respeita limite de tokens (max_tokens)

**Resultado**: chunks com coerência semântica!

### 4️⃣ Workflows de IA

#### **Prompt Chaining** (Múltiplos Passos)
```
Passo 1: Extrair informação → Passo 2: Analisar → Passo 3: Resumir
```

#### **Routing** (Decisão Inteligente)
```
User Input → LLM Decide Route → Execute fluxo apropriado
```

#### **Paralelização** (Execução Simultânea)
```
Task 1 ⊢ Final Result
Task 2 ⊣ (combina resultados)
Task 3 ⊤
```

### 5️⃣ SEC EDGAR Filing Structure
```
10-K (Anual):
├── Item 1: Business
├── Item 1A: Risk Factors ← Análise financeira
├── Item 7: Management's Discussion and Analysis (MD&A)
└── Item 8: Financial Statements

10-Q (Trimestral):
├── Part I: Financial Information
└── Part II: Other Information
```

---

## 📚 Fluxo de Uso Completo - Projeto Financeiro

### Etapa 1: Setup Inicial
```bash
# Instalar dependências
uv sync

# Configurar variáveis de ambiente
cp .env.example .env
# Editar .env com suas chaves
```

### Etapa 2: Criar Coleção Qdrant
```bash
python projeto/create_collection.py
```
**Cria coleção com esquema híbrido:**
- `dense`: 384 dimensões (all-MiniLM-L6-v2)
- `sparse`: BM25 vectors
- `colbert`: 128 dimensões com MultiVectorConfig

### Etapa 3: Ingestão de Dados
```bash
python projeto/ingestion.py
```
**Processa:**
- 10-K filing (relatório anual)
- 10-Q filing (relatório trimestral)
- Extrai itens relevantes (Risk Factors, MD&A, etc)
- Chunking semântico com HDBSCAN
- Gera embeddings híbridos
- Upload para Qdrant

### Etapa 4: Query com Busca Híbrida
```bash
python projeto/test-query.py
```
**Demonstra:**
- Dense similarity search
- Sparse BM25 search
- ColBERT late interaction
- RRF Fusion combination

### Etapa 5: API FastAPI
```bash
cd projeto/app
uvicorn main:app --reload --port 8001
```
**Acesse:**
- API Docs: http://127.0.0.1:8001/docs
- ReDoc: http://127.0.0.1:8001/redoc
- Base URL: http://127.0.0.1:8001

**Exemplo cURL:**
```bash
curl -X POST http://127.0.0.1:8001/events/ \
  -H "Content-Type: application/json" \
  -d '{"event_id":"123","event_type":"analysis","event_data":{"ticker":"AAPL"}}'
```

**Arquitetura:**
```
main.py (inicializa FastAPI)
   ↓
app = FastAPI()
app.include_router(process_router)
   ↓
router.py (orquestra rotas)
   ↓
endpoint.py (implementa endpoints)
   ↓
Lógica de negócio
```

---

## 🎯 Próximos Passos Recomendados

### Para Iniciantes
1. ✅ Executar `fundamentos/tokenization-01.py`
2. ✅ Explorar `llm/llm-01.py`
3. ✅ Testar `agents/exemplo-1-output-estruturado.py`
4. ✅ Entender `docling/1-extration.py`

### Para Intermediários
1. ✅ Combinar embeddings com RAG
2. ✅ Implementar workflows com agentes
3. ✅ Usar memória persistente (Mem0)
4. ✅ Criar APIs com FastAPI

### Para Avançados
1. ✅ Implementar busca híbrida completa
2. ✅ Processar SEC filings em escala
3. ✅ Fine-tuning de modelos
4. ✅ Otimizar performance de queries

---

## 📝 Licença

Este projeto é parte de um curso de especialização em IA. Verifique a licença específica do curso antes de usar em produção.

## 👤 Autor

Desenvolvido durante o curso de **Especialização em IA Dev Eficiente**.

## 📚 Referências e Documentação

### Bibliotecas Principais
- [Docling (IBM)](https://ds4sd.github.io/docling/) - Extração inteligente de PDFs
- [Qdrant](https://qdrant.tech/documentation/) - Vector database
- [FastEmbed](https://github.com/qdrant/fastembed) - Dense, Sparse, ColBERT
- [EdgarTools](https://github.com/dgunning/edgartools) - SEC EDGAR
- [Groq API](https://groq.com/) - LLM Rápido
- [OpenAI API](https://platform.openai.com/) - GPT Models
- [NLTK](https://www.nltk.org/) - NLP
- [Sentence Transformers](https://www.sbert.net/) - Embeddings
- [HDBSCAN](https://hdbscan.readthedocs.io/) - Clustering
- [FastAPI](https://fastapi.tiangolo.com/) - Web Framework
- [Pydantic](https://docs.pydantic.dev/) - Data Validation

### Conceitos de IA
- [RAG (Retrieval-Augmented Generation)](https://arxiv.org/abs/2005.11401)
- [Embeddings e Vector Search](https://en.wikipedia.org/wiki/Word_embedding)
- [Transformers](https://arxiv.org/abs/1706.03762)
- [LLMs](https://arxiv.org/abs/1910.14324)

---

**Última atualização**: Janeiro de 2026  
**Python**: 3.12+  
**Status**: Ativo ✅
