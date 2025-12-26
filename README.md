# 🤖 Curso IA - Especialização IA Dev Eficiente

Um projeto de especialização em IA focado no desenvolvimento eficiente com IA, incluindo fundamentos de NLP, extração de documentos, embeddings e sistemas RAG (Retrieval-Augmented Generation).

## 📋 Estrutura do Projeto

```
curso-ia/
├── fundamentos/          # Conceitos fundamentais de NLP
│   ├── tokenization-01.py   # Tokenização básica com NLTK
│   ├── tokenization-02.py   # Tokenização avançada
│   ├── tokenization-03.py   # Análise de frequência de tokens
│   └── tokenization-04.py   # BM25 para busca por ranking
│
├── docling/             # Extração e processamento de documentos PDF
│   ├── 1-extration.py      # Extração básica de documentos
│   ├── 2-extraction-images.py  # Extração com imagens
│   ├── 3-chunking.py       # Divisão em chunks
│   ├── 4-hybrid-chunker.py # Chunking híbrido
│   ├── 5-metadados.py      # Extração de metadados
│   ├── 6-embeddings.py     # Geração de embeddings
│   └── 2408.09869v5.pdf    # Documento de exemplo
│
├── llm/                 # Integração com Large Language Models
│   ├── llm-01.py        # Utilização da API Groq
│   └── llm-02.py        # Interações avançadas com LLMs
│
├── rag/                 # Sistema de Retrieval-Augmented Generation
│   ├── rag.py           # RAG com busca vetorial
│   └── rag-qdrant.py    # RAG usando Qdrant
│
├── pyproject.toml       # Dependências do projeto
└── README.md            # Este arquivo
```

## 🚀 Tecnologias Utilizadas

- **Processamento de Texto**: NLTK, Whoosh, BM25
- **Embeddings**: Sentence Transformers, FastEmbed
- **Extração de Documentos**: Docling (IBM)
- **LLMs**: Groq, OpenAI
- **Vector Database**: Qdrant
- **Machine Learning**: Scikit-learn
- **Parsing Estruturado**: Pydantic
- **Linguagem**: Python 3.12+

## 📦 Dependências

O projeto utiliza as seguintes bibliotecas principais:

```
docling>=2.65.0                  # Extração de documentos
fastembed>=0.7.4                 # Embeddings rápidos
groq>=1.0.0                      # API Groq
langextract>=1.1.1               # Extração de linguagem
nltk>=3.9.2                      # NLP
openai>=2.6.1                    # API OpenAI
qdrant-client>=1.16.2            # Vector database
rank-bm25>=0.2.2                 # Algoritmo BM25
sentence-transformers>=5.2.0     # Sentence embeddings
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

2. **Crie um ambiente virtual (opcional)**
```bash
python -m venv .venv
source .venv/bin/activate  # No Windows: .venv\Scripts\activate
```

3. **Instale as dependências**
```bash
# Usando pip
pip install -e .

# Ou usando uv (mais rápido)
uv install
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

## 🔐 Configuração de Chaves de API

O projeto utiliza variáveis de ambiente para gerenciar as chaves de API de forma segura. Crie um arquivo `.env` na raiz do projeto:

```bash
GROQ_API_KEY="sua_chave_aqui"
OPENAI_API_KEY="sua_chave_aqui"
GOOGLE_API_KEY="sua_chave_aqui"
```

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

query = "exemplo documento"
scores = bm25.get_scores(query.split())
```

### Extração de Documentos PDF
```python
from docling.document_converter import DocumentConverter

converter = DocumentConverter()
result = converter.convert("documento.pdf")
markdown = result.document.export_to_markdown()
print(markdown)
```

### RAG com Qdrant
```python
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

client = QdrantClient(":memory:")
model = SentenceTransformer("all-MiniLM-L6-v2")

# Embeddings e armazenamento
embeddings = model.encode(["documento 1", "documento 2"])
# ... armazenar em Qdrant
```

## 📝 Licença

Este projeto é parte de um curso de especialização em IA. Verifique a licença específica do curso antes de usar em produção.

## 👤 Autor

Desenvolvido durante o curso de Especialização em IA Dev Eficiente.

## 📞 Suporte

Para dúvidas e suporte, verifique a documentação das bibliotecas utilizadas:
- [Docling Documentation](https://ds4sd.github.io/docling/)
- [Qdrant Documentation](https://qdrant.tech/documentation/)
- [Sentence Transformers](https://www.sbert.net/)
- [NLTK Book](https://www.nltk.org/book/)

---

**Última atualização**: Dezembro de 2025
