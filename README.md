# FinRAG Copilot — Multimodal Financial Retrieval Engine

> An advanced, architecture-focused RAG terminal designed for institutional document intelligence. 
Built using enterprise design patterns, the codebase demonstrates layout-aware ingestion, local dual-vector retrieval, and a high-density analyst user interface.
Ingests 10-Ks, earnings reports, and pitchbooks. Answers with cited evidence, rendered tables, and source charts.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         FinRAG Copilot                              │
│                                                                     │
│  PDF Upload                                                         │
│      │                                                              │
│      ▼                                                              │
│  [ingestion.py]  Layout-Aware Multimodal Extraction                 │
│   ├── Text Blocks   → SectionBoundaryChunker                        │
│   ├── Tables        → Markdown + HTML (heuristic col detection)     │
│   └── Chart Images  → GPT-4o Vision → text surrogate               │
│      │                                                              │
│      ▼                                                              │
│  [vector_store.py]  Hybrid Dual-Vector Retrieval                    │
│   ├── Dense  → OpenAI text-embedding-3-small → ChromaDB (cosine)   │
│   ├── Sparse → Custom Okapi BM25                                    │
│   └── Fusion → Reciprocal Rank Fusion (RRF)                        │
│      │                                                              │
│      ▼                                                              │
│  [reranker.py]  Cross-Encoder Reranking                             │
│   ├── Backend A: Cohere Rerank API (rerank-english-v3.0)            │
│   ├── Backend B: HuggingFace CrossEncoder (local, free)             │
│   └── Backend C: LLM Reranker fallback (GPT-4o scoring)            │
│      │                                                              │
│      ▼                                                              │
│  [app.py]  Streamlit UI + GPT-4o Generation                         │
│   ├── Chat interface with Bloomberg Terminal aesthetic              │
│   ├── Cited financial narrative response                            │
│   ├── Citation cards (confidence score, section, page)              │
│   ├── Rendered Markdown/HTML financial tables                       │
│   └── Source chart image crops (base64 inline)                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 🖥️ User Interface Overview

The platform features a custom-styled, high-density dashboard mirroring a classic financial analyst terminal workspace.

> 💡 **Development Sandbox Note:** The UI displays a fully designed frontend execution layer. To prevent API rate limits and unexpected token fees during presentation, backend network components (OpenAI / Cohere) can run in a pre-configured simulation mode using offline mock records.

### 📊 1. Analytical Control Center & Workspace
3-column financial view: parameter controls on the left, interactive analytical workspace in the center, and context-dependent quick queries on the right.

<p align="center">
<img width="1900" height="905" alt="Dashboard png" src="https://github.com/user-attachments/assets/d2a4ae2d-82f8-4148-a941-1c656be5d64c" />
</p>

### 🔍 2. Granular Source Retrieval & Citation Audit Trace
Surfaces full transparency for metadata citations — exact chunk extraction headers, document page numbers, and reranker semantic alignment scores.

<p align="center">
<img width="1512" height="522" alt="retrieval-citations png" src="https://github.com/user-attachments/assets/e6e795d4-4819-4625-84d9-46590fc54ed2" />
</p>

### 📈 3. Multi-Modal Visual Asset Extractions
Maps text metrics directly alongside tabular balance sheets and internal corporate data charts.

<p align="center">
  <img src="https://github.com/user-attachments/assets/a5ab1d76-d1ca-464d-aba8-655a37ba8af4" width="49%" alt="Extracted Chart View" />
  <img src="https://github.com/user-attachments/assets/cd882b22-38b1-4b2c-a4e9-5a37ca805484" width="49%" alt="Structured Balance Sheet View" />
</p>
---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure API keys

```bash
# Option A: Environment variables
export OPENAI_API_KEY="sk-..."
export COHERE_API_KEY="..."   # Optional — enables Cohere reranker

# Option B: .env file
echo "OPENAI_API_KEY=sk-..." > .env
echo "COHERE_API_KEY=..."    >> .env
```

### 3. Launch

```bash
streamlit run app.py
```

---

## Module Reference

| Module | Responsibility | Key Classes |
|---|---|---|
| `ingestion.py` | PDF extraction, chunking, vision summarisation | `LayoutAwareExtractor`, `SectionBoundaryChunker`, `VisionSummarizer` |
| `vector_store.py` | Dual-vector indexing, hybrid retrieval, RRF | `HybridVectorStore`, `BM25Retriever`, `ChromaAdapter` |
| `reranker.py` | Cross-encoder reranking, context assembly | `CohereReranker`, `CrossEncoderReranker`, `ContextAssembler` |
| `app.py` | Streamlit UI, RAG orchestration, citation rendering | `main()`, `run_rag_query()`, `render_citation_card()` |

---

## Configuration

| Parameter | Default | Description |
|---|---|---|
| Retrieval top-K | `10` | Candidates per retriever before RRF fusion |
| Reranker top-N | `4` | Final context windows sent to LLM |
| Vision enabled | `True` | GPT-4o chart summarisation (~$0.01/chart) |
| Embedding model | `text-embedding-3-small` | 1536-dim, 8191-token context |
| Generation model | `gpt-4o` | 128k context, supports structured output |
| Reranker (auto) | Cohere → CrossEncoder → LLM | Priority based on available keys/packages |

---

## Production Upgrade Path

| Component | Current | Production Upgrade |
|---|---|---|
| Vector DB | ChromaDB (embedded) | Qdrant Cloud / Weaviate / Pinecone |
| Table extraction | Heuristic BBox | `camelot-py` with grid detection |
| Chunking | Font-size heuristic | `unstructured` with `hi_res` strategy |
| PDF parsing | PyMuPDF | LlamaParse (cloud, handles scanned PDFs) |
| BM25 index | In-memory | Elasticsearch / OpenSearch BM25 |
| Auth | None | Auth0 / Okta SSO |
| Caching | Filesystem JSON | Redis / S3 |

---

## Engineering Highlights

- **Pluggable reranker backends** via `RerankerFactory` — demonstrates design patterns awareness
- **Graceful degradation** — every API call has a fallback (Cohere → CrossEncoder → LLM reranker; vision → no vision)
- **RRF over linear combination** — demonstrates knowledge of information retrieval research
- **Section-boundary chunking** — domain expertise; not just "split at 512 tokens"
- **Ingestion cache** — filesystem-based SHA-256 cache; production cost-awareness
- **Custom BM25+ implementation** — demonstrates ML fundamentals, not just library calls
- **XML-tagged context assembly** — reduces hallucination via unambiguous source boundaries
- **Citation confidence labels** — derived from cross-encoder scores, surfaced in UI
- **Full type hints** — `mypy`-compatible, enterprise code standards
- **Structured logging** — production-grade `%(asctime)s | %(levelname)s | %(name)s` format
