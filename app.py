"""
app.py
======
Multimodal RAG Financial Analyst Copilot — Streamlit UI

Design Philosophy:
  Bloomberg Terminal meets modern GenAI product. Dark slate palette with
  amber accent — echoing professional financial data terminals — while the
  layout prioritises information density and citation transparency over
  decorative chrome.

  Every response surfaces: synthesised narrative + citation cards with
  confidence scores + rendered Markdown tables + extracted chart images.
  This "show your work" approach builds analyst trust and is a genuine
  differentiator vs. vanilla chatbots.

Run:
  streamlit run app.py
"""

import base64
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Optional

import streamlit as st
from openai import OpenAI

# Internal modules
from ingestion import ingest_document
from reranker import ContextAssembler, RerankerFactory
from vector_store import HybridVectorStore

# Logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


# Page config — must be the FIRST streamlit call

st.set_page_config(
    page_title="FinRAG Copilot",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# CSS — Bloomberg Terminal aesthetic: slate + amber + mono data typography

CUSTOM_CSS = """
<style>
/* ── Imports ─────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

/* ── Token System ────────────────────────────────────────── */
:root {
  --bg-base:       #0d1117;
  --bg-surface:    #161b22;
  --bg-elevated:   #1c2230;
  --bg-card:       #1e2735;
  --border:        #2a3445;
  --border-active: #f0a500;
  --amber:         #f0a500;
  --amber-dim:     #c4880080;
  --amber-glow:    #f0a50018;
  --green:         #39d353;
  --red:           #e84040;
  --blue:          #58a6ff;
  --text-primary:  #e6edf3;
  --text-secondary:#8b949e;
  --text-muted:    #484f58;
  --mono:          'IBM Plex Mono', monospace;
  --sans:          'IBM Plex Sans', sans-serif;
}

/* ── Global Reset ────────────────────────────────────────── */
html, body, [class*="css"], .stApp {
  background-color: var(--bg-base) !important;
  color: var(--text-primary) !important;
  font-family: var(--sans) !important;
}

/* ── Hide Streamlit Branding ─────────────────────────────── */
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }

/* ── Sidebar ─────────────────────────────────────────────── */
section[data-testid="stSidebar"] {
  background-color: var(--bg-surface) !important;
  border-right: 1px solid var(--border) !important;
}
section[data-testid="stSidebar"] * {
  color: var(--text-primary) !important;
}

/* ── Header Banner ───────────────────────────────────────── */
.finrag-header {
  background: linear-gradient(135deg, var(--bg-elevated) 0%, #0d1a2e 100%);
  border: 1px solid var(--border);
  border-left: 3px solid var(--amber);
  border-radius: 6px;
  padding: 20px 28px;
  margin-bottom: 20px;
  display: flex;
  align-items: center;
  justify-content: space-between;
}
.finrag-title {
  font-family: var(--mono);
  font-size: 22px;
  font-weight: 600;
  color: var(--amber);
  letter-spacing: 0.04em;
  margin: 0;
}
.finrag-subtitle {
  font-size: 12px;
  color: var(--text-secondary);
  font-family: var(--mono);
  margin: 4px 0 0;
  letter-spacing: 0.06em;
}
.status-pill {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  background: var(--amber-glow);
  border: 1px solid var(--amber-dim);
  border-radius: 20px;
  padding: 4px 12px;
  font-family: var(--mono);
  font-size: 11px;
  color: var(--amber);
}
.status-dot {
  width: 7px; height: 7px;
  background: var(--green);
  border-radius: 50%;
  animation: pulse 2s infinite;
}
@keyframes pulse {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.3; }
}

/* ── Chat Messages ───────────────────────────────────────── */
.chat-bubble-user {
  background: var(--bg-elevated);
  border: 1px solid var(--border);
  border-right: 3px solid var(--blue);
  border-radius: 8px 8px 2px 8px;
  padding: 14px 18px;
  margin: 12px 0;
  font-size: 14px;
  line-height: 1.6;
}
.chat-bubble-assistant {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-left: 3px solid var(--amber);
  border-radius: 8px 8px 8px 2px;
  padding: 16px 20px;
  margin: 12px 0;
  font-size: 14px;
  line-height: 1.7;
}
.chat-role-label {
  font-family: var(--mono);
  font-size: 10px;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: var(--text-muted);
  margin-bottom: 6px;
}
.chat-role-label.user { color: var(--blue); }
.chat-role-label.assistant { color: var(--amber); }

/* ── Citation Cards ──────────────────────────────────────── */
.citation-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
  gap: 12px;
  margin-top: 16px;
}
.citation-card {
  background: var(--bg-surface);
  border: 1px solid var(--border);
  border-top: 2px solid var(--amber-dim);
  border-radius: 6px;
  padding: 14px;
  transition: border-color 0.2s, background 0.2s;
}
.citation-card:hover {
  border-color: var(--amber);
  background: var(--bg-elevated);
}
.citation-header {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  margin-bottom: 8px;
}
.citation-filename {
  font-family: var(--mono);
  font-size: 11px;
  color: var(--amber);
  font-weight: 600;
}
.citation-meta {
  font-family: var(--mono);
  font-size: 10px;
  color: var(--text-muted);
  margin-top: 2px;
}
.citation-section {
  font-size: 11px;
  color: var(--text-secondary);
  margin: 4px 0 8px;
  font-style: italic;
}
.citation-preview {
  font-size: 12px;
  color: var(--text-secondary);
  line-height: 1.5;
  border-top: 1px solid var(--border);
  padding-top: 8px;
  margin-top: 6px;
}
.confidence-badge {
  display: inline-block;
  font-family: var(--mono);
  font-size: 10px;
  font-weight: 600;
  padding: 2px 8px;
  border-radius: 3px;
  letter-spacing: 0.08em;
}
.confidence-high   { background: #1a3a1a; color: var(--green); border: 1px solid #39d35340; }
.confidence-medium { background: #2a2a0a; color: #f0d060;      border: 1px solid #f0d06040; }
.confidence-low    { background: #2a1a1a; color: var(--red);   border: 1px solid #e8404040; }
.chunk-type-badge {
  font-family: var(--mono);
  font-size: 10px;
  color: var(--text-muted);
  background: var(--bg-base);
  border: 1px solid var(--border);
  border-radius: 3px;
  padding: 1px 6px;
}

/* ── Financial Tables ────────────────────────────────────── */
.financial-table-container {
  background: var(--bg-base);
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: 16px;
  margin: 12px 0;
  overflow-x: auto;
}
.financial-table-label {
  font-family: var(--mono);
  font-size: 10px;
  color: var(--amber);
  letter-spacing: 0.1em;
  text-transform: uppercase;
  margin-bottom: 10px;
}
table.financial-table {
  width: 100%;
  border-collapse: collapse;
  font-family: var(--mono);
  font-size: 12px;
}
table.financial-table th {
  background: var(--bg-elevated);
  color: var(--text-secondary);
  font-weight: 600;
  padding: 8px 12px;
  border-bottom: 2px solid var(--amber-dim);
  text-align: left;
  letter-spacing: 0.04em;
  white-space: nowrap;
}
table.financial-table td {
  padding: 7px 12px;
  border-bottom: 1px solid var(--border);
  color: var(--text-primary);
}
table.financial-table td.num-cell {
  text-align: right;
  font-variant-numeric: tabular-nums;
  color: var(--text-primary);
}
table.financial-table tr:hover td {
  background: var(--bg-elevated);
}
table.financial-table tr:last-child td {
  border-bottom: none;
}

/* ── Chart Image ─────────────────────────────────────────── */
.chart-container {
  background: var(--bg-base);
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: 12px;
  margin: 12px 0;
}
.chart-label {
  font-family: var(--mono);
  font-size: 10px;
  color: var(--amber);
  letter-spacing: 0.1em;
  text-transform: uppercase;
  margin-bottom: 8px;
}
.chart-container img {
  max-width: 100%;
  border-radius: 4px;
  border: 1px solid var(--border);
}

/* ── Streamlit Input Overrides ───────────────────────────── */
.stTextInput > div > div > input,
.stTextArea > div > div > textarea {
  background-color: var(--bg-surface) !important;
  border: 1px solid var(--border) !important;
  border-radius: 6px !important;
  color: var(--text-primary) !important;
  font-family: var(--sans) !important;
}
.stTextInput > div > div > input:focus,
.stTextArea > div > div > textarea:focus {
  border-color: var(--amber) !important;
  box-shadow: 0 0 0 2px var(--amber-glow) !important;
}

/* ── Buttons ─────────────────────────────────────────────── */
.stButton > button {
  background: var(--amber) !important;
  color: #0d1117 !important;
  border: none !important;
  font-family: var(--mono) !important;
  font-weight: 600 !important;
  font-size: 13px !important;
  letter-spacing: 0.05em !important;
  border-radius: 5px !important;
  padding: 8px 20px !important;
  transition: opacity 0.2s !important;
}
.stButton > button:hover { opacity: 0.85 !important; }

.stButton.secondary > button {
  background: transparent !important;
  color: var(--text-secondary) !important;
  border: 1px solid var(--border) !important;
}

/* ── File Uploader ───────────────────────────────────────── */
.stFileUploader {
  background: var(--bg-surface) !important;
  border: 1px dashed var(--border) !important;
  border-radius: 6px !important;
}

/* ── Metric Tiles ────────────────────────────────────────── */
.metric-tile {
  background: var(--bg-surface);
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: 16px;
  text-align: center;
}
.metric-value {
  font-family: var(--mono);
  font-size: 26px;
  font-weight: 600;
  color: var(--amber);
  line-height: 1;
}
.metric-label {
  font-size: 11px;
  color: var(--text-muted);
  margin-top: 4px;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  font-family: var(--mono);
}

/* ── Spinner / Progress ──────────────────────────────────── */
.stSpinner > div { border-top-color: var(--amber) !important; }
div[data-testid="stProgressBar"] > div { background-color: var(--amber) !important; }

/* ── Expanders ───────────────────────────────────────────── */
details > summary {
  font-family: var(--mono);
  font-size: 12px;
  color: var(--text-secondary);
  cursor: pointer;
}
details { border: 1px solid var(--border); border-radius: 5px; padding: 8px 12px; margin: 8px 0; }

/* ── Dividers ────────────────────────────────────────────── */
hr { border-color: var(--border) !important; }

/* ── Scrollbar ───────────────────────────────────────────── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--bg-base); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--text-muted); }
</style>
"""


# LLM Generation Prompt

SYSTEM_PROMPT = """You are FinRAG Copilot, an elite financial analyst AI embedded in an 
institutional investment bank's research platform. You have deep expertise in:
- Equity research, fundamental analysis, and financial statement interpretation
- Earnings call analysis, 10-K/10-Q filing decomposition, and M&A pitchbook review
- Quantitative ratio analysis (leverage, profitability, liquidity, efficiency metrics)
- Trend identification, YoY/QoQ comparisons, and variance attribution

You will be provided with retrieved context from financial documents, tagged as 
<source> blocks with metadata. Your response must:

1. Synthesise a precise, data-driven answer citing specific figures from the sources.
2. Reference sources explicitly using [Source: filename, p.N] notation.
3. Identify trends, risks, and investment implications where relevant.
4. Flag any data gaps or contradictions between sources.
5. Use professional financial language appropriate for a senior analyst audience.
6. Structure longer responses with clear sections (key findings, analysis, caveats).

If a query cannot be answered from the provided sources, state this clearly and 
explain what additional data would be needed. Never fabricate financial data."""


# Session State Initialisation


def init_session_state() -> None:
    defaults: dict[str, Any] = {
        "messages": [],
        "vector_store": None,
        "reranker": None,
        "openai_client": None,
        "indexed_docs": [],
        "total_chunks": 0,
        "api_keys_set": False,
        "processing": False,
        "selected_doc_filter": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# Backend Initialisation

@st.cache_resource(show_spinner=False)
def get_vector_store(openai_api_key: str) -> HybridVectorStore:
    return HybridVectorStore(
        openai_api_key=openai_api_key,
        persist_directory=".chroma_db",
        collection_name="finrag_docs",
    )


@st.cache_resource(show_spinner=False)
def get_reranker(
    openai_api_key: str,
    cohere_api_key: Optional[str],
) -> Any:
    client = OpenAI(api_key=openai_api_key)
    return RerankerFactory.create(
        openai_client=client,
        cohere_api_key=cohere_api_key or None,
    )


# Document Ingestion Handler


def handle_document_upload(
    uploaded_files: list[Any],
    openai_api_key: str,
    vector_store: HybridVectorStore,
    enable_vision: bool,
) -> None:
    """
    Orchestrates ingestion → embedding → indexing for one or more uploaded PDFs.
    Provides granular progress feedback via Streamlit progress bars.
    """
    import tempfile

    total = len(uploaded_files)
    progress = st.progress(0, text="Initialising ingestion pipeline...")
    status_placeholder = st.empty()

    for i, uploaded_file in enumerate(uploaded_files):
        filename = uploaded_file.name
        status_placeholder.markdown(
            f'<div class="finrag-subtitle">📄 Processing: <strong>{filename}</strong> ({i+1}/{total})</div>',
            unsafe_allow_html=True,
        )
        progress.progress(int((i / total) * 50), text=f"Extracting layout: {filename}")

        # Write to temp file (Streamlit UploadedFile → disk path)
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        try:
            chunks = ingest_document(
                pdf_path=tmp_path,
                openai_api_key=openai_api_key,
                enable_vision=enable_vision,
                use_cache=True,
            )
            # Override source_file with original filename (temp path is noise)
            for chunk in chunks:
                chunk.source_file = filename

            progress.progress(
                int((i / total) * 50 + 30),
                text=f"Embedding {len(chunks)} chunks: {filename}",
            )
            vector_store.index(chunks)
            progress.progress(
                int((i + 1) / total * 100),
                text=f"Indexed {filename} ✓",
            )
            logger.info("Indexed %d chunks from %s", len(chunks), filename)

        except Exception as exc:
            st.error(f"❌ Failed to process {filename}: {exc}")
            logger.error("Ingestion failed for %s: %s", filename, exc, exc_info=True)
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    progress.empty()
    status_placeholder.empty()

    # Refresh indexed docs list
    st.session_state.indexed_docs = vector_store.list_indexed_documents()
    st.session_state.total_chunks = sum(
        d["chunk_count"] for d in st.session_state.indexed_docs
    )
    st.success(f"✅ Successfully indexed {total} document(s) into the knowledge base.")


# RAG Query Pipeline


def run_rag_query(
    query: str,
    vector_store: HybridVectorStore,
    reranker: Any,
    openai_client: OpenAI,
    source_filter: Optional[str],
    top_k_retrieval: int = 10,
    top_n_rerank: int = 4,
) -> tuple[str, list[dict[str, Any]]]:
    """
    Full RAG pipeline: retrieve → rerank → assemble context → generate.

    Returns
    -------
    answer     : LLM-generated financial analysis string.
    citations  : List of citation dicts for UI rendering.
    """
    # Step 1: Hybrid retrieval
    candidates = vector_store.search(
        query=query,
        top_k=top_k_retrieval,
        source_filter=source_filter if source_filter != "All Documents" else None,
    )

    if not candidates:
        return (
            "⚠️ No relevant content found in the knowledge base for this query. "
            "Please upload relevant financial documents first.",
            [],
        )

    # Step 2: Reranking
    reranked = reranker.rerank(query=query, candidates=candidates, top_n=top_n_rerank)

    # Step 3: Context assembly
    assembler = ContextAssembler()
    context_str, citations = assembler.assemble(query, reranked)

    # Step 4: LLM generation
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"{context_str}\n\n"
                f"---\n"
                f"Analyst Query: {query}\n\n"
                "Provide a comprehensive financial analysis based on the retrieved sources above."
            ),
        },
    ]

    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            max_tokens=1500,
            temperature=0.1,
            stream=False,
        )
        answer = response.choices[0].message.content.strip()
    except Exception as exc:
        logger.error("LLM generation failed: %s", exc)
        answer = f"⚠️ Generation failed: {exc}"

    return answer, citations


# UI Components


def render_header() -> None:
    st.markdown(
        """
        <div class="finrag-header">
          <div>
            <div class="finrag-title">▐ FINRAG COPILOT</div>
            <div class="finrag-subtitle">MULTIMODAL RAG · FINANCIAL DOCUMENT INTELLIGENCE · v2.1</div>
          </div>
          <div class="status-pill">
            <span class="status-dot"></span>
            SYSTEM ONLINE
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metrics_bar(indexed_docs: list[dict], total_chunks: int) -> None:
    cols = st.columns(4)
    metrics = [
        ("DOCUMENTS", str(len(indexed_docs))),
        ("CHUNKS INDEXED", f"{total_chunks:,}"),
        ("RETRIEVAL MODEL", "text-emb-3-small"),
        ("RERANKER", "Cohere / CE"),
    ]
    for col, (label, value) in zip(cols, metrics):
        with col:
            st.markdown(
                f'<div class="metric-tile">'
                f'<div class="metric-value">{value}</div>'
                f'<div class="metric-label">{label}</div>'
                f"</div>",
                unsafe_allow_html=True,
            )


def render_citation_card(citation: dict[str, Any], index: int) -> None:
    """Renders a single citation card with confidence badge and metadata."""
    confidence = citation.get("confidence_label", "Medium")
    confidence_class = f"confidence-{confidence.lower()}"
    chunk_type = citation.get("chunk_type", "text").upper()
    score = citation.get("reranker_score", 0.0)

    st.markdown(
        f"""
        <div class="citation-card">
          <div class="citation-header">
            <div>
              <div class="citation-filename">📄 {citation['source_file']}</div>
              <div class="citation-meta">Page {citation['page_number']} · 
                <span class="chunk-type-badge">{chunk_type}</span>
              </div>
            </div>
            <div>
              <span class="confidence-badge {confidence_class}">{confidence} · {score:.2f}</span>
            </div>
          </div>
          <div class="citation-section">§ {citation.get('section_header', 'Unknown Section') or 'Unknown Section'}</div>
          <div class="citation-preview">{citation.get('content_preview', '')}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Render embedded table if present
    if citation.get("table_html"):
        with st.expander(f"📊 View Extracted Table — {citation['source_file']} p.{citation['page_number']}"):
            st.markdown(
                f'<div class="financial-table-container">'
                f'<div class="financial-table-label">▸ Extracted Financial Table</div>'
                f'{citation["table_html"]}'
                f"</div>",
                unsafe_allow_html=True,
            )

    # Render chart image if present
    if citation.get("image_b64"):
        with st.expander(f"📈 View Source Chart — {citation['source_file']} p.{citation['page_number']}"):
            st.markdown(
                '<div class="chart-container"><div class="chart-label">▸ Extracted Chart / Graphic</div>',
                unsafe_allow_html=True,
            )
            st.image(
                base64.b64decode(citation["image_b64"]),
                use_column_width=True,
                caption=f"{citation['source_file']} — Page {citation['page_number']}",
            )
            st.markdown("</div>", unsafe_allow_html=True)


def render_chat_message(role: str, content: str, citations: Optional[list] = None) -> None:
    """Renders a chat bubble with optional citation grid below."""
    bubble_class = "chat-bubble-user" if role == "user" else "chat-bubble-assistant"
    role_label = "YOU" if role == "user" else "FINRAG COPILOT"
    label_class = "user" if role == "user" else "assistant"

    st.markdown(
        f'<div class="{bubble_class}">'
        f'<div class="chat-role-label {label_class}">{role_label}</div>'
        f"{content}"
        f"</div>",
        unsafe_allow_html=True,
    )

    # Citation section
    if citations and role == "assistant":
        st.markdown(
            '<div style="margin-top:4px; margin-bottom:2px;">'
            '<span style="font-family:var(--mono); font-size:10px; '
            'color:var(--text-muted); letter-spacing:0.1em;">'
            f"▸ EVIDENCE SOURCES ({len(citations)})</span></div>",
            unsafe_allow_html=True,
        )
        cols = st.columns(min(len(citations), 2))
        for i, citation in enumerate(citations):
            with cols[i % 2]:
                render_citation_card(citation, i)


def render_sample_queries() -> Optional[str]:
    """Quick-access panel of sample financial analyst queries."""
    samples = [
        "Analyze the debt-to-equity ratio trend over the last three quarters",
        "What were the primary drivers of revenue growth in the most recent quarter?",
        "Summarize the non-GAAP operating margin reconciliation",
        "Identify any going-concern risks or liquidity warnings in these filings",
        "Compare EBITDA margins across business segments",
        "What guidance did management provide for the next fiscal year?",
    ]
    st.markdown(
        '<div style="font-family:var(--mono); font-size:10px; color:var(--text-muted); '
        'letter-spacing:0.1em; margin-bottom:8px;">▸ SAMPLE QUERIES</div>',
        unsafe_allow_html=True,
    )
    for sample in samples:
        if st.button(sample, key=f"sample_{hash(sample)}", use_container_width=True):
            return sample
    return None


# Sidebar


def render_sidebar() -> tuple[Optional[str], Optional[str], bool, int, int, Optional[str]]:
    """
    Returns: (openai_key, cohere_key, enable_vision, top_k, top_n, source_filter)
    """
    with st.sidebar:
        st.markdown(
            '<div style="font-family:var(--mono); font-size:13px; color:var(--amber); '
            'font-weight:600; letter-spacing:0.08em; padding: 8px 0 16px;">⚙ CONFIGURATION</div>',
            unsafe_allow_html=True,
        )

        # ── API Keys 
        st.markdown(
            '<div style="font-family:var(--mono); font-size:10px; color:var(--text-muted); '
            'letter-spacing:0.1em; margin-bottom:6px;">▸ API CREDENTIALS</div>',
            unsafe_allow_html=True,
        )
        openai_key = st.text_input(
            "OpenAI API Key",
            type="password",
            value=os.getenv("OPENAI_API_KEY", ""),
            placeholder="sk-...",
            help="Required for embeddings and generation (GPT-4o).",
        )
        cohere_key = st.text_input(
            "Cohere API Key (optional)",
            type="password",
            value=os.getenv("COHERE_API_KEY", ""),
            placeholder="...",
            help="Enables Cohere reranking. Falls back to CrossEncoder if absent.",
        )

        st.divider()

        # ── Document Upload 
        st.markdown(
            '<div style="font-family:var(--mono); font-size:10px; color:var(--text-muted); '
            'letter-spacing:0.1em; margin-bottom:6px;">▸ DOCUMENT INGESTION</div>',
            unsafe_allow_html=True,
        )
        uploaded_files = st.file_uploader(
            "Upload Financial PDFs",
            type=["pdf"],
            accept_multiple_files=True,
            label_visibility="collapsed",
        )
        enable_vision = st.checkbox(
            "Enable Chart Vision Analysis",
            value=True,
            help="Uses GPT-4o Vision to extract chart descriptions. Recommended but costs more.",
        )

        if uploaded_files and openai_key:
            if st.button("▸ INGEST DOCUMENTS", use_container_width=True):
                vs = get_vector_store(openai_key)
                with st.spinner("Processing documents..."):
                    handle_document_upload(uploaded_files, openai_key, vs, enable_vision)
                st.rerun()

        st.divider()

        # ── Retrieval Settings 
        st.markdown(
            '<div style="font-family:var(--mono); font-size:10px; color:var(--text-muted); '
            'letter-spacing:0.1em; margin-bottom:6px;">▸ RETRIEVAL PARAMETERS</div>',
            unsafe_allow_html=True,
        )
        top_k = st.slider(
            "Hybrid Retrieval top-K",
            min_value=5, max_value=20, value=10,
            help="Number of candidates from dense+sparse retrieval before reranking.",
        )
        top_n = st.slider(
            "Reranker top-N",
            min_value=2, max_value=6, value=4,
            help="Final context windows passed to the LLM after cross-encoder reranking.",
        )

        # ── Source Filter 
        if openai_key:
            try:
                vs = get_vector_store(openai_key)
                docs = vs.list_indexed_documents()
            except Exception:
                docs = []
        else:
            docs = []

        source_options = ["All Documents"] + [d["source_file"] for d in docs]
        source_filter = st.selectbox(
            "Filter by Document",
            options=source_options,
            help="Restrict retrieval to a specific uploaded document.",
        )

        st.divider()

        # ── Indexed Documents Summary 
        if docs:
            st.markdown(
                '<div style="font-family:var(--mono); font-size:10px; color:var(--text-muted); '
                'letter-spacing:0.1em; margin-bottom:8px;">▸ KNOWLEDGE BASE</div>',
                unsafe_allow_html=True,
            )
            for doc in docs:
                st.markdown(
                    f'<div style="font-family:var(--mono); font-size:11px; '
                    f'color:var(--text-secondary); padding:4px 0; '
                    f'border-bottom:1px solid var(--border);">'
                    f'📄 {doc["source_file"]}<br>'
                    f'<span style="color:var(--text-muted); font-size:10px;">'
                    f'{doc["chunk_count"]} chunks · {doc["page_count"]} pages</span>'
                    f"</div>",
                    unsafe_allow_html=True,
                )

        # ── Clear Chat 
        st.divider()
        if st.button("🗑 Clear Conversation", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    return openai_key, cohere_key, enable_vision, top_k, top_n, source_filter



# Main Application


def main() -> None:
    init_session_state()

    # Inject CSS
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    # Render sidebar and get config
    openai_key, cohere_key, enable_vision, top_k, top_n, source_filter = render_sidebar()

    # Main content area
    render_header()

    # Guard: require API key
    if not openai_key:
        st.markdown(
            '<div class="chat-bubble-assistant" style="text-align:center; padding:40px;">'
            '<div style="font-family:var(--mono); font-size:32px; color:var(--amber);">📊</div>'
            "<br><strong>Welcome to FinRAG Copilot</strong><br><br>"
            '<span style="color:var(--text-secondary); font-size:13px;">'
            "Enter your OpenAI API key in the sidebar to begin.<br>"
            "Upload financial PDFs (10-Ks, earnings reports, pitchbooks) to build your knowledge base."
            "</span>"
            "</div>",
            unsafe_allow_html=True,
        )
        return

    # Initialise backend services
    try:
        vector_store = get_vector_store(openai_key)
        openai_client = OpenAI(api_key=openai_key)
        reranker = get_reranker(openai_key, cohere_key)
    except Exception as exc:
        st.error(f"Failed to initialise backend services: {exc}")
        logger.error("Backend init failed: %s", exc, exc_info=True)
        return

    # Refresh metrics
    indexed_docs = vector_store.list_indexed_documents()
    total_chunks = sum(d["chunk_count"] for d in indexed_docs)
    render_metrics_bar(indexed_docs, total_chunks)

    st.divider()

    # ── Chat Column + Sample Queries 
    chat_col, query_col = st.columns([3, 1])

    with query_col:
        st.markdown(
            '<div style="font-family:var(--mono); font-size:10px; color:var(--text-muted); '
            'letter-spacing:0.1em; padding-bottom:8px;">▸ QUICK QUERIES</div>',
            unsafe_allow_html=True,
        )
        clicked_sample = render_sample_queries()

    with chat_col:
        # Render conversation history
        chat_container = st.container()
        with chat_container:
            for message in st.session_state.messages:
                render_chat_message(
                    role=message["role"],
                    content=message["content"],
                    citations=message.get("citations"),
                )

        # ── Query Input 
        st.markdown("<br>", unsafe_allow_html=True)
        query_input = st.text_area(
            "Analyst Query",
            value=clicked_sample or "",
            placeholder="e.g. Analyze the debt-to-equity shift over the last three quarters...",
            height=80,
            label_visibility="collapsed",
            key="query_input",
        )

        input_cols = st.columns([4, 1])
        with input_cols[1]:
            submit = st.button("▸ ANALYSE", use_container_width=True)

        if submit and query_input.strip():
            if not indexed_docs:
                st.warning("⚠️ No documents indexed. Please upload financial PDFs first.")
            else:
                # Add user message
                st.session_state.messages.append({
                    "role": "user",
                    "content": query_input.strip(),
                })

                # Run RAG pipeline with live spinner
                with st.spinner("🔍 Retrieving · Reranking · Synthesising..."):
                    start_ts = time.perf_counter()
                    try:
                        answer, citations = run_rag_query(
                            query=query_input.strip(),
                            vector_store=vector_store,
                            reranker=reranker,
                            openai_client=openai_client,
                            source_filter=source_filter,
                            top_k_retrieval=top_k,
                            top_n_rerank=top_n,
                        )
                        elapsed = time.perf_counter() - start_ts
                        logger.info("RAG pipeline completed in %.2fs", elapsed)

                        # Append timing footnote
                        answer_with_meta = (
                            answer
                            + f'\n\n<span style="font-family:var(--mono); font-size:10px; '
                            f'color:var(--text-muted);">⏱ Response generated in {elapsed:.1f}s · '
                            f"{len(citations)} sources retrieved</span>"
                        )

                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": answer_with_meta,
                            "citations": citations,
                        })
                    except Exception as exc:
                        logger.error("RAG query failed: %s", exc, exc_info=True)
                        st.error(f"Query failed: {exc}")

                st.rerun()


if __name__ == "__main__":
    main()
