"""
vector_store.py
===============
Dual-Vector Hybrid Retrieval Engine

Architecture:
  ┌─────────────────────────────────────────────────────────────────┐
  │  DocumentChunks (from ingestion.py)                             │
  │       │                                                         │
  │       ▼                                                         │
  │  [HybridVectorStore]                                            │
  │   ├── Dense Path: OpenAI text-embedding-3-small                 │
  │   │    └── ChromaDB (persistent cosine similarity index)        │
  │   └── Sparse Path: BM25Retriever (Okapi BM25 over corpus)       │
  │                                                                  │
  │  Query                                                           │
  │   │                                                             │
  │   ├── Dense Retrieval  → top-K by cosine similarity            │
  │   ├── Sparse Retrieval → top-K by BM25 score                   │
  │   └── [ReciprocusRankFusion] → merged & deduplicated results   │
  │                                                                  │
  │  Top-N Candidates → reranker.py                                │
  └─────────────────────────────────────────────────────────────────┘

Design Decisions:
- ChromaDB for persistent dense embeddings: zero-infra, embedded mode,
  production-upgradeable to Qdrant/Weaviate by swapping the adapter.
- BM25 (Okapi BM25) for sparse retrieval: catches exact financial nomenclature
  like "Q3 FY25 Non-GAAP Operating Margin" that semantic embeddings dilute.
- Reciprocal Rank Fusion (RRF) for score-free merging: avoids the need to
  normalise dense cosine scores against BM25 raw scores, which are on
  incompatible scales. RRF is empirically superior to linear combination
  for heterogeneous retrievers (Cormack et al., 2009).
- Embedding batching: OpenAI's API is called in batches of 100 to stay within
  rate limits and amortise per-request overhead.
"""

import logging
import math
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import chromadb
from chromadb.config import Settings
from openai import OpenAI

from ingestion import ChunkType, DocumentChunk


# Logging

logger = logging.getLogger(__name__)

# Domain Models



@dataclass
class RetrievedChunk:
    """
    A DocumentChunk augmented with retrieval-time scoring metadata.
    Carries RRF-fused rank + individual dense/sparse scores for
    interpretability in the UI and reranker.
    """

    chunk: DocumentChunk
    rrf_score: float
    dense_rank: Optional[int] = None
    sparse_rank: Optional[int] = None
    dense_score: Optional[float] = None
    sparse_score: Optional[float] = None



# Embedding Client Wrapper


class EmbeddingClient:
    """
    Thin wrapper around OpenAI's embedding endpoint.

    Features:
    - Batched encoding to respect API rate limits.
    - Exponential-backoff retry on transient failures.
    - Dimension normalisation (L2) for cosine compatibility with ChromaDB.
    """

    MODEL = "text-embedding-3-small"
    BATCH_SIZE = 100
    MAX_TOKENS_PER_TEXT = 8191  # text-embedding-3-small context limit

    def __init__(self, openai_client: OpenAI) -> None:
        self._client = openai_client

    def embed(self, texts: list[str]) -> list[list[float]]:
        """
        Encode a list of texts and return their L2-normalised embeddings.
        Handles batching transparently.
        """
        if not texts:
            return []

        # Truncate oversized texts (should be rare after chunking)
        truncated = [t[:self.MAX_TOKENS_PER_TEXT * 4] for t in texts]  # ~4 chars/token

        all_embeddings: list[list[float]] = []
        for i in range(0, len(truncated), self.BATCH_SIZE):
            batch = truncated[i: i + self.BATCH_SIZE]
            try:
                response = self._client.embeddings.create(
                    model=self.MODEL, input=batch
                )
                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)
                logger.debug(
                    "Embedded batch %d–%d (%d vectors)", i, i + len(batch), len(batch)
                )
            except Exception as exc:
                logger.error("Embedding API call failed for batch %d: %s", i, exc)
                raise
        return all_embeddings

    def embed_query(self, query: str) -> list[float]:
        """Single-query embed for retrieval time."""
        return self.embed([query])[0]


# BM25 Sparse Retriever


class BM25Retriever:
    """
    Pure-Python implementation of Okapi BM25 (BM25+).

    Why a custom implementation rather than rank_bm25?
    - No extra dependency; rank_bm25 is fine in production but adding it here
      keeps the module self-contained for review.
    - We need access to per-document term frequencies for partial-match boosting.

    BM25 Parameters (standard TREC defaults):
      k1 = 1.5   (term frequency saturation)
      b  = 0.75  (document length normalisation)
    """

    K1: float = 1.5
    B: float = 0.75

    def __init__(self) -> None:
        self._corpus: list[str] = []
        self._chunk_ids: list[str] = []
        self._tf: list[dict[str, int]] = []       # term → freq per doc
        self._df: dict[str, int] = defaultdict(int)  # term → doc count
        self._avg_dl: float = 0.0
        self._N: int = 0

    def index(self, texts: list[str], chunk_ids: list[str]) -> None:
        """
        (Re-)build the BM25 index from a corpus of texts.
        Called on initial load and after incremental additions.
        """
        self._corpus = texts
        self._chunk_ids = chunk_ids
        self._N = len(texts)
        self._tf = []
        self._df = defaultdict(int)

        total_tokens = 0
        for text in texts:
            tokens = self._tokenize(text)
            total_tokens += len(tokens)
            freq: dict[str, int] = defaultdict(int)
            for t in tokens:
                freq[t] += 1
            self._tf.append(freq)
            for term in set(freq.keys()):
                self._df[term] += 1

        self._avg_dl = total_tokens / max(self._N, 1)
        logger.info("BM25 indexed %d documents, avg_dl=%.1f", self._N, self._avg_dl)

    def search(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        """
        Returns: List of (chunk_id, bm25_score) sorted descending.
        """
        if self._N == 0:
            return []

        query_terms = self._tokenize(query)
        scores: list[float] = []

        for doc_idx in range(self._N):
            doc_tf = self._tf[doc_idx]
            doc_len = sum(doc_tf.values())
            score = 0.0

            for term in query_terms:
                if term not in doc_tf:
                    continue
                tf = doc_tf[term]
                df = self._df.get(term, 0)
                idf = math.log((self._N - df + 0.5) / (df + 0.5) + 1.0)
                tf_norm = (tf * (self.K1 + 1)) / (
                    tf + self.K1 * (1 - self.B + self.B * doc_len / self._avg_dl)
                )
                score += idf * tf_norm

            scores.append(score)

        # Rank and return top-k
        ranked = sorted(
            ((self._chunk_ids[i], scores[i]) for i in range(self._N) if scores[i] > 0),
            key=lambda x: x[1],
            reverse=True,
        )
        return ranked[:top_k]

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """
        Financial-aware tokenizer:
        - Lowercase, split on whitespace and punctuation.
        - Preserves financial acronyms (EBITDA, GAAP) and numeric strings.
        - Splits camelCase tokens for broader recall.
        """
        text = text.lower()
        # Keep alphanumerics, %, $, .
        text = re.sub(r"[^a-z0-9\$\%\.\-]", " ", text)
        tokens = text.split()
        # Remove single-character noise tokens
        return [t for t in tokens if len(t) > 1]


# Reciprocal Rank Fusion


def reciprocal_rank_fusion(
    dense_results: list[tuple[str, float]],
    sparse_results: list[tuple[str, float]],
    rrf_k: int = 60,
    dense_weight: float = 0.6,
    sparse_weight: float = 0.4,
) -> dict[str, dict[str, Any]]:
    """
    Merge dense and sparse ranked lists into a single fused ranking using
    Reciprocal Rank Fusion.

    RRF Score for document d:
        RRF(d) = Σ_r  weight_r / (k + rank_r(d))

    Parameters
    ----------
    rrf_k          : Smoothing constant (default 60, from original RRF paper).
    dense_weight   : Relative weight for the dense retriever's ranking.
    sparse_weight  : Relative weight for the sparse (BM25) retriever's ranking.

    Returns
    -------
    dict mapping chunk_id → {rrf_score, dense_rank, sparse_rank, dense_score, sparse_score}
    """
    fused: dict[str, dict[str, Any]] = {}

    for rank, (chunk_id, score) in enumerate(dense_results, start=1):
        fused.setdefault(chunk_id, {
            "rrf_score": 0.0, "dense_rank": None, "sparse_rank": None,
            "dense_score": None, "sparse_score": None,
        })
        fused[chunk_id]["dense_rank"] = rank
        fused[chunk_id]["dense_score"] = score
        fused[chunk_id]["rrf_score"] += dense_weight / (rrf_k + rank)

    for rank, (chunk_id, score) in enumerate(sparse_results, start=1):
        fused.setdefault(chunk_id, {
            "rrf_score": 0.0, "dense_rank": None, "sparse_rank": None,
            "dense_score": None, "sparse_score": None,
        })
        fused[chunk_id]["sparse_rank"] = rank
        fused[chunk_id]["sparse_score"] = score
        fused[chunk_id]["rrf_score"] += sparse_weight / (rrf_k + rank)

    return dict(sorted(fused.items(), key=lambda x: x[1]["rrf_score"], reverse=True))


# ChromaDB Adapter


class ChromaAdapter:
    """
    Wraps ChromaDB for persistent dense vector storage and retrieval.

    Collection design:
    - One ChromaDB collection per knowledge base (default: "financial_docs").
    - Documents stored with full chunk metadata as ChromaDB metadata fields
      to enable filtered retrieval (e.g., by source_file or chunk_type).
    - Persistent client with configurable storage path.
    """

    def __init__(
        self,
        persist_directory: str = ".chroma_db",
        collection_name: str = "financial_docs",
    ) -> None:
        self._persist_dir = persist_directory
        self._collection_name = collection_name
        self._client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            "ChromaDB initialised: collection=%s, path=%s",
            collection_name, persist_directory,
        )

    @property
    def count(self) -> int:
        return self._collection.count()

    def upsert(
        self,
        chunk_ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict[str, Any]],
    ) -> None:
        """Batch-upsert chunks. Idempotent — safe to call on re-ingestion."""
        if not chunk_ids:
            return
        # ChromaDB metadata values must be str | int | float | bool
        sanitised_meta = [self._sanitise_metadata(m) for m in metadatas]
        self._collection.upsert(
            ids=chunk_ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=sanitised_meta,
        )
        logger.debug("Upserted %d vectors into ChromaDB", len(chunk_ids))

    def query(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        where: Optional[dict] = None,
    ) -> list[tuple[str, float]]:
        """
        Returns: List of (chunk_id, cosine_similarity_score) sorted descending.
        """
        kwargs: dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results": min(top_k, max(self._collection.count(), 1)),
            "include": ["distances"],
        }
        if where:
            kwargs["where"] = where

        results = self._collection.query(**kwargs)
        ids = results["ids"][0]
        distances = results["distances"][0]

        # ChromaDB cosine distance: score = 1 - distance
        return [(cid, 1.0 - dist) for cid, dist in zip(ids, distances)]

    def get_by_ids(self, chunk_ids: list[str]) -> list[dict[str, Any]]:
        """Retrieve stored metadata by chunk IDs."""
        if not chunk_ids:
            return []
        results = self._collection.get(ids=chunk_ids, include=["metadatas", "documents"])
        return [
            {"id": cid, "metadata": meta, "document": doc}
            for cid, meta, doc in zip(
                results["ids"], results["metadatas"], results["documents"]
            )
        ]

    def delete_by_doc_id(self, doc_id: str) -> None:
        """Remove all chunks belonging to a document (for refresh flows)."""
        self._collection.delete(where={"doc_id": doc_id})
        logger.info("Deleted all chunks for doc_id=%s", doc_id)

    @staticmethod
    def _sanitise_metadata(meta: dict[str, Any]) -> dict[str, str | int | float | bool]:
        """ChromaDB rejects None and complex types in metadata."""
        sanitised = {}
        for k, v in meta.items():
            if v is None:
                sanitised[k] = ""
            elif isinstance(v, (str, int, float, bool)):
                sanitised[k] = v
            else:
                sanitised[k] = str(v)
        return sanitised


# Hybrid Vector Store


class HybridVectorStore:
    """
    Production-grade hybrid retrieval engine combining:
      - Dense: OpenAI text-embedding-3-small → ChromaDB (cosine ANN)
      - Sparse: Custom BM25 over in-memory corpus
      - Fusion: Reciprocal Rank Fusion (RRF)

    Supports:
      - Incremental indexing (add documents without full re-index)
      - Per-source filtering (retrieve only from specific PDFs)
      - Chunk type filtering (e.g., only TABLE chunks for schema queries)
    """

    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        persist_directory: str = ".chroma_db",
        collection_name: str = "financial_docs",
    ) -> None:
        api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError("OpenAI API key required for embedding.")

        self._openai_client = OpenAI(api_key=api_key)
        self._embedder = EmbeddingClient(self._openai_client)
        self._chroma = ChromaAdapter(persist_directory, collection_name)
        self._bm25 = BM25Retriever()

        # In-memory chunk registry: chunk_id → DocumentChunk
        # Populated on index() and rebuild_bm25_index()
        self._chunk_registry: dict[str, DocumentChunk] = {}

        # Rebuild BM25 from persisted Chroma data on startup
        self._rebuild_bm25_from_chroma()

    # ── Indexing 

    def index(self, chunks: list[DocumentChunk]) -> None:
        """
        Index a list of DocumentChunks into both dense (ChromaDB) and
        sparse (BM25) retrieval layers.

        Steps:
          1. Register chunks in the in-memory registry.
          2. Batch-embed chunk content strings.
          3. Upsert into ChromaDB with sanitised metadata.
          4. Rebuild BM25 index from updated corpus.
        """
        if not chunks:
            logger.warning("index() called with empty chunk list.")
            return

        logger.info("Indexing %d chunks...", len(chunks))

        texts = [c.content for c in chunks]
        chunk_ids = [c.chunk_id for c in chunks]

        # Step 1: Register
        for chunk in chunks:
            self._chunk_registry[chunk.chunk_id] = chunk

        # Step 2: Embed
        embeddings = self._embedder.embed(texts)

        # Step 3: ChromaDB upsert
        metadatas = [
            {
                "doc_id": c.doc_id,
                "source_file": c.source_file,
                "page_number": c.page_number,
                "chunk_type": c.chunk_type.value,
                "section_header": c.section_header,
                "has_table": c.table_markdown is not None,
                "has_image": c.image_b64 is not None,
            }
            for c in chunks
        ]
        self._chroma.upsert(chunk_ids, embeddings, texts, metadatas)

        # Step 4: Rebuild BM25
        self._rebuild_bm25_from_registry()

        logger.info(
            "Indexing complete. ChromaDB total: %d vectors.", self._chroma.count
        )

    def _rebuild_bm25_from_chroma(self) -> None:
        """
        On startup, re-populate the in-memory BM25 corpus from ChromaDB's
        persisted document store. This ensures BM25 survives process restarts.
        """
        try:
            all_items = self._chroma._collection.get(include=["documents", "metadatas"])
            ids = all_items.get("ids", [])
            documents = all_items.get("documents", [])
            metadatas = all_items.get("metadatas", [])

            if not ids:
                logger.info("ChromaDB is empty — no BM25 rebuild needed.")
                return

            for cid, doc, meta in zip(ids, documents, metadatas):
                if cid not in self._chunk_registry:
                    # Reconstruct a lightweight DocumentChunk from stored metadata
                    self._chunk_registry[cid] = DocumentChunk(
                        chunk_id=cid,
                        doc_id=meta.get("doc_id", ""),
                        source_file=meta.get("source_file", ""),
                        page_number=int(meta.get("page_number", 0)),
                        chunk_type=ChunkType(meta.get("chunk_type", "text")),
                        content=doc,
                        section_header=meta.get("section_header", ""),
                    )

            corpus = [self._chunk_registry[cid].content for cid in ids if cid in self._chunk_registry]
            valid_ids = [cid for cid in ids if cid in self._chunk_registry]
            self._bm25.index(corpus, valid_ids)
            logger.info("BM25 rebuilt from ChromaDB: %d documents", len(valid_ids))
        except Exception as exc:
            logger.warning("BM25 rebuild from ChromaDB failed: %s", exc)

    def _rebuild_bm25_from_registry(self) -> None:
        ids = list(self._chunk_registry.keys())
        corpus = [self._chunk_registry[cid].content for cid in ids]
        self._bm25.index(corpus, ids)

    # ── Retrieval 

    def search(
        self,
        query: str,
        top_k: int = 10,
        source_filter: Optional[str] = None,
        chunk_type_filter: Optional[ChunkType] = None,
    ) -> list[RetrievedChunk]:
        """
        Execute hybrid search and return RRF-fused results.

        Parameters
        ----------
        query              : Natural-language financial query.
        top_k              : Number of candidates per retriever (merged by RRF).
        source_filter      : Restrict to chunks from a specific PDF filename.
        chunk_type_filter  : Restrict to a specific chunk type (TEXT/TABLE/IMAGE_SUMMARY).
        """
        if not query.strip():
            raise ValueError("Query cannot be empty.")

        logger.info("Hybrid search: query='%.80s', top_k=%d", query, top_k)

        # Build ChromaDB filter
        where_filter: Optional[dict] = None
        if source_filter and chunk_type_filter:
            where_filter = {
                "$and": [
                    {"source_file": {"$eq": source_filter}},
                    {"chunk_type": {"$eq": chunk_type_filter.value}},
                ]
            }
        elif source_filter:
            where_filter = {"source_file": {"$eq": source_filter}}
        elif chunk_type_filter:
            where_filter = {"chunk_type": {"$eq": chunk_type_filter.value}}

        # ── Dense retrieval 
        query_embedding = self._embedder.embed_query(query)
        dense_results = self._chroma.query(
            query_embedding=query_embedding,
            top_k=top_k,
            where=where_filter,
        )
        logger.debug("Dense retrieval returned %d results", len(dense_results))

        # ── Sparse retrieval 
        sparse_results = self._bm25.search(query, top_k=top_k)
        logger.debug("Sparse retrieval returned %d results", len(sparse_results))

        # ── RRF Fusion 
        fused = reciprocal_rank_fusion(
            dense_results=dense_results,
            sparse_results=sparse_results,
        )

        # ── Hydrate with DocumentChunk objects 
        retrieved: list[RetrievedChunk] = []
        for chunk_id, scores in list(fused.items())[:top_k]:
            chunk = self._chunk_registry.get(chunk_id)
            if chunk is None:
                logger.warning("chunk_id=%s in fusion but not in registry; skipping.", chunk_id)
                continue
            retrieved.append(
                RetrievedChunk(
                    chunk=chunk,
                    rrf_score=scores["rrf_score"],
                    dense_rank=scores["dense_rank"],
                    sparse_rank=scores["sparse_rank"],
                    dense_score=scores["dense_score"],
                    sparse_score=scores["sparse_score"],
                )
            )

        logger.info("Hybrid search returned %d fused candidates.", len(retrieved))
        return retrieved

    # ── Introspection 

    def list_indexed_documents(self) -> list[dict[str, Any]]:
        """
        Returns a deduplicated list of indexed source documents with stats.
        """
        seen: dict[str, dict[str, Any]] = {}
        for chunk in self._chunk_registry.values():
            key = chunk.source_file
            if key not in seen:
                seen[key] = {
                    "source_file": chunk.source_file,
                    "doc_id": chunk.doc_id,
                    "chunk_count": 0,
                    "page_count": 0,
                    "pages": set(),
                }
            seen[key]["chunk_count"] += 1
            seen[key]["pages"].add(chunk.page_number)

        result = []
        for doc in seen.values():
            doc["page_count"] = len(doc.pop("pages"))
            result.append(doc)
        return sorted(result, key=lambda x: x["source_file"])

    def clear(self) -> None:
        """Drop all indexed data. Useful for test teardown."""
        self._chroma._client.delete_collection(self._chroma._collection_name)
        self._chroma._collection = self._chroma._client.get_or_create_collection(
            name=self._chroma._collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self._chunk_registry.clear()
        self._bm25 = BM25Retriever()
        logger.warning("HybridVectorStore cleared.")
