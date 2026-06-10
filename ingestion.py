"""
ingestion.py
============
Layout-Aware Multimodal Ingestion Pipeline for Financial Documents

Architecture:
  ┌──────────────────────────────────────────────────────────────┐
  │  PDF / Financial Document                                     │
  │       │                                                       │
  │       ▼                                                       │
  │  [LayoutAwareExtractor]                                       │
  │   ├── Text Blocks (with bbox, font, section metadata)        │
  │   ├── Tables      (Markdown + raw HTML preserved)            │
  │   └── Charts/Imgs (cropped PIL.Image → Vision LLM summary)  │
  │       │                                                       │
  │       ▼                                                       │
  │  [SectionBoundaryChunker]                                     │
  │   └── Semantic chunks respecting financial section headers   │
  │       │                                                       │
  │       ▼                                                       │
  │  List[DocumentChunk]  → vector_store.py                      │
  └──────────────────────────────────────────────────────────────┘

Design Decisions:
- PyMuPDF (fitz) used for low-level layout extraction; it exposes per-block
  bounding boxes, font sizes, and flags — essential for inferring headers vs body.
- Tables are extracted via heuristic column alignment detection and serialized
  as Markdown for LLM consumption while the raw HTML is persisted for UI rendering.
- Chart images are base64-encoded and sent to a vision-capable LLM (GPT-4o) to
  generate a rich textual surrogate. This allows charts to be semantically
  searchable in the vector store without needing a separate image embedding model.
- Chunking is section-boundary-aware: we detect bold/large-font headings and
  only split at those natural boundaries, preventing a Balance Sheet from being
  torn across chunks mid-row.
"""

import base64
import hashlib
import io
import json
import logging
import os
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional

import fitz  # PyMuPDF
from openai import OpenAI
from PIL import Image

# Logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


# Domain Models


class ChunkType(str, Enum):
    TEXT = "text"
    TABLE = "table"
    IMAGE_SUMMARY = "image_summary"


@dataclass
class DocumentChunk:
    """
    Atomic unit of retrieval. Each chunk carries its content alongside
    rich metadata that powers citation rendering in the UI.
    """

    chunk_id: str
    doc_id: str
    source_file: str
    page_number: int
    chunk_type: ChunkType
    content: str                          # Primary text handed to the embedder
    section_header: str = ""             # Nearest ancestor section heading
    table_markdown: Optional[str] = None  # Preserved Markdown table (if TABLE)
    table_html: Optional[str] = None      # Raw HTML table (for UI rendering)
    image_b64: Optional[str] = None       # Base64 PNG crop (for UI rendering)
    bbox: Optional[tuple[float, float, float, float]] = None  # (x0,y0,x1,y1)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "source_file": self.source_file,
            "page_number": self.page_number,
            "chunk_type": self.chunk_type.value,
            "content": self.content,
            "section_header": self.section_header,
            "table_markdown": self.table_markdown,
            "table_html": self.table_html,
            "image_b64": self.image_b64,
            "bbox": list(self.bbox) if self.bbox else None,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "DocumentChunk":
        d["chunk_type"] = ChunkType(d["chunk_type"])
        d["bbox"] = tuple(d["bbox"]) if d.get("bbox") else None
        return cls(**d)


# Vision LLM — Chart Summarisation



class VisionSummarizer:
    """
    Sends extracted chart/image crops to GPT-4o Vision and returns a
    structured financial description suitable for dense embedding.

    The prompt is crafted to elicit: chart type, axes, trend direction,
    key data points, and business implication — maximising retrieval utility.
    """

    CHART_PROMPT = """You are a senior equity research analyst reviewing an image 
extracted from a financial document (earnings report, 10-K, or pitchbook).

Describe this image with maximum precision for downstream semantic search:
1. Chart/visual type (e.g., grouped bar chart, line trend, pie chart, waterfall).
2. Axes labels and units if visible.
3. Key data points, values, or percentages.
4. The direction of any trends (e.g., revenue CAGR, margin compression).
5. Business context and what this metric implies for the company's financial health.
6. Any annotations, legends, or callout boxes present.

Return ONLY the structured description. Do NOT add commentary about image quality."""

    def __init__(self, openai_client: OpenAI) -> None:
        self._client = openai_client

    def summarise(self, image_b64: str, context_hint: str = "") -> str:
        """
        Parameters
        ----------
        image_b64   : Base64-encoded PNG/JPEG of the chart crop.
        context_hint: Optional surrounding text (e.g., section title) to help
                      the model disambiguate similar-looking charts.
        """
        messages: list[dict] = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": self.CHART_PROMPT
                        + (f"\n\nContext from surrounding text: {context_hint}" if context_hint else ""),
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_b64}",
                            "detail": "high",
                        },
                    },
                ],
            }
        ]
        try:
            response = self._client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                max_tokens=512,
                temperature=0.0,
            )
            return response.choices[0].message.content.strip()
        except Exception as exc:
            logger.error("VisionSummarizer failed: %s", exc)
            return f"[Chart image — vision summarisation unavailable: {exc}]"


# Table Extractor


class TableExtractor:
    """
    Heuristic-based financial table extraction from PyMuPDF text blocks.

    Strategy:
    - Identify rows where ≥3 tokens are right-aligned with consistent column
      spacing (characteristic of financial statement tables).
    - Reconstruct as Markdown (for LLM embedding) and HTML (for UI display).
    - Preserve numeric formatting (parentheses for negatives, $ symbols, %).

    For production at scale, drop in `camelot-py` or `pdfplumber` for
    higher-fidelity table parsing with grid detection.
    """

    # Minimum column count to qualify as a "table" row
    MIN_COLS = 3

    def extract_tables_from_page(
        self, page: fitz.Page
    ) -> list[dict[str, Any]]:
        """
        Returns a list of detected tables, each containing:
          - 'rows': List[List[str]]
          - 'bbox': (x0, y0, x1, y1)
          - 'markdown': str
          - 'html': str
        """
        blocks = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)["blocks"]
        tables = []

        # Group text spans by approximate Y-coordinate (same row = within 3pt)
        row_map: dict[int, list[dict]] = {}
        for block in blocks:
            if block.get("type") != 0:  # type 0 = text
                continue
            for line in block.get("lines", []):
                y_key = round(line["bbox"][1] / 3) * 3  # bucket to 3pt rows
                if y_key not in row_map:
                    row_map[y_key] = []
                row_map[y_key].extend(line.get("spans", []))

        # Identify rows with numeric-heavy content (financial table rows)
        table_rows: list[list[str]] = []
        table_bboxes: list[tuple] = []
        numeric_pattern = re.compile(r"[\d,\.\$\%\(\)\-]+")

        for y_key in sorted(row_map.keys()):
            spans = sorted(row_map[y_key], key=lambda s: s["bbox"][0])
            texts = [s["text"].strip() for s in spans if s["text"].strip()]
            if len(texts) < self.MIN_COLS:
                continue
            numeric_cols = sum(1 for t in texts if numeric_pattern.search(t))
            if numeric_cols >= self.MIN_COLS - 1:
                table_rows.append(texts)
                xs = [s["bbox"][0] for s in spans]
                ys = [s["bbox"][1] for s in spans]
                xe = [s["bbox"][2] for s in spans]
                ye = [s["bbox"][3] for s in spans]
                table_bboxes.append((min(xs), min(ys), max(xe), max(ye)))

        if not table_rows:
            return []

        # Cluster consecutive rows into a single table block
        clusters: list[list[list[str]]] = []
        cluster_bboxes: list[list[tuple]] = []
        current_cluster: list[list[str]] = [table_rows[0]]
        current_bboxes: list[tuple] = [table_bboxes[0]]

        for i in range(1, len(table_rows)):
            y_gap = table_bboxes[i][1] - table_bboxes[i - 1][3]
            if y_gap < 20:  # rows within 20pt belong to same table
                current_cluster.append(table_rows[i])
                current_bboxes.append(table_bboxes[i])
            else:
                clusters.append(current_cluster)
                cluster_bboxes.append(current_bboxes)
                current_cluster = [table_rows[i]]
                current_bboxes = [table_bboxes[i]]
        clusters.append(current_cluster)
        cluster_bboxes.append(current_bboxes)

        for rows, bboxes in zip(clusters, cluster_bboxes):
            if len(rows) < 2:
                continue
            md = self._rows_to_markdown(rows)
            html = self._rows_to_html(rows)
            x0 = min(b[0] for b in bboxes)
            y0 = min(b[1] for b in bboxes)
            x1 = max(b[2] for b in bboxes)
            y1 = max(b[3] for b in bboxes)
            tables.append(
                {"rows": rows, "bbox": (x0, y0, x1, y1), "markdown": md, "html": html}
            )
        return tables

    @staticmethod
    def _rows_to_markdown(rows: list[list[str]]) -> str:
        if not rows:
            return ""
        # Determine max columns across all rows
        max_cols = max(len(r) for r in rows)
        # Pad rows to uniform width
        padded = [r + [""] * (max_cols - len(r)) for r in rows]
        # Column widths
        widths = [max(len(padded[i][j]) for i in range(len(padded))) for j in range(max_cols)]
        lines = []
        for idx, row in enumerate(padded):
            line = "| " + " | ".join(cell.ljust(widths[j]) for j, cell in enumerate(row)) + " |"
            lines.append(line)
            if idx == 0:  # header separator
                sep = "| " + " | ".join("-" * widths[j] for j in range(max_cols)) + " |"
                lines.append(sep)
        return "\n".join(lines)

    @staticmethod
    def _rows_to_html(rows: list[list[str]]) -> str:
        if not rows:
            return ""
        html_parts = [
            '<table class="financial-table">',
            "<thead><tr>",
        ]
        for cell in rows[0]:
            html_parts.append(f"<th>{cell}</th>")
        html_parts.append("</tr></thead><tbody>")
        for row in rows[1:]:
            html_parts.append("<tr>")
            for cell in row:
                css = 'class="num-cell"' if re.search(r"[\d,\.\$\%]", cell) else ""
                html_parts.append(f"<td {css}>{cell}</td>")
            html_parts.append("</tr>")
        html_parts.append("</tbody></table>")
        return "".join(html_parts)



# Section Boundary Chunker


class SectionBoundaryChunker:
    """
    Splits a page's text blocks into semantically coherent chunks by detecting
    financial section headings (large/bold font) as natural boundaries.

    Why not fixed token windows?
    Fixed windows blindly bisect tables mid-row and split financial narratives
    at arbitrary points, increasing hallucination risk. Section-boundary chunking
    keeps "Consolidated Balance Sheet" or "Segment Revenue Analysis" atomic,
    which is exactly what a financial analyst reading the document would expect.

    Configurable:
      max_chunk_chars: Hard cap to prevent embedding model overflow (default 3000).
      heading_font_size_threshold: Font size above which a span is treated as a heading.
    """

    FINANCIAL_SECTION_PATTERNS = [
        r"^(consolidated\s+)?(balance\s+sheet|income\s+statement|cash\s+flow)",
        r"^(segment|geographic|product)\s+(revenue|performance|analysis)",
        r"^(management[''s]?\s+discussion|md&a)",
        r"^(risk\s+factors|liquidity|capital\s+resources)",
        r"^(notes?\s+to\s+(the\s+)?financial)",
        r"^(quarterly|annual)\s+(results|summary|overview)",
        r"^(non.gaap|adjusted)\s+(measures?|reconciliation)",
        r"^\d+\.\s+[A-Z]",  # Numbered sections in 10-K
    ]

    def __init__(
        self,
        max_chunk_chars: int = 3000,
        heading_font_size_threshold: float = 12.0,
    ) -> None:
        self._max_chars = max_chunk_chars
        self._heading_threshold = heading_font_size_threshold
        self._compiled_patterns = [
            re.compile(p, re.IGNORECASE) for p in self.FINANCIAL_SECTION_PATTERNS
        ]

    def is_section_heading(self, span_text: str, font_size: float, font_flags: int) -> bool:
        """
        Classifies a text span as a section heading via:
          1. Font size above threshold (headings are typically 14–18pt)
          2. Bold flag set (bit 4 in PyMuPDF font flags)
          3. Matches known financial section patterns
        """
        is_large = font_size >= self._heading_threshold
        is_bold = bool(font_flags & 16)  # PyMuPDF bold flag
        matches_pattern = any(p.search(span_text.strip()) for p in self._compiled_patterns)
        return (is_large and is_bold) or matches_pattern

    def chunk_text_blocks(
        self,
        blocks: list[dict[str, Any]],
        page_number: int,
        doc_id: str,
        source_file: str,
    ) -> list[DocumentChunk]:
        """
        Parameters
        ----------
        blocks : Raw PyMuPDF text blocks from page.get_text("dict").
        """
        chunks: list[DocumentChunk] = []
        current_section = "Preamble"
        buffer_lines: list[str] = []
        buffer_start_bbox: Optional[tuple] = None

        def flush_buffer() -> None:
            nonlocal buffer_lines, buffer_start_bbox
            text = "\n".join(buffer_lines).strip()
            if not text:
                buffer_lines = []
                buffer_start_bbox = None
                return
            # Hard-cap: if buffer exceeds max_chunk_chars, split by sentence
            sub_chunks = self._split_by_sentence_if_needed(text)
            for sub in sub_chunks:
                chunk_id = _make_chunk_id(doc_id, page_number, sub[:64])
                chunks.append(
                    DocumentChunk(
                        chunk_id=chunk_id,
                        doc_id=doc_id,
                        source_file=source_file,
                        page_number=page_number,
                        chunk_type=ChunkType.TEXT,
                        content=sub,
                        section_header=current_section,
                        bbox=buffer_start_bbox,
                    )
                )
            buffer_lines = []
            buffer_start_bbox = None

        for block in blocks:
            if block.get("type") != 0:
                continue
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    text = span.get("text", "").strip()
                    if not text:
                        continue
                    font_size = span.get("size", 10.0)
                    font_flags = span.get("flags", 0)

                    if self.is_section_heading(text, font_size, font_flags):
                        flush_buffer()
                        current_section = text
                        # Section heading itself becomes a lightweight anchor chunk
                        # (not standalone — it seeds the next chunk's section_header)
                    else:
                        if buffer_start_bbox is None:
                            buffer_start_bbox = tuple(span.get("bbox", (0, 0, 0, 0)))
                        buffer_lines.append(text)

                        combined = " ".join(buffer_lines)
                        if len(combined) >= self._max_chars:
                            flush_buffer()

        flush_buffer()
        return chunks

    def _split_by_sentence_if_needed(self, text: str) -> list[str]:
        """
        Falls back to sentence-level splitting when a section exceeds max_chunk_chars.
        Preserves paragraph integrity where possible.
        """
        if len(text) <= self._max_chars:
            return [text]
        # Simple sentence splitter — replace with spaCy sentencizer in production
        sentences = re.split(r"(?<=[.!?])\s+", text)
        sub_chunks: list[str] = []
        current = ""
        for sent in sentences:
            if len(current) + len(sent) + 1 <= self._max_chars:
                current = (current + " " + sent).strip()
            else:
                if current:
                    sub_chunks.append(current)
                current = sent
        if current:
            sub_chunks.append(current)
        return sub_chunks or [text]


# Core Layout-Aware Extractor


class LayoutAwareExtractor:
    """
    Orchestrates end-to-end extraction of a financial PDF into DocumentChunks.

    Extraction order per page:
      1. Detect and extract structured tables → DocumentChunk(TABLE)
      2. Extract chart/image crops → VisionSummarizer → DocumentChunk(IMAGE_SUMMARY)
      3. Section-boundary-chunk remaining text blocks → DocumentChunk(TEXT)

    Note on image resolution:
      Charts are rendered at 2x DPI (144 dpi) to ensure the vision model
      receives legible axis labels and small numerics.
    """

    IMAGE_RENDER_DPI: int = 144  # 2× standard PDF DPI for crisp vision input
    MIN_IMAGE_AREA_PX: int = 5000  # Ignore tiny decorative icons (< 5000 px²)

    def __init__(
        self,
        openai_client: OpenAI,
        enable_vision: bool = True,
    ) -> None:
        self._vision = VisionSummarizer(openai_client)
        self._table_extractor = TableExtractor()
        self._chunker = SectionBoundaryChunker()
        self._enable_vision = enable_vision

    def extract(self, pdf_path: str | Path) -> list[DocumentChunk]:
        """
        Main entry point. Returns all DocumentChunks for an entire PDF.

        Parameters
        ----------
        pdf_path : Path to the financial PDF file.
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        doc_id = _make_doc_id(str(pdf_path))
        source_file = pdf_path.name
        all_chunks: list[DocumentChunk] = []

        logger.info("Opening PDF: %s (doc_id=%s)", source_file, doc_id)

        with fitz.open(str(pdf_path)) as pdf_doc:
            total_pages = len(pdf_doc)
            logger.info("Total pages: %d", total_pages)

            for page_idx, page in enumerate(pdf_doc):
                page_num = page_idx + 1
                logger.debug("Processing page %d/%d", page_num, total_pages)

                try:
                    page_chunks = self._process_page(
                        page=page,
                        page_num=page_num,
                        doc_id=doc_id,
                        source_file=source_file,
                    )
                    all_chunks.extend(page_chunks)
                except Exception as exc:
                    logger.error(
                        "Failed to process page %d of %s: %s",
                        page_num, source_file, exc, exc_info=True,
                    )
                    continue

        logger.info(
            "Extraction complete: %d chunks from %s", len(all_chunks), source_file
        )
        return all_chunks

    def _process_page(
        self,
        page: fitz.Page,
        page_num: int,
        doc_id: str,
        source_file: str,
    ) -> list[DocumentChunk]:
        chunks: list[DocumentChunk] = []

        # ── Step 1: Tables 
        tables = self._table_extractor.extract_tables_from_page(page)
        table_bboxes = set()
        for tbl in tables:
            chunk_id = _make_chunk_id(doc_id, page_num, tbl["markdown"][:64])
            # Build a text representation for the embedder: header + first 5 rows
            text_repr = f"Financial Table:\n{tbl['markdown']}"
            chunks.append(
                DocumentChunk(
                    chunk_id=chunk_id,
                    doc_id=doc_id,
                    source_file=source_file,
                    page_number=page_num,
                    chunk_type=ChunkType.TABLE,
                    content=text_repr,
                    table_markdown=tbl["markdown"],
                    table_html=tbl["html"],
                    bbox=tbl["bbox"],
                )
            )
            table_bboxes.add(tbl["bbox"])
            logger.debug("Extracted table on page %d (rows=%d)", page_num, len(tbl["rows"]))

        # ── Step 2: Images / Charts 
        if self._enable_vision:
            image_chunks = self._extract_image_chunks(
                page=page,
                page_num=page_num,
                doc_id=doc_id,
                source_file=source_file,
            )
            chunks.extend(image_chunks)

        # ── Step 3: Text Blocks 
        raw_dict = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)
        text_blocks = raw_dict.get("blocks", [])
        text_chunks = self._chunker.chunk_text_blocks(
            blocks=text_blocks,
            page_number=page_num,
            doc_id=doc_id,
            source_file=source_file,
        )
        chunks.extend(text_chunks)

        return chunks

    def _extract_image_chunks(
        self,
        page: fitz.Page,
        page_num: int,
        doc_id: str,
        source_file: str,
    ) -> list[DocumentChunk]:
        """
        Renders embedded images from the page at high DPI, filters by minimum
        area, and generates vision LLM summaries.
        """
        image_chunks: list[DocumentChunk] = []
        image_list = page.get_images(full=True)

        for img_index, img_info in enumerate(image_list):
            xref = img_info[0]
            try:
                base_image = page.parent.extract_image(xref)
                img_bytes = base_image["image"]
                img_ext = base_image["ext"]

                # Filter: skip tiny decorative icons
                pil_img = Image.open(io.BytesIO(img_bytes))
                area = pil_img.width * pil_img.height
                if area < self.MIN_IMAGE_AREA_PX:
                    logger.debug(
                        "Skipping small image xref=%d (area=%dpx²)", xref, area
                    )
                    continue

                # Re-encode as PNG for normalised base64
                png_buf = io.BytesIO()
                pil_img.convert("RGB").save(png_buf, format="PNG")
                image_b64 = base64.b64encode(png_buf.getvalue()).decode("utf-8")

                logger.debug(
                    "Summarising image xref=%d on page %d (%dx%d)",
                    xref, page_num, pil_img.width, pil_img.height,
                )
                summary = self._vision.summarise(image_b64)

                chunk_id = _make_chunk_id(doc_id, page_num, f"img_{xref}")
                image_chunks.append(
                    DocumentChunk(
                        chunk_id=chunk_id,
                        doc_id=doc_id,
                        source_file=source_file,
                        page_number=page_num,
                        chunk_type=ChunkType.IMAGE_SUMMARY,
                        content=summary,
                        image_b64=image_b64,
                        metadata={"image_ext": img_ext, "width": pil_img.width, "height": pil_img.height},
                    )
                )
            except Exception as exc:
                logger.warning(
                    "Could not process image xref=%d on page %d: %s",
                    xref, page_num, exc,
                )
                continue

        return image_chunks



# Cache Layer — avoid re-ingesting unchanged PDFs

class IngestionCache:
    """
    Filesystem-based cache keyed on (file_path, mtime, size) to prevent
    redundant re-ingestion of unchanged PDFs. Cache entries are JSON files.
    """

    def __init__(self, cache_dir: str = ".ingestion_cache") -> None:
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_key(self, pdf_path: Path) -> str:
        stat = pdf_path.stat()
        fingerprint = f"{pdf_path.name}_{stat.st_mtime}_{stat.st_size}"
        return hashlib.sha256(fingerprint.encode()).hexdigest()

    def get(self, pdf_path: Path) -> Optional[list[DocumentChunk]]:
        key = self._cache_key(pdf_path)
        cache_file = self._cache_dir / f"{key}.json"
        if cache_file.exists():
            try:
                with cache_file.open("r", encoding="utf-8") as f:
                    raw = json.load(f)
                logger.info("Cache HIT for %s", pdf_path.name)
                return [DocumentChunk.from_dict(d) for d in raw]
            except Exception as exc:
                logger.warning("Cache read failed for %s: %s", pdf_path.name, exc)
        return None

    def set(self, pdf_path: Path, chunks: list[DocumentChunk]) -> None:
        key = self._cache_key(pdf_path)
        cache_file = self._cache_dir / f"{key}.json"
        try:
            with cache_file.open("w", encoding="utf-8") as f:
                json.dump([c.to_dict() for c in chunks], f, ensure_ascii=False, indent=2)
            logger.info("Cached %d chunks for %s", len(chunks), pdf_path.name)
        except Exception as exc:
            logger.error("Cache write failed for %s: %s", pdf_path.name, exc)



# Public API


def ingest_document(
    pdf_path: str | Path,
    openai_api_key: Optional[str] = None,
    enable_vision: bool = True,
    use_cache: bool = True,
) -> list[DocumentChunk]:
    """
    Top-level ingestion function. Wraps extraction with caching and returns
    a list of DocumentChunks ready for vector store indexing.

    Parameters
    ----------
    pdf_path      : Path to the PDF file.
    openai_api_key: OpenAI API key (falls back to OPENAI_API_KEY env var).
    enable_vision : Whether to run vision LLM on embedded charts. Disable
                    during development to save API costs.
    use_cache     : Whether to use the filesystem cache layer.
    """
    pdf_path = Path(pdf_path)
    api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "OpenAI API key required. Set OPENAI_API_KEY or pass openai_api_key."
        )

    client = OpenAI(api_key=api_key)
    cache = IngestionCache()

    if use_cache:
        cached = cache.get(pdf_path)
        if cached:
            return cached

    extractor = LayoutAwareExtractor(openai_client=client, enable_vision=enable_vision)
    chunks = extractor.extract(pdf_path)

    if use_cache:
        cache.set(pdf_path, chunks)

    return chunks


# Utility helpers


def _make_doc_id(path: str) -> str:
    return hashlib.sha256(path.encode()).hexdigest()[:16]


def _make_chunk_id(doc_id: str, page: int, content_prefix: str) -> str:
    raw = f"{doc_id}_{page}_{content_prefix}"
    return hashlib.sha256(raw.encode()).hexdigest()[:24]
