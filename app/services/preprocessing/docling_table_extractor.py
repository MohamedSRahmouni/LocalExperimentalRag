# app/services/preprocessing/docling_table_extractor.py
"""
Docling-powered table extractor
PDF → Docling → Table → DataFrame → to_markdown() → single chunk
"""

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import re

logger = logging.getLogger(__name__)


class DoclingTableExtractor:
    """
    Extract tables from PDF using Docling.
    
    Pipeline:
        PDF
         ↓
        Docling (DocumentConverter)
         ↓
        Table → pandas DataFrame
         ↓
        DataFrame.to_markdown()
         ↓
        Chunk metadata type="table"   ← single chunk per table
    """

    def __init__(self, preserve_caption: bool = True):
        self.preserve_caption = preserve_caption
        self._converter = None
        self._docling_available = False
        self._init_docling()

    # ──────────────────────────────────────────────────────────
    # INIT
    # ──────────────────────────────────────────────────────────

    def _init_docling(self):
        try:
            from docling.document_converter import DocumentConverter
            self._converter = DocumentConverter()
            self._docling_available = True
            logger.info("✅ Docling DocumentConverter ready")
        except ImportError:
            logger.warning(
                "⚠️  docling not installed — "
                "run: pip install docling"
            )
        except Exception as e:
            logger.warning(f"⚠️  Docling init failed: {e}")

    @property
    def available(self) -> bool:
        return self._docling_available and self._converter is not None

    # ──────────────────────────────────────────────────────────
    # MAIN: extract tables from a PDF
    # ──────────────────────────────────────────────────────────

    def extract_tables(
        self,
        pdf_path: Path,
    ) -> List[Dict[str, Any]]:
        """
        Extract all tables from a PDF file.

        Returns:
            List of table dicts:
            {
                "text":          str,          # markdown table
                "type":          "table",      # ← metadata marker
                "is_table":      True,
                "table_index":   int,
                "page_no":       int | None,
                "caption":       str | None,
                "row_count":     int,
                "col_count":     int,
                "source":        str,
            }
        """
        if not self.available:
            logger.warning("⚠️  Docling unavailable — no table extraction")
            return []

        try:
            import pandas as pd

            logger.info(f"📊 Docling extracting tables: {pdf_path.name}")

            # ── Convert PDF ────────────────────────────────────
            result = self._converter.convert(str(pdf_path))
            doc    = result.document

            tables_out: List[Dict[str, Any]] = []
            table_index = 0

            for element, _level in doc.iterate_items():
                # Docling exposes tables as TableItem
                if not self._is_table_item(element):
                    continue

                try:
                    # ── DataFrame ─────────────────────────────
                    df = element.export_to_dataframe()

                    if df is None or df.empty:
                        logger.debug(f"  ↳ Table {table_index}: empty, skipped")
                        continue

                    # ── to_markdown() ──────────────────────────
                    markdown_table = df.to_markdown(index=False)

                    if not markdown_table or not markdown_table.strip():
                        continue

                    # ── caption ────────────────────────────────
                    caption = self._extract_caption(element)

                    # ── page number ────────────────────────────
                    page_no = self._extract_page_no(element)

                    # ── Build full text (caption + table) ──────
                    parts = []
                    if caption and self.preserve_caption:
                        parts.append(f"**{caption}**")
                    parts.append(markdown_table.strip())
                    full_text = "\n\n".join(parts)

                    tables_out.append({
                        "text":        full_text,
                        "type":        "table",       # ← key metadata
                        "is_table":    True,
                        "table_index": table_index,
                        "page_no":     page_no,
                        "caption":     caption,
                        "row_count":   len(df),
                        "col_count":   len(df.columns),
                        "source":      str(pdf_path),
                        "filename":    pdf_path.name,
                    })

                    logger.info(
                        f"  ✅ Table {table_index}: "
                        f"{len(df)} rows × {len(df.columns)} cols"
                        + (f" | page {page_no}" if page_no else "")
                        + (f" | '{caption[:40]}'" if caption else "")
                    )

                    table_index += 1

                except Exception as tbl_err:
                    logger.warning(
                        f"  ⚠️  Table {table_index} export failed: {tbl_err}"
                    )
                    table_index += 1
                    continue

            logger.info(
                f"📊 Docling extracted {len(tables_out)} tables "
                f"from {pdf_path.name}"
            )
            return tables_out

        except Exception as e:
            logger.error(f"❌ Docling extraction error: {e}", exc_info=True)
            return []

    # ──────────────────────────────────────────────────────────
    # HELPERS
    # ──────────────────────────────────────────────────────────

    @staticmethod
    def _is_table_item(element) -> bool:
        """Check whether a Docling element is a TableItem."""
        try:
            # Docling v2 class name
            return type(element).__name__ == "TableItem"
        except Exception:
            return False

    @staticmethod
    def _extract_caption(element) -> Optional[str]:
        """Extract caption from a Docling TableItem."""
        try:
            if hasattr(element, "caption") and element.caption:
                return str(element.caption).strip()
            if hasattr(element, "label") and element.label:
                return str(element.label).strip()
        except Exception:
            pass
        return None

    @staticmethod
    def _extract_page_no(element) -> Optional[int]:
        """Extract page number from a Docling TableItem."""
        try:
            prov = getattr(element, "prov", None)
            if prov:
                if isinstance(prov, list) and prov:
                    return getattr(prov[0], "page_no", None)
                return getattr(prov, "page_no", None)
        except Exception:
            pass
        return None