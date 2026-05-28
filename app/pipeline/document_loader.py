"""
Enterprise Document Loader - Structure-Aware + Semantic + Parent Context
Version 100% compatible avec dependencies.py
"""

import logging
import hashlib
import re
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter, SentenceTransformersTokenTextSplitter

logger = logging.getLogger(__name__)


class ContentAnalyzer:
    @staticmethod
    def detect_structure(text: str) -> Dict[str, Any]:
        headers = []
        patterns = [
            r'^#{1,6}\s+(.+)$',
            r'^\d+\.\s+([A-Z].+)$',
            r'^[A-Z][A-Z\s&]{4,}:?\s*$',
            r'^[IVX]+\.\s+(.+)$',
        ]
        for pattern in patterns:
            for match in re.finditer(pattern, text, re.MULTILINE):
                headers.append({
                    "position": match.start(),
                    "title": match.group(1).strip() if len(match.groups()) > 0 else match.group(0).strip()
                })

        paragraphs = [p.strip() for p in text.split('\n\n') if len(p.strip()) > 30]
        return {
            "headers": headers,
            "header_count": len(headers),
            "paragraphs": len(paragraphs),
            "has_structure": len(headers) >= 2
        }


class LangChainDocumentLoader:

    def __init__(
        self,
        lang: str = 'fr',
        use_gpu: bool = False,
        chunk_size: int = 750,
        chunk_overlap: int = 150,
        min_chunk_size: int = 100,
        similarity_threshold: float = 0.5,
        chunking_method: str = "semantic",
        embedding_model: str = "./models/multilingual-e5-small",
        preserve_tables: bool = True,
        max_table_size: int = 5000,
        micro_threshold: int = 300,
        simple_threshold: int = 1200,
        structure_headers_min: int = 2,
        density_threshold: float = 0.45,
        enable_small_chunks: bool = False,
        small_chunk_size: int = 250,
    ):
        # Tous les paramètres d'origine conservés pour compatibilité
        self.lang = lang
        self.use_gpu = use_gpu
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        self.similarity_threshold = similarity_threshold
        self.chunking_method = chunking_method
        self.embedding_model = embedding_model
        self.preserve_tables = preserve_tables
        self.max_table_size = max_table_size
        self.micro_threshold = micro_threshold
        self.simple_threshold = simple_threshold
        self.structure_headers_min = structure_headers_min
        self.density_threshold = density_threshold
        self.enable_small_chunks = enable_small_chunks
        self.small_chunk_size = small_chunk_size

        self.processed_hashes = set()
        self.analyzer = ContentAnalyzer()
        self.splitter = self._create_splitter()

        logger.info("✅ LangChainDocumentLoader Enterprise v2 - Structure-Aware + Semantic + Parent Context Initialized")


    def _create_splitter(self):
        try:
            return SentenceTransformersTokenTextSplitter(
                model_name=self.embedding_model,
                tokens_per_chunk=self.chunk_size,
                chunk_overlap=self.chunk_overlap
            )
        except Exception:
            return RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""]
            )


    def _extract_tables_from_pdf(self, pdf_path: Path) -> List[Document]:
        if not self.preserve_tables:
            return []
        try:
            import camelot
            tables_docs = []
            for flavor in ["lattice", "stream"]:
                try:
                    tables = camelot.read_pdf(str(pdf_path), pages="all", flavor=flavor)
                    for idx, table in enumerate(tables):
                        df = table.df
                        if df.shape[0] < 2 or df.shape[1] < 2:
                            continue
                        if (df == "").sum().sum() / (df.shape[0] * df.shape[1]) > 0.6:
                            continue
                        markdown = df.to_markdown(index=False)
                        tables_docs.append(Document(
                            page_content=markdown,
                            metadata={"type": "table", "is_table": True, "table_index": idx, "model_type": "table_single_chunk"}
                        ))
                except:
                    continue
            return tables_docs
        except Exception:
            return []


    def _structure_aware_semantic_chunk(self, text: str, file_path: Path) -> List[Dict]:
        structure = self.analyzer.detect_structure(text)
        chunks = []
        index = 0

        if structure["has_structure"] and structure["header_count"] >= self.structure_headers_min:
            boundaries = [0] + [h["position"] for h in structure["headers"]] + [len(text)]
            for i in range(len(boundaries)-1):
                section = text[boundaries[i]:boundaries[i+1]].strip()
                if not section: continue
                parent = structure["headers"][i]["title"] if i < len(structure["headers"]) else "Section"

                if len(section) > self.chunk_size * 1.6:
                    for sub in self.splitter.split_text(section):
                        if len(sub) < self.min_chunk_size: continue
                        chunks.append({
                            "text": sub, "length": len(sub), "is_table": False,
                            "source": str(file_path), "chunk_index": index,
                            "model_type": "semantic_structure",
                            "parent_header": parent, "section_context": parent
                        })
                        index += 1
                else:
                    chunks.append({
                        "text": section, "length": len(section), "is_table": False,
                        "source": str(file_path), "chunk_index": index,
                        "model_type": "structured_semantic",
                        "parent_header": parent, "section_context": parent
                    })
                    index += 1
        else:
            paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
            current = []
            current_len = 0
            for para in paragraphs:
                if current_len + len(para) > self.chunk_size and current:
                    chunk_text = "\n\n".join(current)
                    chunks.append({
                        "text": chunk_text, "length": len(chunk_text), "is_table": False,
                        "source": str(file_path), "chunk_index": index,
                        "model_type": "semantic_paragraph",
                        "parent_header": "Main Content", "section_context": "Main Content"
                    })
                    index += 1
                    current = [current[-1]]
                    current_len = len(current[0])
                current.append(para)
                current_len += len(para)

            if current:
                chunks.append({
                    "text": "\n\n".join(current), "length": sum(len(p) for p in current),
                    "is_table": False, "source": str(file_path), "chunk_index": index,
                    "model_type": "semantic_paragraph",
                    "parent_header": "Main Content", "section_context": "Main Content"
                })

        return chunks


    def _chunk_documents(self, documents: List[Document], file_path: Path) -> List[Dict]:
        chunks = []
        for doc in documents:
            if doc.metadata.get("is_table"):
                chunks.append({
                    "text": doc.page_content, "length": len(doc.page_content),
                    "is_table": True, "source": str(file_path), "chunk_index": len(chunks),
                    "model_type": "table_single_chunk", "parent_header": "Table"
                })
                continue
            chunks.extend(self._structure_aware_semantic_chunk(doc.page_content, file_path))
        return chunks


    def _load_pdf(self, path: Path):
        from langchain_community.document_loaders import PyPDFLoader
        loader = PyPDFLoader(str(path))
        return loader.load() + self._extract_tables_from_pdf(path)

    def _load_docx(self, path: Path):
        from langchain_community.document_loaders import Docx2txtLoader
        return Docx2txtLoader(str(path)).load()

    def _load_txt(self, path: Path):
        from langchain_community.document_loaders import TextLoader
        return TextLoader(str(path)).load()


    def process_file(self, file_path: str) -> Dict[str, Any]:
        file_path = Path(file_path)
        result = {"success": False, "chunks": [], "chunk_count": 0, "chunks_with_metadata": [],
                  "errors": [], "metadata": {}}

        if not file_path.exists():
            result["errors"].append("File not found")
            return result

        try:
            ext = file_path.suffix.lower()
            if ext == ".pdf":
                documents = self._load_pdf(file_path)
            elif ext == ".docx":
                documents = self._load_docx(file_path)
            else:
                documents = self._load_txt(file_path)

            full_text = "\n\n".join(d.page_content for d in documents if d.page_content.strip())

            if len(full_text) < 30:
                result["errors"].append("Text too short")
                return result

            text_hash = hashlib.md5(full_text.encode()).hexdigest()
            if text_hash in self.processed_hashes:
                result["errors"].append("Duplicate")
                return result

            self.processed_hashes.add(text_hash)

            chunked = self._chunk_documents(documents, file_path)

            result.update({
                "success": True,
                "chunks": [c["text"] for c in chunked],
                "chunks_with_metadata": chunked,
                "chunk_count": len(chunked),
                "processed_text": full_text,
                "metadata": {
                    "file_type": ext,
                    "processed_at": datetime.now().isoformat(),
                    "table_chunks": sum(1 for c in chunked if c.get("is_table")),
                    "chunking_method": "structure_aware_semantic",
                    "parent_context_enabled": True
                }
            })
            return result

        except Exception as e:
            result["errors"].append(str(e))
            return result


    def process_multiple_files(
        self,
        file_paths: List[str],
        similarity_threshold: float = 0.5,
        min_chunk_size: int = 100,
        max_chunk_size: int = 750,
        dynamic_threshold: bool = False
    ) -> Dict[str, Any]:
        start_time = datetime.now()
        results = {
            "total_files": len(file_paths), "successful": 0, "failed": 0, "duplicate": 0,
            "total_chunks": 0, "total_unique_chunks": 0, "scanned_count": 0, "ocr_count": 0,
            "files": [], "all_chunks": [], "all_chunks_with_metadata": [], "errors": [],
            "processing_time": 0.0, "statistics": {}
        }

        for path in file_paths:
            res = self.process_file(path)
            results["files"].append(res)
            if res.get("success"):
                results["successful"] += 1
                results["total_chunks"] += res.get("chunk_count", 0)
                results["all_chunks"].extend(res.get("chunks", []))
                results["all_chunks_with_metadata"].extend(res.get("chunks_with_metadata", []))
            elif "Duplicate" in str(res.get("errors", [])):
                results["duplicate"] += 1
            else:
                results["failed"] += 1

        results["all_chunks"] = list(dict.fromkeys(results["all_chunks"]))
        results["total_unique_chunks"] = len(results["all_chunks"])
        results["processing_time"] = (datetime.now() - start_time).total_seconds()

        results["statistics"] = {
            "average_chunks_per_file": round(results["total_unique_chunks"] / results["successful"], 2) if results["successful"] > 0 else 0,
            "success_rate": round(results["successful"] / results["total_files"] * 100, 2),
            "chunking_method": "structure_aware_semantic",
            "parent_context": True
        }
        return results