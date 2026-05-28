"""
Vector Store Pipeline - Qdrant
LangChain-powered Qdrant integration for local Docker deployment
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct,
    Filter, FieldCondition, MatchValue,
    ScoredPoint
)

# ── LangChain Qdrant wrapper (optional — only used for search) ──
# Try multiple import paths depending on installed version
_VectorStoreClass = None

try:
    from langchain_qdrant import Qdrant as _VectorStoreClass
    _LANGCHAIN_BACKEND = "langchain_qdrant.Qdrant"
except ImportError:
    pass

if _VectorStoreClass is None:
    try:
        from langchain_qdrant import QdrantVectorStore as _VectorStoreClass
        _LANGCHAIN_BACKEND = "langchain_qdrant.QdrantVectorStore"
    except ImportError:
        pass

if _VectorStoreClass is None:
    try:
        from langchain_community.vectorstores import Qdrant as _VectorStoreClass
        _LANGCHAIN_BACKEND = "langchain_community.Qdrant"
    except ImportError:
        pass

if _VectorStoreClass is None:
    _LANGCHAIN_BACKEND = "none"
    import warnings
    warnings.warn(
        "No LangChain Qdrant integration found. "
        "Semantic search via LangChain wrapper disabled. "
        "Native Qdrant search still works.",
        ImportWarning,
        stacklevel=2,
    )

logger = logging.getLogger(__name__)
logger.info(f"Qdrant LangChain backend: {_LANGCHAIN_BACKEND}")


class LangChainQdrantStore:
    """
    Unified Qdrant vector store using LangChain.
    
    Handles:
        - Connection to local Qdrant Docker
        - Document storage with metadata
        - Semantic and hybrid search
        - Collection administration
    """

    def __init__(
        self,
        url: str = "http://localhost:6333",
        api_key: Optional[str] = None,
        collection_name: str = "rag_documents",
        embeddings=None,
        vector_size: int = 384,
        distance: str = "Cosine",
    ):
        self.url = url
        self.api_key = api_key if api_key else None
        self.collection_name = collection_name
        self.embeddings = embeddings
        self.vector_size = vector_size
        self.distance = self._parse_distance(distance)

        self.client: Optional[QdrantClient] = None
        self.vectorstore: Optional[QdrantVectorStore] = None
        self._connected: bool = False

        self._connect()

    # ================================================================
    # CONNECTION
    # ================================================================

    @staticmethod
    def _parse_distance(distance_str: str) -> Distance:
        """Convert string to Qdrant Distance enum."""
        distance_map = {
            "cosine": Distance.COSINE,
            "euclid": Distance.EUCLID,
            "euclidean": Distance.EUCLID,
            "dot": Distance.DOT,
        }
        return distance_map.get(distance_str.lower(), Distance.COSINE)

    def _connect(self) -> bool:
        """Establish connection to Qdrant."""
        logger.info("=" * 70)
        logger.info("🔌 Connecting to Qdrant")
        logger.info("=" * 70)

        try:
            self.client = QdrantClient(
                url=self.url,
                api_key=self.api_key,
                timeout=30,
            )

            # Health check
            collections = self.client.get_collections()
            logger.info(f"✅ Connected to Qdrant at {self.url}")
            logger.info(f"   Collections: {len(collections.collections)}")
            
            self._connected = True
            self._ensure_collection()

            if self.embeddings:
                self._init_vectorstore()

            logger.info("=" * 70)
            return True

        except Exception as e:
            logger.error("=" * 70)
            logger.error(f"❌ Qdrant connection failed: {e}")
            logger.error("Troubleshooting:")
            logger.error("  1. Check Docker: docker ps | grep qdrant")
            logger.error("  2. Check URL: http://localhost:6333/dashboard")
            logger.error("  3. Restart: docker restart qdrant")
            logger.error("=" * 70)
            self._connected = False
            return False

    def _reconnect_if_needed(self) -> bool:
        """Lightweight reconnection guard."""
        if self._connected and self.client is not None:
            return True

        logger.warning("⚠️  Connection lost — attempting reconnect...")
        return self._connect()

    def is_connected(self) -> bool:
        """Health check with real API call."""
        if self.client is None:
            return False

        try:
            self.client.get_collections()
            self._connected = True
            return True
        except Exception:
            self._connected = False
            return False

    def close(self):
        """Close Qdrant connection."""
        if self.client:
            self.client.close()
        self._connected = False
        logger.info("✅ Qdrant connection closed")

    # ================================================================
    # COLLECTION MANAGEMENT
    # ================================================================

    def _ensure_collection(self) -> bool:
        """Create collection if it doesn't exist."""
        try:
            collections = self.client.get_collections().collections
            exists = any(c.name == self.collection_name for c in collections)

            if exists:
                logger.info(f"✅ Collection '{self.collection_name}' exists")
                return True

            logger.info(f"📦 Creating collection '{self.collection_name}'...")

            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.vector_size,
                    distance=self.distance,
                ),
            )

            # Create payload indexes for fast filtering
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="filename",
                field_schema="keyword",
            )
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="chunk_id",
                field_schema="keyword",
            )

            logger.info(f"✅ Collection '{self.collection_name}' created")
            return True

        except Exception as e:
            logger.error(f"❌ Collection creation failed: {e}", exc_info=True)
            return False

    def _init_vectorstore(self):
        """Initialize LangChain wrapper for search helpers."""
        if not LANGCHAIN_AVAILABLE:
            logger.warning("⚠️  LangChain Qdrant integration not available")
            self.vectorstore = None
            return

        try:
            self.vectorstore = Qdrant(
                client=self.client,
                collection_name=self.collection_name,
                embeddings=self.embeddings,
            )
            logger.info("✅ LangChain Qdrant wrapper initialized")
        except Exception as e:
            logger.warning(f"⚠️  LangChain wrapper failed: {e}")
            self.vectorstore = None
    # ================================================================
    # STORAGE
    # ================================================================

    def store_batch(self, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Store multiple documents in Qdrant."""
        if not self._reconnect_if_needed():
            return {
                "documents_processed": 0,
                "chunks_stored": 0,
                "chunks_failed": sum(
                    len(d.get("embedded_chunks", [])) for d in documents
                ),
            }

        logger.info("=" * 70)
        logger.info(f"💾 Storing batch: {len(documents)} documents")
        logger.info("=" * 70)

        total_stored = 0
        total_failed = 0
        docs_processed = 0

        for doc in documents:
            if not doc.get("embedding_complete"):
                logger.warning(
                    f"⚠️  Skipping {doc.get('filename')}: not embedded"
                )
                continue

            result = self._store_document(doc)
            total_stored += result["success"]
            total_failed += result["failed"]

            if result["success"] > 0:
                docs_processed += 1

        logger.info("=" * 70)
        logger.info("✅ STORAGE COMPLETE")
        logger.info(f"   Documents : {docs_processed}")
        logger.info(f"   Stored    : {total_stored} chunks")
        logger.info(f"   Failed    : {total_failed} chunks")
        logger.info("=" * 70)

        return {
            "documents_processed": docs_processed,
            "chunks_stored": total_stored,
            "chunks_failed": total_failed,
        }

    def _store_document(self, document_data: Dict[str, Any]) -> Dict[str, int]:
        """Store a single document's chunks."""
        filename = document_data.get("filename", "unknown")
        embedded_chunks = document_data.get("embedded_chunks", [])

        if not embedded_chunks:
            return {"success": 0, "failed": 0}

        logger.info(f"💾 Storing: {filename} ({len(embedded_chunks)} chunks)")

        points = []
        now_str = datetime.now(timezone.utc).isoformat()
        file_type = document_data.get("metadata", {}).get("file_type", "")

        for i, chunk in enumerate(embedded_chunks):
            embedding = chunk.get("embedding", [])
            if not embedding:
                continue

            # Generate unique point ID (Qdrant uses integers or UUIDs)
            point_id = hash(chunk.get("chunk_id", f"{filename}_{i}")) % (2**63)

            payload = {
                "chunk_id": chunk.get("chunk_id", ""),
                "text": chunk.get("text", ""),
                "filename": filename,
                "file_type": file_type,
                "chunk_index": i,
                "total_chunks": len(embedded_chunks),
                "embedding_model": chunk.get("embedding_model", ""),
                "model_type": chunk.get("model_type", ""),
                "text_length": chunk.get("text_length", len(chunk.get("text", ""))),
                "embedded_at": chunk.get("embedded_at", now_str),
                "indexed_at": now_str,
                "similarity_score": chunk.get("similarity_score", 0.0),
                "sentence_count": chunk.get("sentence_count", 0),
                # Table metadata
                "type": chunk.get("type", "text"),
                "is_table": chunk.get("is_table", False),
                "table_index": chunk.get("table_index"),
                "page_no": chunk.get("page_no"),
                "caption": chunk.get("caption"),
                "row_count": chunk.get("row_count"),
                "col_count": chunk.get("col_count"),
            }

            points.append(PointStruct(
                id=point_id,
                vector=embedding,
                payload=payload,
            ))

        try:
            self.client.upsert(
                collection_name=self.collection_name,
                points=points,
            )

            logger.info(f"✅ {filename}: {len(points)} chunks stored")
            return {"success": len(points), "failed": 0}

        except Exception as e:
            logger.error(f"❌ Storage error for {filename}: {e}", exc_info=True)
            return {"success": 0, "failed": len(embedded_chunks)}

    # ================================================================
    # SEARCH
    # ================================================================

    def _embed_query(self, query_text: str) -> List[float]:
        """Embed query using configured embeddings model."""
        if self.embeddings is None:
            raise RuntimeError("No embeddings model configured")
        return self.embeddings.embed_query(query_text)

    @staticmethod
    def _build_result(point: ScoredPoint) -> Dict[str, Any]:
        """Convert Qdrant ScoredPoint to standard result dict."""
        payload = point.payload or {}
        text = payload.get("text", "")

        return {
            "chunk_id": payload.get("chunk_id", ""),
            "text": text,
            "score": point.score,
            "similarity": point.score,
            "distance": 1.0 - point.score if point.score else None,
            "metadata": {
                "filename": payload.get("filename", ""),
                "file_type": payload.get("file_type", ""),
                "chunk_index": payload.get("chunk_index", 0),
                "total_chunks": payload.get("total_chunks", 0),
                "similarity_score": payload.get("similarity_score", 0.0),
                "sentence_count": payload.get("sentence_count", 0),
                "model_type": payload.get("model_type", ""),
                "type": payload.get("type", "text"),
                "is_table": payload.get("is_table", False),
            },
            "text_length": payload.get("text_length", len(text)),
            "model": payload.get("embedding_model", ""),
            "preview": text[:200] + "..." if len(text) > 200 else text,
        }

    def semantic_search(
        self,
        query_text: str,
        embedder=None,
        top_k: int = 5,
        min_score: float = 0.5,
        filters: Optional[Dict] = None,
    ) -> List[Dict[str, Any]]:
        """Vector similarity search."""
        if not self._reconnect_if_needed():
            logger.error("❌ Not connected to Qdrant")
            return []

        try:
            logger.info(f"🔍 Semantic search: '{query_text[:60]}'")

            query_vector = self._embed_query(query_text)

            # Build filter if provided
            qdrant_filter = None
            if filters and filters.get("filename"):
                qdrant_filter = Filter(
                    must=[
                        FieldCondition(
                            key="filename",
                            match=MatchValue(value=filters["filename"]),
                        )
                    ]
                )

            search_result = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=top_k,
                query_filter=qdrant_filter,
                score_threshold=min_score,
            )

            results = [
                self._build_result(point)
                for point in search_result
            ]

            logger.info(f"✅ Semantic: {len(results)} results")
            return results

        except Exception as e:
            logger.error(f"❌ Semantic search error: {e}", exc_info=True)
            return []

    def hybrid_search(
        self,
        query_text: str,
        embedder=None,
        top_k: int = 5,
        min_score: float = 0.5,
        alpha: float = 0.7,
        filters: Optional[Dict] = None,
    ) -> List[Dict[str, Any]]:
        """
        Hybrid search (vector + text).
        Qdrant doesn't have built-in hybrid like Weaviate,
        so we do semantic search with text pre-filtering.
        """
        logger.info(f"🔍 Hybrid search (α={alpha}): '{query_text[:60]}'")
        
        # For now, fallback to semantic
        # To implement true hybrid, you'd need to:
        # 1. Do BM25-style text search (requires external lib)
        # 2. Combine with vector search using alpha weighting
        
        return self.semantic_search(
            query_text=query_text,
            top_k=top_k,
            min_score=min_score,
            filters=filters,
        )

    # ================================================================
    # UTILITY SEARCH
    # ================================================================

    def search_by_filename(
        self, filename: str, limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Return all chunks for a specific file."""
        if not self._reconnect_if_needed():
            return []

        try:
            scroll_result = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(
                            key="filename",
                            match=MatchValue(value=filename),
                        )
                    ]
                ),
                limit=limit,
            )

            points = scroll_result[0]  # (points, next_offset)
            results = []

            for point in points:
                payload = point.payload or {}
                results.append({
                    "chunk_id": payload.get("chunk_id"),
                    "text": payload.get("text"),
                    "metadata": {
                        "filename": payload.get("filename"),
                        "chunk_index": payload.get("chunk_index"),
                        "total_chunks": payload.get("total_chunks"),
                    },
                    "chunk_index": payload.get("chunk_index", 0),
                })

            results.sort(key=lambda x: x.get("chunk_index", 0))
            logger.info(f"✅ {len(results)} chunks for '{filename}'")
            return results

        except Exception as e:
            logger.error(f"❌ search_by_filename error: {e}", exc_info=True)
            return []

    def get_random_samples(self, count: int = 5) -> List[Dict[str, Any]]:
        """Return random documents for debugging."""
        if not self._reconnect_if_needed():
            return []

        try:
            scroll_result = self.client.scroll(
                collection_name=self.collection_name,
                limit=count,
            )

            points = scroll_result[0]
            results = []

            for point in points:
                payload = point.payload or {}
                text = payload.get("text", "")
                results.append({
                    "chunk_id": payload.get("chunk_id"),
                    "text": text,
                    "metadata": {
                        "filename": payload.get("filename"),
                        "file_type": payload.get("file_type"),
                    },
                    "preview": text[:150] + "..." if len(text) > 150 else text,
                })

            return results

        except Exception as e:
            logger.error(f"❌ get_random_samples error: {e}", exc_info=True)
            return []

    # ================================================================
    # STATISTICS & ADMIN
    # ================================================================

    def get_stats(self) -> Dict[str, Any]:
        """Return collection statistics."""
        try:
            if not self._reconnect_if_needed():
                return {"document_count": 0}

            info = self.client.get_collection(self.collection_name)

            return {
                "document_count": info.points_count,
                "collection_name": self.collection_name,
                "vector_size": info.config.params.vectors.size,
                "distance": str(info.config.params.vectors.distance),
            }

        except Exception as e:
            logger.error(f"❌ get_stats error: {e}", exc_info=True)
            return {"document_count": 0}

    def delete_collection(self) -> bool:
        """Delete the Qdrant collection."""
        try:
            self.client.delete_collection(self.collection_name)
            logger.info(f"✅ Deleted collection: {self.collection_name}")
            return True
        except Exception as e:
            logger.error(f"❌ delete_collection error: {e}", exc_info=True)
            return False

    def recreate_collection(self) -> bool:
        """Drop and recreate collection."""
        try:
            self.delete_collection()
            return self._ensure_collection()
        except Exception as e:
            logger.error(f"❌ recreate_collection error: {e}", exc_info=True)
            return False