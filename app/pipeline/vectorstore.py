"""
Vector Store Pipeline
LangChain-powered Weaviate integration
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone

import weaviate
import weaviate.classes.init as wvc_init
from weaviate.auth import AuthApiKey
from weaviate.classes.config import Configure, Property, DataType
from weaviate.classes.data import DataObject
from weaviate.classes.query import MetadataQuery, Filter
from langchain_weaviate import WeaviateVectorStore

logger = logging.getLogger(__name__)  # FIX: was `name` (missing double underscores)


class LangChainVectorStore:
    """
    Unified Weaviate vector store using LangChain.
    Handles:
        - Connection management with reconnect
        - Document storage via native Weaviate client
        - Semantic and hybrid search
        - Collection administration
    """

    def __init__(
        self,
        url: str,
        api_key: str,
        class_name: str = "Ragdocument",
        embeddings=None,
        vector_dims: int = 1536,
    ):
        self.url = url
        self.api_key = api_key
        self.class_name = class_name
        self.embeddings = embeddings
        self.vector_dims = vector_dims

        self.weaviate_client: Optional[weaviate.WeaviateClient] = None
        self.vectorstore: Optional[WeaviateVectorStore] = None

        # Cache connection state to avoid repeated HTTP calls
        self._connected: bool = False

        self._connect()

    # ================================================================
    # CONNECTION
    # ================================================================

    def _build_client(self) -> weaviate.WeaviateClient:
        """Create and return a new Weaviate client (not yet verified)."""
        cluster_url = self.url
        if not cluster_url.startswith(("http://", "https://")):
            cluster_url = f"https://{cluster_url}"

        return weaviate.connect_to_wcs(
            cluster_url=cluster_url,
            auth_credentials=AuthApiKey(self.api_key),
            skip_init_checks=True,
            additional_config=wvc_init.AdditionalConfig(
                timeout=wvc_init.Timeout(init=30, query=30, insert=120)
            ),
        )

    def _connect(self) -> bool:
        """
        Establish connection to Weaviate.
        Returns True on success, False on failure.
        """
        logger.info("=" * 70)
        logger.info("🔌 Connecting to Weaviate Cloud")
        logger.info("=" * 70)

        # Close any stale client before reconnecting
        self._safe_close()

        try:
            self.weaviate_client = self._build_client()

            if not self.weaviate_client.is_ready():
                logger.error("❌ Weaviate not ready after connect")
                self._connected = False
                return False

            logger.info("✅ Connected to Weaviate Cloud")
            self._connected = True

            self._ensure_collection()

            if self.embeddings:
                self._init_vectorstore()

            logger.info("=" * 70)
            return True

        except Exception as e:
            logger.error("=" * 70)
            logger.error(f"❌ Weaviate connection failed: {e}")
            logger.error("Troubleshooting:")
            logger.error("  1. Check WEAVIATE_URL in .env")
            logger.error("  2. Verify WEAVIATE_API_KEY")
            logger.error("  3. Check cluster status in Weaviate Console")
            logger.error("=" * 70)
            self._connected = False
            self.weaviate_client = None
            self.vectorstore = None
            return False

    def _reconnect_if_needed(self) -> bool:
        """
        Lightweight guard: only makes an HTTP call when our cached
        state says we're disconnected. Attempts one reconnect.
        """
        if self._connected and self.weaviate_client is not None:
            # Optimistic: trust cached state, avoid HTTP call
            return True

        logger.warning("⚠️  Connection lost — attempting reconnect...")
        return self._connect()

    def _safe_close(self):
        """Close existing client without raising."""
        if self.weaviate_client is not None:
            try:
                self.weaviate_client.close()
            except Exception:
                pass
            self.weaviate_client = None
        self._connected = False

    def is_connected(self) -> bool:
        """
        Public health check.
        Makes one real HTTP call; updates cached state.
        Call sparingly (e.g. health endpoints, not per-document).
        """
        if self.weaviate_client is None:
            self._connected = False
            return False

        try:
            ready = self.weaviate_client.is_ready()
            self._connected = ready
            return ready
        except Exception:
            self._connected = False
            return False

    def close(self):
        """Gracefully close the Weaviate connection."""
        self._safe_close()
        logger.info("✅ Weaviate connection closed")

    # ================================================================
    # COLLECTION MANAGEMENT
    # ================================================================

    def _ensure_collection(self) -> bool:
        """Create collection if it does not exist."""
        try:
            if self.weaviate_client.collections.exists(self.class_name):
                logger.info(f"✅ Collection '{self.class_name}' exists")
                return True

            logger.info(f"📦 Creating collection '{self.class_name}'...")

            self.weaviate_client.collections.create(
                name=self.class_name,
                description="RAG document chunks with embeddings",
                vectorizer_config=Configure.Vectorizer.none(),
                properties=[
                    Property(
                        name="chunk_id",
                        data_type=DataType.TEXT,
                        description="Unique chunk identifier",
                    ),
                    Property(
                        name="text",
                        data_type=DataType.TEXT,
                        description="Chunk text content",
                    ),
                    Property(
                        name="text_length",
                        data_type=DataType.INT,
                        description="Text length in characters",
                    ),
                    Property(
                        name="embedding_model",
                        data_type=DataType.TEXT,
                        description="Embedding model name",
                    ),
                    Property(
                        name="model_type",
                        data_type=DataType.TEXT,
                        description="Model type",
                    ),
                    Property(
                        name="embedded_at",
                        data_type=DataType.DATE,
                        description="Embedding timestamp",
                    ),
                    Property(
                        name="indexed_at",
                        data_type=DataType.DATE,
                        description="Indexing timestamp",
                    ),
                    Property(
                        name="filename",
                        data_type=DataType.TEXT,
                        description="Source filename",
                    ),
                    Property(
                        name="file_type",
                        data_type=DataType.TEXT,
                        description="File extension",
                    ),
                    Property(
                        name="chunk_index",
                        data_type=DataType.INT,
                        description="Chunk position in document",
                    ),
                    Property(
                        name="total_chunks",
                        data_type=DataType.INT,
                        description="Total chunks in document",
                    ),
                    Property(
                        name="similarity_score",
                        data_type=DataType.NUMBER,
                        skip_vectorization=True,
                    ),
                    Property(
                        name="sentence_count",
                        data_type=DataType.INT,
                        skip_vectorization=True,
                    ),
                ],
            )

            logger.info(f"✅ Collection '{self.class_name}' created")
            return True

        except Exception as e:
            logger.error(f"❌ Collection creation failed: {e}", exc_info=True)
            return False

    def _init_vectorstore(self):
        """Initialise LangChain wrapper (used for search helpers only)."""
        try:
            self.vectorstore = WeaviateVectorStore(
                client=self.weaviate_client,
                index_name=self.class_name,
                text_key="text",
                embedding=self.embeddings,
                attributes=[
                    "chunk_id",
                    "filename",
                    "file_type",
                    "chunk_index",
                    "total_chunks",
                    "embedding_model",
                    "model_type",
                    "text_length",
                    "sentence_count",
                    "similarity_score",
                ],
            )
            logger.info("✅ LangChain VectorStore initialised (search only)")
        except Exception as e:
            logger.warning(
                f"⚠️  LangChain wrapper failed: {e} — search may be limited"
            )
            self.vectorstore = None

    # ================================================================
    # STORAGE
    # ================================================================

    def store_batch(self, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Store multiple documents in Weaviate.
        Guard uses _reconnect_if_needed() + native client only.
        Does not require self.vectorstore to be present.
        """
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

    @staticmethod
    def _normalise_timestamp(value: Optional[str], fallback: str) -> str:
        """
        Ensure RFC-3339 / ISO-8601 format Weaviate accepts.
        e.g. '2024-01-15T10:30:00.000Z'
        """
        if value and "T" in value:
            base = value.split(".")[0]  # strip sub-seconds
            return f"{base}.000Z"
        return fallback

    def _build_batch_objects(
        self,
        document_data: Dict[str, Any],
    ) -> tuple[List[DataObject], int]:
        """
        Convert embedded chunks → DataObject list.

        Returns:
            (batch_objects, skipped_count)
        """
        embedded_chunks = document_data.get("embedded_chunks", [])
        filename = document_data.get("filename", "unknown")
        file_type = document_data.get("metadata", {}).get("file_type", "")
        total = len(embedded_chunks)
        now_str = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")

        batch_objects: List[DataObject] = []
        skipped = 0

        for i, chunk in enumerate(embedded_chunks):
            embedding = chunk.get("embedding", [])
            if not embedding:
                logger.warning(
                    f"⚠️  Empty embedding for chunk {i} in {filename}, skipping"
                )
                skipped += 1
                continue

            # Safe type coercions
            similarity_score = chunk.get("similarity_score")
            sentence_count = chunk.get("sentence_count")
            text_length = chunk.get("text_length")
            text = chunk.get("text", "")

            properties = {
                "chunk_id":         chunk.get("chunk_id", ""),
                "text":             text,
                "filename":         filename,
                "file_type":        file_type,
                "chunk_index":      i,
                "total_chunks":     total,
                "embedding_model":  chunk.get("embedding_model", ""),
                "model_type":       chunk.get("model_type", ""),
                "text_length":      int(text_length) if text_length is not None else len(text),
                "embedded_at":      self._normalise_timestamp(
                                        chunk.get("embedded_at"), now_str
                                    ),
                "indexed_at":       now_str,
                "similarity_score": float(similarity_score) if similarity_score is not None else 0.0,
                "sentence_count":   int(sentence_count) if sentence_count is not None else 0,
            }

            batch_objects.append(DataObject(properties=properties, vector=embedding))

        return batch_objects, skipped

    def _store_document(self, document_data: Dict[str, Any]) -> Dict[str, int]:
        """
        Store a single document's chunks via native Weaviate batch.

        Weaviate v4 _BatchCollection does NOT expose failed_objects on the
        context-manager object. Errors are instead returned per add_object()
        call as WeaviateObject responses, or raised as exceptions.
        We track failures by catching per-object errors explicitly.
        """
        filename = document_data.get("filename", "unknown")
        embedded_chunks = document_data.get("embedded_chunks", [])

        if not embedded_chunks:
            return {"success": 0, "failed": 0}

        logger.info(f"💾 Storing: {filename} ({len(embedded_chunks)} chunks)")

        batch_objects, skipped = self._build_batch_objects(document_data)

        if not batch_objects:
            logger.warning(f"⚠️  No valid chunks to store for {filename}")
            return {"success": 0, "failed": skipped}

        success_count = 0
        failed_count = skipped

        try:
            collection = self.weaviate_client.collections.get(self.class_name)

            with collection.batch.dynamic() as batch:
                for i, obj in enumerate(batch_objects):
                    try:
                        batch.add_object(
                            properties=obj.properties,
                            vector=obj.vector,
                        )
                        success_count += 1
                    except Exception as obj_err:
                        failed_count += 1
                        logger.error(
                            f"   ❌ Chunk {i} failed "
                            f"({obj.properties.get('chunk_id', '?')}): {obj_err}"
                        )

            logger.info(
                f"✅ {filename}: {success_count} stored, {failed_count} failed"
            )

            # Mark as potentially disconnected if everything failed
            if success_count == 0 and len(batch_objects) > 0:
                self._connected = False

            return {"success": success_count, "failed": failed_count}

        except weaviate.exceptions.WeaviateConnectionError as e:
            logger.error(f"❌ Connection lost storing {filename}: {e}")
            self._connected = False
            return {"success": 0, "failed": len(embedded_chunks)}

        except Exception as e:
            logger.error(f"❌ Storage error for {filename}: {e}", exc_info=True)
            return {"success": 0, "failed": len(embedded_chunks)}

    # ================================================================
    # SEARCH (shared helpers)
    # ================================================================

    _RETURN_PROPERTIES = [
        "chunk_id", "text", "filename", "file_type",
        "chunk_index", "total_chunks", "embedding_model",
        "model_type", "text_length", "sentence_count", "similarity_score",
    ]

    @staticmethod
    def _safe_get(d: dict, key: str, default=None):
        val = d.get(key, default)
        return val if val is not None else default

    @staticmethod
    def _build_result(
        props: dict,
        similarity: float,
        distance: Optional[float],
    ) -> Dict[str, Any]:
        """Shared result-dict builder for both search methods."""
        sg = LangChainVectorStore._safe_get
        text = sg(props, "text", "")
        return {
            "chunk_id":   sg(props, "chunk_id", ""),
            "text":       text,
            "score":      similarity,
            "similarity": similarity,
            "distance":   distance,
            "metadata": {
                "filename":         sg(props, "filename", ""),
                "file_type":        sg(props, "file_type", ""),
                "chunk_index":      sg(props, "chunk_index", 0),
                "total_chunks":     sg(props, "total_chunks", 0),
                "similarity_score": sg(props, "similarity_score", 0.0),
                "sentence_count":   sg(props, "sentence_count", 0),
                "model_type":       sg(props, "model_type", ""),
            },
            "text_length": sg(props, "text_length", len(text)),
            "model":       sg(props, "embedding_model", ""),
            "preview":     text[:200] + "..." if len(text) > 200 else text,
        }

    def _embed_query(self, query_text: str) -> List[float]:
        """
        Embed a query string.
        Raises RuntimeError if no embeddings model is configured.
        """
        if self.embeddings is None:
            raise RuntimeError("No embeddings model configured")
        return self.embeddings.embed_query(query_text)

    # ================================================================
    # SEMANTIC SEARCH
    # ================================================================

    def semantic_search(
        self,
        query_text: str,
        embedder=None,       # kept for interface compatibility; self.embeddings used
        top_k: int = 5,
        min_score: float = 0.5,
        filters: Optional[Dict] = None,
    ) -> List[Dict[str, Any]]:
        """
        Vector similarity search.
        `embedder` param is documented as unused;
        self.embeddings is always used for consistency.
        Guard uses _reconnect_if_needed().
        """
        if not self._reconnect_if_needed():
            logger.error("❌ Not connected to Weaviate")
            return []

        try:
            logger.info(f"🔍 Semantic search: '{query_text[:60]}'")

            query_vector = self._embed_query(query_text)
            collection = self.weaviate_client.collections.get(self.class_name)

            response = collection.query.near_vector(
                near_vector=query_vector,
                limit=top_k,
                return_metadata=MetadataQuery(distance=True, certainty=True),
                return_properties=self._RETURN_PROPERTIES,
            )

            if not getattr(response, "objects", None):
                logger.warning("⚠️  No objects returned from Weaviate")
                return []

            results: List[Dict[str, Any]] = []

            for obj in response.objects:
                try:
                    props = getattr(obj, "properties", {}) or {}
                    meta = getattr(obj, "metadata", None)

                    distance = getattr(meta, "distance", None) if meta else None
                    certainty = getattr(meta, "certainty", None) if meta else None

                    if certainty is not None:
                        similarity = float(certainty)
                    elif distance is not None:
                        similarity = max(0.0, 1.0 - float(distance))
                    else:
                        similarity = 0.0

                    if similarity < min_score:
                        continue

                    results.append(self._build_result(props, similarity, distance))

                except Exception as obj_err:
                    logger.warning(f"⚠️  Error parsing result object: {obj_err}")

            results.sort(key=lambda x: x["similarity"], reverse=True)
            logger.info(
                f"✅ Semantic: {len(results)} results (min_score={min_score})"
            )
            return results

        except Exception as e:
            logger.error(f"❌ Semantic search error: {e}", exc_info=True)
            return []

    # ================================================================
    # HYBRID SEARCH
    # ================================================================

    def hybrid_search(
        self,
        query_text: str,
        embedder=None,       # kept for interface compatibility
        top_k: int = 5,
        min_score: float = 0.5,
        alpha: float = 0.7,
        filters: Optional[Dict] = None,
    ) -> List[Dict[str, Any]]:
        """
        Hybrid (vector + BM25) search with semantic fallback.
        Guard uses _reconnect_if_needed().
        """
        if not self._reconnect_if_needed():
            logger.error("❌ Not connected to Weaviate")
            return []

        try:
            logger.info(f"🔍 Hybrid search (α={alpha}): '{query_text[:60]}'")

            query_vector = self._embed_query(query_text)
            collection = self.weaviate_client.collections.get(self.class_name)

            response = collection.query.hybrid(
                query=query_text,
                vector=query_vector,
                alpha=alpha,
                limit=top_k,
                return_metadata=MetadataQuery(score=True),
                return_properties=self._RETURN_PROPERTIES,
            )

            if not getattr(response, "objects", None):
                logger.warning("⚠️  No hybrid results — falling back to semantic")
                return self.semantic_search(
                    query_text, top_k=top_k, min_score=min_score
                )

            results: List[Dict[str, Any]] = []

            for obj in response.objects:
                try:
                    props = getattr(obj, "properties", {}) or {}
                    meta = getattr(obj, "metadata", None)
                    score = float(getattr(meta, "score", 0.0) or 0.0)

                    # Hybrid scores are not bounded [0,1]; skip min_score filter
                    results.append(self._build_result(props, score, None))

                except Exception as obj_err:
                    logger.warning(f"⚠️  Error parsing hybrid result: {obj_err}")

            results.sort(key=lambda x: x["similarity"], reverse=True)
            logger.info(f"✅ Hybrid: {len(results)} results")
            return results

        except Exception as e:
            logger.error(
                f"❌ Hybrid search error: {e} — falling back to semantic",
                exc_info=True,
            )
            return self.semantic_search(query_text, top_k=top_k, min_score=min_score)

    # ================================================================
    # UTILITY SEARCH
    # ================================================================

    def search_by_filename(
        self, filename: str, limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Return all chunks belonging to a specific file."""
        if not self._reconnect_if_needed():
            return []

        try:
            collection = self.weaviate_client.collections.get(self.class_name)
            response = collection.query.fetch_objects(
                limit=limit,
                filters=Filter.by_property("filename").equal(filename),
            )

            results = []
            for obj in getattr(response, "objects", []):
                props = getattr(obj, "properties", {}) or {}
                results.append({
                    "chunk_id": props.get("chunk_id"),
                    "text":     props.get("text"),
                    "metadata": {
                        "filename":     props.get("filename"),
                        "chunk_index":  props.get("chunk_index"),
                        "total_chunks": props.get("total_chunks"),
                    },
                    "chunk_index": props.get("chunk_index", 0),
                })

            results.sort(key=lambda x: x.get("chunk_index", 0))
            logger.info(f"✅ {len(results)} chunks for '{filename}'")
            return results

        except Exception as e:
            logger.error(f"❌ search_by_filename error: {e}", exc_info=True)
            return []

    def get_random_samples(self, count: int = 5) -> List[Dict[str, Any]]:
        """Return a handful of documents for debugging."""
        if not self._reconnect_if_needed():
            return []

        try:
            collection = self.weaviate_client.collections.get(self.class_name)
            response = collection.query.fetch_objects(limit=count)

            results = []
            for obj in getattr(response, "objects", []):
                props = getattr(obj, "properties", {}) or {}
                text = props.get("text", "")
                results.append({
                    "chunk_id": props.get("chunk_id"),
                    "text":     text,
                    "metadata": {
                        "filename":  props.get("filename"),
                        "file_type": props.get("file_type"),
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

            collection = self.weaviate_client.collections.get(self.class_name)
            count = collection.aggregate.over_all(total_count=True)

            return {
                "document_count":  count.total_count,
                "collection_name": self.class_name,
            }

        except Exception as e:
            logger.error(f"❌ get_stats error: {e}", exc_info=True)
            return {"document_count": 0}

    def delete_collection(self) -> bool:
        """Delete the Weaviate collection."""
        try:
            if self.weaviate_client.collections.exists(self.class_name):
                self.weaviate_client.collections.delete(self.class_name)
                logger.info(f"✅ Deleted collection: {self.class_name}")
                return True
            return False
        except Exception as e:
            logger.error(f"❌ delete_collection error: {e}", exc_info=True)
            return False

    def recreate_collection(self) -> bool:
        """Drop and recreate the collection (useful after schema changes)."""
        try:
            self.delete_collection()
            return self._ensure_collection()
        except Exception as e:
            logger.error(f"❌ recreate_collection error: {e}", exc_info=True)
            return False