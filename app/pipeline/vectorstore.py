"""
Vector Store Pipeline
LangChain-powered Weaviate integration

Replaces:
    - app/data/vectordb/weaviate_client.py
    - app/data/vectordb/vector_store.py
    - app/data/vectordb/search_engine.py
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone

import weaviate
from weaviate.auth import AuthApiKey
from weaviate.classes.config import Configure, Property, DataType
from langchain_weaviate import WeaviateVectorStore
from langchain.schema import Document

logger = logging.getLogger(__name__)


class LangChainVectorStore:
    """
    Unified Weaviate vector store using LangChain
    
    Replaces:
        - WeaviateClient (connection management)
        - VectorStore (storage operations)
        - SearchEngine (search operations)
        
    New features:
        - Automatic retry logic
        - Better error handling
        - Simplified API
        - Same interfaces as original
    """
    
    def __init__(
        self,
        url: str,
        api_key: str,
        class_name: str = "Ragdocument",
        embeddings=None,
        vector_dims: int = 1536
    ):
        """
        Initialize Vector Store
        
        Args:
            url: Weaviate Cloud URL
            api_key: Weaviate API key
            class_name: Collection name (same as WEAVIATE_CLASS_NAME)
            embeddings: LangChain embeddings instance
            vector_dims: Vector dimension (same as WEAVIATE_VECTOR_DIMS)
        """
        self.url = url
        self.api_key = api_key
        self.class_name = class_name
        self.embeddings = embeddings
        self.vector_dims = vector_dims
        
        # Connection state
        self.weaviate_client = None
        self.vectorstore = None
        
        # Connect
        self._connect()
    
    # ============================================================
    # CONNECTION (replaces WeaviateClient)
    # ============================================================
    
    def _connect(self):
        """
        Connect to Weaviate Cloud
        Same logic as WeaviateClient._connect()
        """
        try:
            logger.info("="*70)
            logger.info("🔌 Connecting to Weaviate Cloud")
            logger.info("="*70)
            
            # Add https:// if needed
            cluster_url = self.url
            if not cluster_url.startswith(('http://', 'https://')):
                cluster_url = f"https://{cluster_url}"
            
            logger.info(f"   URL: {cluster_url}")
            
            # Connect to Weaviate Cloud
            self.weaviate_client = weaviate.connect_to_wcs(
                cluster_url=cluster_url,
                auth_credentials=AuthApiKey(self.api_key),
                skip_init_checks=False
            )
            
            # Verify connection
            if self.weaviate_client.is_ready():
                logger.info("✅ Connected to Weaviate Cloud")
                
                # Ensure collection exists
                self._ensure_collection()
                
                # Initialize LangChain VectorStore wrapper
                if self.embeddings:
                    self._init_vectorstore()
                
                logger.info("="*70)
                
            else:
                logger.error("❌ Weaviate connected but not ready")
                self.weaviate_client = None
                
        except Exception as e:
            logger.error("="*70)
            logger.error("❌ Weaviate connection failed")
            logger.error("="*70)
            logger.error(f"Error: {str(e)}")
            logger.error("\nTroubleshooting:")
            logger.error("  1. Check WEAVIATE_URL in .env")
            logger.error("  2. Verify WEAVIATE_API_KEY")
            logger.error("  3. Check cluster status in Weaviate Console")
            logger.error("="*70)
            import traceback
            traceback.print_exc()
            self.weaviate_client = None
            self.vectorstore = None
    
    def _ensure_collection(self):
        """
        Ensure collection exists
        Same logic as WeaviateClient.create_collection()
        """
        try:
            if self.weaviate_client.collections.exists(self.class_name):
                logger.info(f"✅ Collection '{self.class_name}' exists")
                return True
            
            logger.info(f"📦 Creating collection '{self.class_name}'...")
            
            # Create collection with same schema as your WeaviateClient
            self.weaviate_client.collections.create(
                name=self.class_name,
                description="RAG document chunks with embeddings",
                vectorizer_config=Configure.Vectorizer.none(),
                properties=[
                    Property(
                        name="chunk_id",
                        data_type=DataType.TEXT,
                        description="Unique chunk identifier"
                    ),
                    Property(
                        name="text",
                        data_type=DataType.TEXT,
                        description="Chunk text content"
                    ),
                    Property(
                        name="text_length",
                        data_type=DataType.INT,
                        description="Text length in characters"
                    ),
                    Property(
                        name="embedding_model",
                        data_type=DataType.TEXT,
                        description="Embedding model name"
                    ),
                    Property(
                        name="model_type",
                        data_type=DataType.TEXT,
                        description="Model type"
                    ),
                    Property(
                        name="embedded_at",
                        data_type=DataType.DATE,
                        description="Embedding timestamp"
                    ),
                    Property(
                        name="indexed_at",
                        data_type=DataType.DATE,
                        description="Indexing timestamp"
                    ),
                    Property(
                        name="filename",
                        data_type=DataType.TEXT,
                        description="Source filename"
                    ),
                    Property(
                        name="file_type",
                        data_type=DataType.TEXT,
                        description="File extension"
                    ),
                    Property(
                        name="chunk_index",
                        data_type=DataType.INT,
                        description="Chunk position in document"
                    ),
                    Property(
                        name="total_chunks",
                        data_type=DataType.INT,
                        description="Total chunks in document"
                    ),
                    # Optional fields (skip_vectorization)
                    Property(
                        name="similarity_score",
                        data_type=DataType.NUMBER,
                        skip_vectorization=True
                    ),
                    Property(
                        name="sentence_count",
                        data_type=DataType.INT,
                        skip_vectorization=True
                    )
                ]
            )
            
            logger.info(f"✅ Collection '{self.class_name}' created")
            return True
            
        except Exception as e:
            logger.error(f"❌ Collection creation failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _init_vectorstore(self):
        """
        Initialize LangChain VectorStore wrapper
        Used ONLY for search, not for storage
        """
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
                    "similarity_score"
                ]
            )
            logger.info("✅ LangChain VectorStore initialized (search only)")
            
        except Exception as e:
            logger.warning(
                f"⚠️  VectorStore search wrapper failed: {e}"
                f" - Search may be limited"
            )
            self.vectorstore = None

    def is_connected(self) -> bool:
        """Same interface as WeaviateClient.is_connected()"""
        return (
            self.weaviate_client is not None and
            self.weaviate_client.is_ready()
        )
    
    def close(self):
        """Same interface as WeaviateClient.close()"""
        if self.weaviate_client:
            self.weaviate_client.close()
            logger.info("✅ Weaviate connection closed")
    
    # ============================================================
    # STORAGE (replaces VectorStore)
    # ============================================================
    
    def store_batch(
        self,
        documents: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Store multiple documents in Weaviate
        Same interface as VectorStore.store_batch()
        
        Args:
            documents: List of documents with embedded_chunks
            
        Returns:
            Same format as VectorStore.store_batch()
        """
        if not self.is_connected() or not self.vectorstore:
            logger.error("❌ Not connected to Weaviate")
            return {
                'documents_processed': 0,
                'chunks_stored': 0,
                'chunks_failed': len(documents)
            }
        
        logger.info("="*70)
        logger.info(f"💾 Storing batch: {len(documents)} documents")
        logger.info("="*70)
        
        total_stored = 0
        total_failed = 0
        docs_processed = 0
        
        for doc in documents:
            if not doc.get('embedding_complete'):
                logger.warning(
                    f"⚠️  Skipping {doc.get('filename')}: not embedded"
                )
                continue
            
            result = self._store_document(doc)
            total_stored += result['success']
            total_failed += result['failed']
            
            if result['success'] > 0:
                docs_processed += 1
        
        logger.info("="*70)
        logger.info("✅ STORAGE COMPLETE")
        logger.info(f"   Documents : {docs_processed}")
        logger.info(f"   Stored    : {total_stored} chunks")
        logger.info(f"   Failed    : {total_failed} chunks")
        logger.info("="*70)
        
        return {
            'documents_processed': docs_processed,
            'chunks_stored': total_stored,
            'chunks_failed': total_failed
        }
    def _store_document(
        self,
        document_data: Dict[str, Any]
    ) -> Dict[str, int]:
        """Store single document using native Weaviate client"""
        embedded_chunks = document_data.get('embedded_chunks', [])
        
        if not embedded_chunks:
            return {'success': 0, 'failed': 0}
        
        filename = document_data.get('filename', 'unknown')
        logger.info(f"💾 Storing: {filename} ({len(embedded_chunks)} chunks)")
        
        try:
            # ── Use native Weaviate client directly ───────────────
            # langchain-weaviate==0.0.1 does not have add_embeddings
            # Use weaviate_client.collections directly instead
            collection = self.weaviate_client.collections.get(self.class_name)
            
            success_count = 0
            failed_count = 0
            
            # ── Batch insert ──────────────────────────────────────
            from weaviate.classes.data import DataObject
            
            batch_objects = []
            
            for i, chunk in enumerate(embedded_chunks):
                
                # ── Safe type conversions ──────────────────────────
                similarity_score = chunk.get('similarity_score')
                similarity_score = (
                    float(similarity_score)
                    if similarity_score is not None
                    else 0.0
                )
                
                sentence_count = chunk.get('sentence_count')
                sentence_count = (
                    int(sentence_count)
                    if sentence_count is not None
                    else 0
                )
                
                text_length = chunk.get('text_length')
                text_length = (
                    int(text_length)
                    if text_length is not None
                    else len(chunk.get('text', ''))
                )
                
                # ── Format timestamps for Weaviate ─────────────────
                now_str = datetime.now(timezone.utc).strftime(
                    "%Y-%m-%dT%H:%M:%S.000Z"
                )
                
                embedded_at = chunk.get('embedded_at', now_str)
                # Ensure proper RFC3339 format
                if embedded_at and 'T' in embedded_at:
                    if not embedded_at.endswith('Z'):
                        embedded_at = embedded_at.split('.')[0] + ".000Z"
                else:
                    embedded_at = now_str
                
                # ── Build properties ───────────────────────────────
                properties = {
                    'chunk_id':        chunk.get('chunk_id', ''),
                    'text':            chunk.get('text', ''),
                    'filename':        filename,
                    'file_type':       document_data.get(
                                        'metadata', {}
                                    ).get('file_type', ''),
                    'chunk_index':     i,
                    'total_chunks':    len(embedded_chunks),
                    'embedding_model': chunk.get('embedding_model', ''),
                    'model_type':      chunk.get('model_type', ''),
                    'text_length':     text_length,
                    'embedded_at':     embedded_at,
                    'indexed_at':      now_str,
                    'similarity_score': similarity_score,
                    'sentence_count':  sentence_count,
                }
                
                # ── Get embedding vector ───────────────────────────
                embedding = chunk.get('embedding', [])
                
                if not embedding:
                    logger.warning(
                        f"⚠️  Empty embedding for chunk {i}, skipping"
                    )
                    failed_count += 1
                    continue
                
                batch_objects.append(
                    DataObject(
                        properties=properties,
                        vector=embedding
                    )
                )
            
            # ── Send batch to Weaviate ─────────────────────────────
            if batch_objects:
                with self.weaviate_client.batch.dynamic() as batch:
                    for obj in batch_objects:
                        batch.add_object(
                            collection=self.class_name,
                            properties=obj.properties,
                            vector=obj.vector
                        )
                
                # Check for errors
                if self.weaviate_client.batch.failed_objects:
                    failed = len(self.weaviate_client.batch.failed_objects)
                    success_count = len(batch_objects) - failed
                    failed_count += failed
                    
                    for err in self.weaviate_client.batch.failed_objects[:3]:
                        logger.error(f"   Batch error: {err}")
                else:
                    success_count = len(batch_objects)
            
            logger.info(
                f"✅ {filename}: {success_count} chunks stored, "
                f"{failed_count} failed"
            )
            
            return {
                'success': success_count,
                'failed': failed_count
            }
            
        except Exception as e:
            logger.error(f"❌ Storage error for {filename}: {e}")
            import traceback
            traceback.print_exc()
            return {
                'success': 0,
                'failed': len(embedded_chunks)
            }
 
    # ============================================================
    # SEARCH (replaces SearchEngine)
    # ============================================================
    
    def semantic_search(
            self,
            query_text: str,
            embedder,
            top_k: int = 5,
            min_score: float = 0.5,
            filters: Optional[Dict] = None
        ) -> List[Dict[str, Any]]:
            """
            Semantic search using native Weaviate client directly
            Bypasses langchain-weaviate deserialization bug
            """
            if not self.is_connected():
                logger.error("❌ Not connected to Weaviate")
                return []
            
            if not self.embeddings:
                logger.error("❌ No embeddings model available")
                return []
            
            try:
                logger.info(f"🔍 Searching: '{query_text[:50]}...'")
                
                # ── Step 1: Embed the query ────────────────────────
                query_vector = self.embeddings.embed_query(query_text)
                
                # ── Step 2: Native Weaviate vector search ──────────
                from weaviate.classes.query import MetadataQuery
                
                collection = self.weaviate_client.collections.get(
                    self.class_name
                )
                
                response = collection.query.near_vector(
                    near_vector=query_vector,
                    limit=top_k,
                    return_metadata=MetadataQuery(
                        distance=True,
                        certainty=True
                    ),
                    return_properties=[
                        "chunk_id",
                        "text",
                        "filename",
                        "file_type",
                        "chunk_index",
                        "total_chunks",
                        "embedding_model",
                        "model_type",
                        "text_length",
                        "sentence_count",
                        "similarity_score"
                    ]
                )
                
                # ── Step 3: Parse results ──────────────────────────
                results = []
                
                if not hasattr(response, 'objects') or not response.objects:
                    logger.warning("⚠️  No objects returned from Weaviate")
                    return []
                
                for obj in response.objects:
                    try:
                        props = (
                            obj.properties
                            if hasattr(obj, 'properties') else {}
                        )
                        meta = (
                            obj.metadata
                            if hasattr(obj, 'metadata') else None
                        )
                        
                        # ── Distance → similarity ──────────────────
                        distance   = None
                        similarity = 0.0
                        certainty  = None
                        
                        if meta:
                            distance  = getattr(meta, 'distance',  None)
                            certainty = getattr(meta, 'certainty', None)
                        
                        if certainty is not None:
                            similarity = float(certainty)
                        elif distance is not None:
                            similarity = max(0.0, 1.0 - float(distance))
                        
                        # ── Filter by min_score ────────────────────
                        if similarity < min_score:
                            logger.debug(
                                f"   Skipping (score {similarity:.3f} "
                                f"< {min_score})"
                            )
                            continue
                        
                        # ── Safe property getter ───────────────────
                        def safe_get(d, key, default=None):
                            val = d.get(key, default)
                            return val if val is not None else default
                        
                        chunk_text = safe_get(props, 'text', '')
                        
                        results.append({
                            'chunk_id':   safe_get(props, 'chunk_id', ''),
                            'text':       chunk_text,
                            'score':      similarity,
                            'similarity': similarity,
                            'distance':   distance,
                            'metadata': {
                                'filename':     safe_get(props, 'filename', ''),
                                'file_type':    safe_get(props, 'file_type', ''),
                                'chunk_index':  safe_get(props, 'chunk_index', 0),
                                'total_chunks': safe_get(props, 'total_chunks', 0),
                                'similarity_score': safe_get(
                                    props, 'similarity_score', 0.0
                                ),
                                'sentence_count': safe_get(
                                    props, 'sentence_count', 0
                                ),
                                'model_type': safe_get(props, 'model_type', ''),
                            },
                            'text_length': safe_get(
                                props, 'text_length', len(chunk_text)
                            ),
                            'model':   safe_get(props, 'embedding_model', ''),
                            'preview': (
                                chunk_text[:200] + "..."
                                if len(chunk_text) > 200
                                else chunk_text
                            )
                        })
                        
                    except Exception as obj_err:
                        logger.warning(
                            f"⚠️  Error parsing result: {obj_err}"
                        )
                        continue
                
                # Sort by similarity
                results.sort(key=lambda x: x['similarity'], reverse=True)
                
                logger.info(
                    f"✅ Found {len(results)} results "
                    f"(min_score={min_score})"
                )
                
                for i, r in enumerate(results[:3], 1):
                    logger.debug(
                        f"   #{i} score={r['similarity']:.3f} "
                        f"file={r['metadata']['filename']}"
                    )
                
                return results
                
            except Exception as e:
                logger.error(f"❌ Search error: {e}")
                import traceback
                traceback.print_exc()
                return []
            

    def hybrid_search(
        self,
        query_text: str,
        embedder,
        top_k: int = 5,
        min_score: float = 0.5,
        alpha: float = 0.7,
        filters: Optional[Dict] = None
    ) -> List[Dict[str, Any]]:
        """
        Hybrid search using native Weaviate client
        Bypasses langchain-weaviate deserialization bug
        """
        if not self.is_connected():
            logger.error("❌ Not connected to Weaviate")
            return []
        
        try:
            logger.info(
                f"🔍 Hybrid search (alpha={alpha}): "
                f"'{query_text[:50]}...'"
            )
            
            # ── Embed query ────────────────────────────────────
            query_vector = self.embeddings.embed_query(query_text)
            
            # ── Native hybrid search ───────────────────────────
            from weaviate.classes.query import MetadataQuery
            
            collection = self.weaviate_client.collections.get(
                self.class_name
            )
            
            response = collection.query.hybrid(
                query=query_text,
                vector=query_vector,
                alpha=alpha,
                limit=top_k,
                return_metadata=MetadataQuery(score=True),
                return_properties=[
                    "chunk_id",
                    "text",
                    "filename",
                    "file_type",
                    "chunk_index",
                    "total_chunks",
                    "embedding_model",
                    "model_type",
                    "text_length",
                    "sentence_count",
                    "similarity_score"
                ]
            )
            
            # ── Parse results ──────────────────────────────────
            results = []
            
            if not hasattr(response, 'objects') or not response.objects:
                logger.warning("⚠️  No hybrid results, falling back...")
                return self.semantic_search(
                    query_text=query_text,
                    embedder=embedder,
                    top_k=top_k,
                    min_score=min_score
                )
            
            for obj in response.objects:
                try:
                    props = (
                        obj.properties
                        if hasattr(obj, 'properties') else {}
                    )
                    meta = (
                        obj.metadata
                        if hasattr(obj, 'metadata') else None
                    )
                    
                    hybrid_score = 0.0
                    if meta:
                        score = getattr(meta, 'score', None)
                        if score is not None:
                            hybrid_score = float(score)
                    
                    def safe_get(d, key, default=None):
                        val = d.get(key, default)
                        return val if val is not None else default
                    
                    chunk_text = safe_get(props, 'text', '')
                    
                    results.append({
                        'chunk_id':   safe_get(props, 'chunk_id', ''),
                        'text':       chunk_text,
                        'score':      hybrid_score,
                        'similarity': hybrid_score,
                        'distance':   None,
                        'metadata': {
                            'filename':     safe_get(props, 'filename', ''),
                            'file_type':    safe_get(props, 'file_type', ''),
                            'chunk_index':  safe_get(props, 'chunk_index', 0),
                            'total_chunks': safe_get(props, 'total_chunks', 0),
                            'similarity_score': safe_get(
                                props, 'similarity_score', 0.0
                            ),
                            'sentence_count': safe_get(
                                props, 'sentence_count', 0
                            ),
                            'model_type': safe_get(props, 'model_type', ''),
                        },
                        'text_length': safe_get(
                            props, 'text_length', len(chunk_text)
                        ),
                        'model':   safe_get(props, 'embedding_model', ''),
                        'preview': (
                            chunk_text[:200] + "..."
                            if len(chunk_text) > 200
                            else chunk_text
                        )
                    })
                    
                except Exception as obj_err:
                    logger.warning(
                        f"⚠️  Error parsing hybrid result: {obj_err}"
                    )
                    continue
            
            results.sort(key=lambda x: x['similarity'], reverse=True)
            logger.info(f"✅ Hybrid found {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"❌ Hybrid search error: {e}")
            # Always fallback to semantic
            return self.semantic_search(
                query_text=query_text,
                embedder=embedder,
                top_k=top_k,
                min_score=min_score
            )
        
    def search_by_filename(
        self,
        filename: str,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get all chunks from a specific file
        Same interface as SearchEngine.search_by_filename()
        """
        if not self.is_connected():
            return []
        
        try:
            from weaviate.classes.query import Filter
            
            collection = self.weaviate_client.collections.get(
                self.class_name
            )
            
            response = collection.query.fetch_objects(
                limit=limit,
                filters=Filter.by_property("filename").equal(filename)
            )
            
            results = []
            if hasattr(response, 'objects'):
                for obj in response.objects:
                    props = obj.properties if hasattr(obj, 'properties') else {}
                    results.append({
                        'chunk_id': props.get('chunk_id'),
                        'text': props.get('text'),
                        'metadata': {
                            'filename': props.get('filename'),
                            'chunk_index': props.get('chunk_index'),
                            'total_chunks': props.get('total_chunks')
                        },
                        'chunk_index': props.get('chunk_index', 0)
                    })
            
            # Sort by chunk_index
            results.sort(key=lambda x: x.get('chunk_index', 0))
            
            logger.info(f"✅ Found {len(results)} chunks for '{filename}'")
            return results
            
        except Exception as e:
            logger.error(f"❌ Search by filename error: {e}")
            return []
    
    def get_random_samples(self, count: int = 5) -> List[Dict[str, Any]]:
        """
        Get random sample documents
        Same interface as SearchEngine.get_random_samples()
        """
        if not self.is_connected():
            return []
        
        try:
            collection = self.weaviate_client.collections.get(
                self.class_name
            )
            
            response = collection.query.fetch_objects(limit=count)
            
            results = []
            if hasattr(response, 'objects'):
                for obj in response.objects:
                    props = obj.properties if hasattr(obj, 'properties') else {}
                    results.append({
                        'chunk_id': props.get('chunk_id'),
                        'text': props.get('text'),
                        'metadata': {
                            'filename': props.get('filename'),
                            'file_type': props.get('file_type')
                        },
                        'preview': props.get('text', '')[:150] + "..."
                    })
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Random samples error: {e}")
            return []
    
    # ============================================================
    # STATISTICS & ADMIN
    # ============================================================
    
    def get_stats(self) -> Dict[str, Any]:
        """Same interface as VectorStore.get_stats()"""
        try:
            if not self.is_connected():
                return {'document_count': 0}
            
            collection = self.weaviate_client.collections.get(
                self.class_name
            )
            count = collection.aggregate.over_all(total_count=True)
            
            return {
                'document_count': count.total_count,
                'collection_name': self.class_name
            }
            
        except Exception as e:
            logger.error(f"❌ Stats error: {e}")
            return {'document_count': 0}
    
    def delete_collection(self) -> bool:
        """
        Delete the collection
        Same interface as WeaviateClient.delete_collection()
        """
        try:
            if self.weaviate_client.collections.exists(self.class_name):
                self.weaviate_client.collections.delete(self.class_name)
                logger.info(f"✅ Deleted collection: {self.class_name}")
                return True
            return False
        except Exception as e:
            logger.error(f"❌ Delete error: {e}")
            return False
    
    def recreate_collection(self) -> bool:
        """
        Delete and recreate collection
        Useful for schema changes
        """
        try:
            # Delete if exists
            if self.weaviate_client.collections.exists(self.class_name):
                self.delete_collection()
            
            # Recreate
            return self._ensure_collection()
            
        except Exception as e:
            logger.error(f"❌ Recreate error: {e}")
            return False