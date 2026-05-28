"""
Embeddings Pipeline - multilingual-e5-small
LangChain-powered embedding generation with multilingual-e5-small
"""

import logging
import hashlib
import json
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from langchain.embeddings.base import Embeddings
from langchain.embeddings.cache import CacheBackedEmbeddings
from langchain.storage import LocalFileStore

logger = logging.getLogger(__name__)


# ============================================================
# MULTILINGUAL-E5-SMALL EMBEDDINGS
# ============================================================

class E5Embeddings(Embeddings):
    """
    multilingual-e5-small embeddings using sentence-transformers
    
    Features:
        - Multi-lingual (100+ languages)
        - 384 dimensions
        - Max 512 tokens
        - Offline mode
        - Very Low RAM (~470MB)
        - IMPORTANT: Requires "query: " / "passage: " prefixes
    """
    
    def __init__(
        self,
        model_path: str,
        device: str = 'cpu',
        batch_size: int = 32,          # small model → bigger batches OK
        max_length: int = 512,
        normalize_embeddings: bool = True
    ):
        self.model_path = model_path
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length
        self.normalize_embeddings = normalize_embeddings
        self.model = None
        
        self._load_model()
    
    def _get_ram_usage(self) -> float:
        """Get RAM usage in GB"""
        try:
            import psutil
            process = psutil.Process(os.getpid())
            return process.memory_info().rss / (1024 ** 3)
        except Exception:
            return 0.0
    
    def _load_model(self):
        """Load multilingual-e5-small - offline mode"""
        try:
            from sentence_transformers import SentenceTransformer
            import sentence_transformers
            
            logger.info(
                f"📦 sentence-transformers: "
                f"{sentence_transformers.__version__}"
            )
            logger.info(f"⏳ Loading E5 from: {self.model_path}")
            logger.info(f"   Device: {self.device}")
            logger.info(f"   RAM before: {self._get_ram_usage():.1f} GB")
            
            os.environ['TRANSFORMERS_OFFLINE'] = '1'
            os.environ['HF_DATASETS_OFFLINE'] = '1'
            os.environ['HF_HUB_OFFLINE'] = '1'
            
            self.model = SentenceTransformer(
                self.model_path,
                device=self.device
            )
            
            self.model.max_seq_length = self.max_length
            dim = self.model.get_sentence_embedding_dimension()
            
            logger.info(f"✅ E5 loaded | RAM: {self._get_ram_usage():.1f} GB")
            logger.info(f"   Dimension: {dim}")
            logger.info(f"   Max length: {self.max_length}")
            logger.info(f"   NOTE: Using 'passage: ' / 'query: ' prefixes")
            
        except ImportError:
            logger.error("❌ sentence-transformers not installed!")
            logger.error("   pip install sentence-transformers")
            raise
        except Exception as e:
            logger.error(f"❌ Model load failed: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    # ── E5 REQUIRES PREFIXES ──────────────────────────────────
    # Documents get "passage: " prefix
    # Queries   get "query: "   prefix
    # This is NOT optional - skipping degrades quality significantly
    
    def _add_passage_prefix(self, texts: List[str]) -> List[str]:
        """Add 'passage: ' prefix for documents"""
        return [
            f"passage: {t}" if not t.startswith("passage: ") else t
            for t in texts
        ]
    
    def _add_query_prefix(self, text: str) -> str:
        """Add 'query: ' prefix for queries"""
        if not text.startswith("query: "):
            return f"query: {text}"
        return text
    
    # ─────────────────────────────────────────────────────────
    
    def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed batch with cleanup"""
        import gc
        
        try:
            embeddings = self.model.encode(
                texts,
                batch_size=self.batch_size,
                show_progress_bar=False,
                normalize_embeddings=self.normalize_embeddings,
                convert_to_numpy=True
            )
            
            result = embeddings.tolist()
            del embeddings
            gc.collect()
            return result
            
        except Exception as e:
            logger.error(f"❌ Batch embed error: {e}")
            raise
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed multiple documents.
        Automatically prepends 'passage: ' prefix (E5 requirement).
        """
        if not texts:
            return []
        
        # Add passage prefix - required for E5 models
        prefixed_texts = self._add_passage_prefix(texts)
        
        all_embeddings = []
        total_batches = (
            len(prefixed_texts) + self.batch_size - 1
        ) // self.batch_size
        
        for i in range(0, len(prefixed_texts), self.batch_size):
            batch = prefixed_texts[i:i + self.batch_size]
            batch_num = i // self.batch_size + 1
            logger.debug(
                f"Embedding batch {batch_num}/{total_batches} "
                f"({len(batch)} texts)"
            )
            all_embeddings.extend(self._embed_batch(batch))
        
        return all_embeddings
    
    def embed_query(self, text: str) -> List[float]:
        """
        Embed single query.
        Automatically prepends 'query: ' prefix (E5 requirement).
        """
        prefixed = self._add_query_prefix(text)
        return self._embed_batch([prefixed])[0]


# ============================================================
# EMBEDDING MANAGER
# ============================================================

class LangChainEmbeddingManager:
    """
    Unified embedding manager - multilingual-e5-small + LangChain cache
    """
    
    def __init__(
        self,
        model_name: str = "./models/multilingual-e5-small",  # ← CHANGED
        use_gpu: bool = False,
        batch_size: int = 32,                                 # ← CHANGED (small model)
        cache_dir: str = "./embeddings_cache",
        output_dir: str = "./embeddings_output",
        save_intermediate: bool = True,
        max_workers: int = 1
    ):
        self.model_name = model_name
        self.use_gpu = use_gpu
        self.batch_size = batch_size
        self.cache_dir = Path(cache_dir)
        self.output_dir = Path(output_dir)
        self.save_intermediate = save_intermediate
        self.max_workers = max_workers
        
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self._init_embeddings()
        
        self.embedding_dim = self._get_embedding_dim()
        self._model_loaded = self.base_embeddings is not None
        
        logger.info("="*70)
        logger.info("✅ LangChainEmbeddingManager initialized")
        logger.info(f"   Model      : {model_name}")
        logger.info(f"   Type       : multilingual-e5-small")   # ← CHANGED
        logger.info(f"   Device     : {'GPU' if use_gpu else 'CPU'}")
        logger.info(f"   Dimension  : {self.embedding_dim}")
        logger.info(f"   Batch size : {batch_size}")
        logger.info(f"   Cache dir  : {cache_dir}")
        logger.info(f"   Prefixes   : query / passage (auto)")    # ← NEW
        logger.info("="*70)
    
    def _resolve_model_path(self) -> str:
        """Resolve model path"""
        model_path = self.model_name
        
        if model_path.startswith('./') or model_path.startswith('.\\'):
            absolute_path = Path(model_path).resolve()
            if absolute_path.exists():
                logger.info(f"✅ Model found: {absolute_path}")
                return str(absolute_path)
            
            model_name_only = Path(model_path).name
            
            candidates = [
                Path("app/scripts/models") / model_name_only,
                Path("scripts/models") / model_name_only,
                Path("models") / model_name_only,
            ]
            
            for candidate in candidates:
                if candidate.resolve().exists():
                    logger.info(f"✅ Model found: {candidate.resolve()}")
                    return str(candidate.resolve())
            
            logger.error(f"❌ Model not found: {model_path}")
            for candidate in [absolute_path] + candidates:
                logger.error(f"   Tried: {candidate.resolve()}")
            
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        return model_path
    
    def _init_embeddings(self):
        """Initialize E5 with cache"""
        try:
            model_path = self._resolve_model_path()
            device = 'cuda' if self.use_gpu else 'cpu'
            
            # ← CHANGED: E5Embeddings instead of BGEEmbeddings
            self.base_embeddings = E5Embeddings(
                model_path=model_path,
                device=device,
                batch_size=self.batch_size,
                max_length=512,
                normalize_embeddings=True
            )
            
            cache_store = LocalFileStore(str(self.cache_dir / "vectors"))
            
            safe_namespace = (
                model_path
                .replace('/', '_')
                .replace('\\', '_')
                .replace('.', '_')
                .replace(':', '_')
            )
            
            self.embeddings = CacheBackedEmbeddings.from_bytes_store(
                self.base_embeddings,
                cache_store,
                namespace=safe_namespace
            )
            
            logger.info("✅ E5 embeddings initialized with cache")
            
        except FileNotFoundError as e:
            logger.error(f"❌ Model not found: {e}")
            self.base_embeddings = None
            self.embeddings = None
            
        except Exception as e:
            logger.error(f"❌ Init failed: {e}")
            import traceback
            traceback.print_exc()
            self.base_embeddings = None
            self.embeddings = None
    
    def _get_embedding_dim(self) -> int:
        """Get dimension"""
        try:
            test_embedding = self.base_embeddings.embed_query("test")
            dim = len(test_embedding)
            logger.info(f"✅ Dimension detected: {dim}")
            return dim
        except Exception:
            model_lower = self.model_name.lower()
            # ← CHANGED: E5 dimension fallbacks
            if 'e5-small' in model_lower:
                return 384
            elif 'e5-base' in model_lower:
                return 768
            elif 'e5-large' in model_lower:
                return 1024
            else:
                return 384   # default for e5-small
    
    # ── Everything below is IDENTICAL to original ─────────────
    # (is_ready, embed_query, embed_single_document,
    #  embed_multiple_documents, embed_chunks_only,
    #  get_system_info, clear_cache, _embed_document,
    #  _process_sequential, _process_parallel,
    #  _generate_chunk_id, _get_embedding_stats,
    #  _save_document_embeddings)
    # Only update model_type strings ─────────────────────────

    def is_ready(self) -> bool:
        return (
            self._model_loaded and
            self.base_embeddings is not None and
            self.embeddings is not None
        )
    
    def embed_query(self, text: str) -> List[float]:
        if not self.is_ready():
            raise RuntimeError("Embedding manager not ready")
        return self.base_embeddings.embed_query(text)
    
    def embed_single_document(
        self,
        document_data: Dict[str, Any],
        show_progress: bool = True
    ) -> Dict[str, Any]:
        if not self.is_ready():
            raise RuntimeError("Embedding manager not ready")
        return self._embed_document(document_data)
    
    def embed_multiple_documents(
        self,
        documents: List[Dict[str, Any]],
        parallel: bool = False,
        show_progress: bool = True
    ) -> Dict[str, Any]:
        if not self.is_ready():
            raise RuntimeError("Embedding manager not ready")
        
        logger.info("="*70)
        logger.info(f"📦 Embedding batch: {len(documents)} documents")
        logger.info("="*70)
        
        results = {
            'total_documents': len(documents),
            'successful': 0,
            'failed': 0,
            'total_chunks_embedded': 0,
            'documents': [],
            'errors': [],
            'processing_time': None,
            'statistics': {}
        }
        
        start_time = datetime.now()
        
        if parallel and self.max_workers > 1:
            results = self._process_parallel(documents, results)
        else:
            results = self._process_sequential(
                documents, results, show_progress
            )
        
        duration = (datetime.now() - start_time).total_seconds()
        results['processing_time'] = duration
        results['statistics'] = {
            'success_rate': (
                results['successful'] / results['total_documents'] * 100
                if results['total_documents'] > 0 else 0
            ),
            'avg_chunks_per_document': (
                results['total_chunks_embedded'] / results['successful']
                if results['successful'] > 0 else 0
            ),
            'avg_processing_time': (
                duration / results['total_documents']
                if results['total_documents'] > 0 else 0
            ),
            'chunks_per_second': (
                results['total_chunks_embedded'] / duration
                if duration > 0 else 0
            )
        }
        
        logger.info("="*70)
        logger.info("✅ BATCH COMPLETE")
        logger.info(f"   Total  : {results['total_documents']}")
        logger.info(f"   Success: {results['successful']}")
        logger.info(f"   Failed : {results['failed']}")
        logger.info(f"   Chunks : {results['total_chunks_embedded']}")
        logger.info(f"   Time   : {duration:.2f}s")
        logger.info(
            f"   Speed  : "
            f"{results['statistics']['chunks_per_second']:.1f} chunks/s"
        )
        logger.info("="*70)
        
        return results
    
    def embed_chunks_only(
        self,
        chunks: List[str],
        chunks_metadata: Optional[List[Dict[str, Any]]] = None,
        show_progress: bool = True
    ) -> List[Dict[str, Any]]:
        if not self.is_ready():
            raise RuntimeError("Embedding manager not ready")
        
        if not chunks:
            return []
        
        logger.info(f"📦 Embedding {len(chunks)} chunks...")
        
        try:
            embeddings = self.embeddings.embed_documents(chunks)
            
            embedded_chunks = []
            for i, (chunk_text, embedding) in enumerate(
                zip(chunks, embeddings)
            ):
                chunk_id = self._generate_chunk_id(chunk_text)
                metadata = (
                    chunks_metadata[i]
                    if chunks_metadata and i < len(chunks_metadata)
                    else {}
                )
                
                embedded_chunks.append({
                    'chunk_id': chunk_id,
                    'text': chunk_text,
                    'embedding': embedding,
                    'embedding_dim': len(embedding),
                    'text_length': len(chunk_text),
                    'embedded_at': datetime.now().isoformat(),
                    'embedding_model': self.model_name,
                    'model_type': 'multilingual-e5-small',  # ← CHANGED
                    'metadata': metadata
                })
            
            logger.info(f"✅ Embedded {len(embedded_chunks)} chunks")
            return embedded_chunks
            
        except Exception as e:
            logger.error(f"❌ Embedding error: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def get_system_info(self) -> Dict[str, Any]:
        return {
            'model_name': self.model_name,
            'model_type': 'multilingual-e5-small',          # ← CHANGED
            'embedding_dim': self.embedding_dim,
            'device': 'GPU' if self.use_gpu else 'CPU',
            'batch_size': self.batch_size,
            'is_ready': self.is_ready(),
            'cache_enabled': True,
            'output_dir': str(self.output_dir),
            'cache_dir': str(self.cache_dir),
            'features': [
                'multilingual',
                'dense-retrieval',
                'normalized-embeddings',
                'cache-backed',
                'offline-mode',
                'query-passage-prefixes',                    # ← NEW
                'lightweight-384d'                           # ← NEW
            ]
        }
    
    def clear_cache(self):
        try:
            cache_path = self.cache_dir / "vectors"
            if cache_path.exists():
                import shutil
                shutil.rmtree(cache_path)
                cache_path.mkdir(parents=True)
                logger.info("✅ Cache cleared")
        except Exception as e:
            logger.warning(f"⚠️  Cache clear error: {e}")
    
    # ── Private ───────────────────────────────────────────────
    
    def _embed_document(
        self, document_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        if not document_data.get('success'):
            logger.warning(
                f"⚠️  Skipping failed: {document_data.get('filename')}"
            )
            return document_data
        
        filename = document_data.get('filename', 'unknown')
        logger.info(f"📄 Embedding: {filename}")
        
        try:
            chunks_with_metadata = document_data.get(
                'chunks_with_metadata', []
            )
            
            if not chunks_with_metadata:
                plain_chunks = document_data.get('chunks', [])
                if plain_chunks:
                    chunks_with_metadata = [
                        {'text': chunk} for chunk in plain_chunks
                    ]
                else:
                    logger.warning(f"⚠️  No chunks in {filename}")
                    return document_data
            
            texts = [
                c.get('text', '')
                for c in chunks_with_metadata
                if c.get('text')
            ]
            
            if not texts:
                logger.warning(f"⚠️  No valid texts in {filename}")
                return document_data
            
            embeddings = self.embeddings.embed_documents(texts)
            
            embedded_chunks = []
            for chunk_dict, embedding in zip(
                chunks_with_metadata, embeddings
            ):
                chunk_text = chunk_dict.get('text', '')
                chunk_id = self._generate_chunk_id(chunk_text)
                
                embedded_chunks.append({
                    **chunk_dict,
                    'chunk_id': chunk_id,
                    'embedding': embedding,
                    'embedding_dim': len(embedding),
                    'text_length': len(chunk_text),
                    'embedded_at': datetime.now().isoformat(),
                    'embedding_model': self.model_name,
                    'model_type': 'multilingual-e5-small'  # ← CHANGED
                })
            
            embedding_stats = self._get_embedding_stats(embedded_chunks)
            
            result = {
                **document_data,
                'embedded_chunks': embedded_chunks,
                'embedding_stats': embedding_stats,
                'embedding_complete': True,
                'embedding_timestamp': datetime.now().isoformat()
            }
            
            if self.save_intermediate:
                self._save_document_embeddings(result)
            
            logger.info(
                f"✅ {filename}: {len(embedded_chunks)} chunks "
                f"({self.embedding_dim}D)"
            )
            return result
            
        except Exception as e:
            logger.error(f"❌ Error embedding {filename}: {e}")
            import traceback
            traceback.print_exc()
            document_data['embedding_error'] = str(e)
            document_data['embedding_complete'] = False
            return document_data
    
    def _process_sequential(
        self,
        documents: List[Dict[str, Any]],
        results: Dict[str, Any],
        show_progress: bool
    ) -> Dict[str, Any]:
        for i, doc in enumerate(documents, 1):
            if show_progress:
                logger.info(
                    f"📄 Processing {i}/{len(documents)}: "
                    f"{doc.get('filename', 'unknown')}"
                )
            
            result = self._embed_document(doc)
            results['documents'].append(result)
            
            if result.get('embedding_complete'):
                results['successful'] += 1
                results['total_chunks_embedded'] += len(
                    result.get('embedded_chunks', [])
                )
            else:
                results['failed'] += 1
                if 'embedding_error' in result:
                    results['errors'].append({
                        'filename': result.get('filename'),
                        'error': result['embedding_error']
                    })
        
        return results
    
    def _process_parallel(
        self,
        documents: List[Dict[str, Any]],
        results: Dict[str, Any]
    ) -> Dict[str, Any]:
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_doc = {
                executor.submit(self._embed_document, doc): doc
                for doc in documents
            }
            
            for future in as_completed(future_to_doc):
                try:
                    result = future.result()
                    results['documents'].append(result)
                    
                    if result.get('embedding_complete'):
                        results['successful'] += 1
                        results['total_chunks_embedded'] += len(
                            result.get('embedded_chunks', [])
                        )
                    else:
                        results['failed'] += 1
                        if 'embedding_error' in result:
                            results['errors'].append({
                                'filename': result.get('filename'),
                                'error': result['embedding_error']
                            })
                        
                except Exception as e:
                    logger.error(f"❌ Parallel error: {e}")
                    results['failed'] += 1
        
        return results
    
    def _generate_chunk_id(self, text: str) -> str:
        return hashlib.md5(text.encode('utf-8')).hexdigest()
    
    def _get_embedding_stats(
        self,
        embedded_chunks: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        if not embedded_chunks:
            return {}
        
        embedding_arrays = [
            np.array(c['embedding']) for c in embedded_chunks
        ]
        text_lengths = [
            c.get('text_length', len(c.get('text', '')))
            for c in embedded_chunks
        ]
        
        return {
            'total_chunks': len(embedded_chunks),
            'embedding_dim': embedded_chunks[0].get('embedding_dim', 0),
            'model_name': self.model_name,
            'model_type': 'multilingual-e5-small',         # ← CHANGED
            'text_stats': {
                'min_length': min(text_lengths) if text_lengths else 0,
                'max_length': max(text_lengths) if text_lengths else 0,
                'avg_length': (
                    sum(text_lengths) / len(text_lengths)
                    if text_lengths else 0
                ),
                'total_characters': sum(text_lengths)
            },
            'embedding_stats': {
                'mean_norm': float(
                    np.mean([np.linalg.norm(e) for e in embedding_arrays])
                ) if embedding_arrays else 0,
                'std_norm': float(
                    np.std([np.linalg.norm(e) for e in embedding_arrays])
                ) if embedding_arrays else 0
            }
        }
    
    def _save_document_embeddings(
        self, document_data: Dict[str, Any]
    ):
        try:
            filename = document_data.get('filename', 'unknown')
            safe_filename = "".join(
                c for c in filename
                if c.isalnum() or c in (' ', '-', '_')
            ).rstrip()
            
            output_file = (
                self.output_dir / f"{safe_filename}_embeddings.json"
            )
            
            save_data = {
                'filename': filename,
                'chunk_count': len(
                    document_data.get('embedded_chunks', [])
                ),
                'embedding_stats': document_data.get('embedding_stats'),
                'embedded_at': document_data.get('embedding_timestamp'),
                'model': self.model_name,
                'model_type': 'multilingual-e5-small',     # ← CHANGED
                'embedding_dim': self.embedding_dim
            }
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, indent=2, ensure_ascii=False)
            
            logger.debug(f"💾 Saved metadata: {output_file.name}")
            
        except Exception as e:
            logger.warning(f"⚠️  Could not save metadata: {e}")