"""
Vector Store - High-level vector storage operations for Weaviate
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import uuid
from datetime import datetime, timezone

from .weaviate_client import WeaviateClient

logger = logging.getLogger(__name__)


class VectorStore:
    """
    Manages vector storage operations in Weaviate
    Handles document chunking, embedding storage, and indexing
    """
    
    def __init__(
        self,
        weaviate_client: WeaviateClient,
        class_name: str = "Ragdocument",
        vector_dims: int = 768
    ):
        """
        Initialize vector store
        
        Args:
            weaviate_client: Weaviate client instance
            class_name: Name of the collection
            vector_dims: Dimension of embedding vectors
        """
        self.client = weaviate_client
        self.class_name = class_name
        self.vector_dims = vector_dims
        
        # Create collection if it doesn't exist
        self._ensure_collection()
    
    def _ensure_collection(self):
        """Create collection with proper schema for vectors"""
        self.client.create_collection(
            class_name=self.class_name,
            description="RAG document chunks with embeddings",
            vector_dims=self.vector_dims
        )
 

    def store_embeddings(
        self,
        embedded_chunks: List[Dict[str, Any]],
        document_metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, int]:
        """Store embedded chunks in Weaviate"""
        
        logger.info("="*70)
        logger.info("STORE EMBEDDINGS - Starting")
        logger.info("="*70)
        logger.info(f"Chunks to store: {len(embedded_chunks)}")
        logger.info(f"Connected: {self.client.is_connected()}")
        logger.info(f"Collection name: '{self.class_name}'")
        
        if not embedded_chunks:
            logger.warning("No chunks to store")
            return {"success": 0, "failed": 0}
        
        if not self.client.is_connected():
            logger.error("Not connected to Weaviate")
            return {"success": 0, "failed": 0}
        
        try:
            # Get collection
            collection = self.client.get_collection(self.class_name)
            
            if collection is None:
                logger.error(f"Failed to get collection object")
                return {"success": 0, "failed": 0}
            
            logger.info(f"✓ Collection retrieved")
            
            # Batch processing with error capture
            success_count = 0
            failed_count = 0
            failed_objects = []
            
            # Create batch inserter with rate limits disabled for better error reporting
            batch_size = min(20, len(embedded_chunks))  # Smaller batches for better error tracking
            
            for batch_start in range(0, len(embedded_chunks), batch_size):
                batch_end = min(batch_start + batch_size, len(embedded_chunks))
                current_batch = embedded_chunks[batch_start:batch_end]
                
                logger.info(f"Processing batch {batch_start}-{batch_end}...")
                
                # Use fixed-size batch for better error tracking
                with collection.batch.fixed_size(batch_size=batch_size) as batch:
                    for idx_in_batch, chunk in enumerate(current_batch):
                        global_idx = batch_start + idx_in_batch
                        
                        try:
                            # Get embedding
                            vector = chunk.get("embedding")
                            if not vector:
                                logger.warning(f"Chunk {global_idx} missing embedding")
                                failed_count += 1
                                continue
                            
                            # Convert numpy array to list if needed
                            if hasattr(vector, 'tolist'):
                                vector = vector.tolist()
                            
                            # Validate vector
                            if not isinstance(vector, list):
                                logger.error(f"Chunk {global_idx}: vector is not a list")
                                failed_count += 1
                                continue
                            
                            if len(vector) != 768:
                                logger.error(f"Chunk {global_idx}: vector length is {len(vector)}, expected 768")
                                failed_count += 1
                                continue
                            
                            # Prepare properties with strict type conversion
                           # Find this in your store_embeddings method:
                            properties = {
                                "chunk_id": str(chunk.get("chunk_id", f"chunk_{global_idx}")),
                                "text": str(chunk.get("text", "")),
                                "text_length": int(chunk.get("text_length") or len(chunk.get("text", ""))),
                                "embedding_model": str(chunk.get("embedding_model") or chunk.get("model_name") or "unknown"),
                                "model_type": str(chunk.get("model_type", "sentence-transformer")),
                                "embedded_at": self._parse_datetime(chunk.get("embedded_at")),
                                "indexed_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z",  # ← FIX THIS
                                "filename": str(document_metadata.get("filename", "")) if document_metadata else "",
                                "file_type": str(document_metadata.get("extension", "")) if document_metadata else "",
                                "chunk_index": int(global_idx),
                                "total_chunks": int(len(embedded_chunks))
                            }
                            
                            # Add optional fields only if they exist and are valid
                            metadata = chunk.get("metadata", {})
                            
                            similarity_score = chunk.get("similarity_score") or metadata.get("similarity_score")
                            if similarity_score is not None:
                                try:
                                    properties["similarity_score"] = float(similarity_score)
                                except (ValueError, TypeError):
                                    pass  # Skip if invalid
                            
                            sentence_count = chunk.get("sentence_count") or metadata.get("sentence_count")
                            if sentence_count is not None:
                                try:
                                    properties["sentence_count"] = int(sentence_count)
                                except (ValueError, TypeError):
                                    pass  # Skip if invalid
                            
                            # Debug first object
                            if global_idx == 0:
                                logger.info("="*70)
                                logger.info("FIRST OBJECT:")
                                logger.info(f"Properties: {properties}")
                                logger.info(f"Vector length: {len(vector)}")
                                logger.info(f"Vector type: {type(vector)}")
                                logger.info(f"Vector sample: {vector[:3]}")
                                logger.info("="*70)
                            
                            # Add to batch
                            uuid_result = batch.add_object(
                                properties=properties,
                                vector=vector
                            )
                            
                            success_count += 1
                            
                        except Exception as e:
                            logger.error(f"Chunk {global_idx} error: {str(e)}")
                            import traceback
                            traceback.print_exc()
                            failed_count += 1
                    
                    # Check for failures after batch completes
                    if hasattr(batch, 'failed_objects') and batch.failed_objects:
                        batch_failures = batch.failed_objects
                        logger.error(f"❌ Batch {batch_start}-{batch_end} had {len(batch_failures)} failures")
                        
                        for fail in batch_failures[:3]:  # Show first 3
                            logger.error(f"Failed object details:")
                            logger.error(f"  Error: {fail}")
                            if hasattr(fail, 'message'):
                                logger.error(f"  Message: {fail.message}")
                            if hasattr(fail, 'object'):
                                logger.error(f"  Object properties: {fail.object.properties if hasattr(fail.object, 'properties') else 'N/A'}")
                        
                        failed_objects.extend(batch_failures)
                        # Adjust counts
                        actual_batch_failed = len(batch_failures)
                        success_count -= actual_batch_failed
                        failed_count += actual_batch_failed
            
            logger.info("="*70)
            logger.info("BATCH PROCESSING COMPLETE")
            logger.info(f"  Success: {success_count}")
            logger.info(f"  Failed: {failed_count}")
            logger.info(f"  Total failed objects collected: {len(failed_objects)}")
            logger.info("="*70)
            
            # If we have detailed failures, log them
            if failed_objects:
                logger.error("DETAILED FAILURE ANALYSIS:")
                for i, fail in enumerate(failed_objects[:5]):
                    logger.error(f"\nFailure #{i+1}:")
                    logger.error(f"  Type: {type(fail)}")
                    logger.error(f"  Repr: {repr(fail)}")
                    logger.error(f"  Dir: {[attr for attr in dir(fail) if not attr.startswith('_')]}")
            
            # Verify with collection count
            try:
                count = collection.aggregate.over_all(total_count=True)
                logger.info(f"Collection count verification: {count.total_count} objects")
                
                if count.total_count == 0 and success_count > 0:
                    logger.error("⚠️  WARNING: Reported success but collection is empty!")
                    logger.error("   All objects were rejected by Weaviate")
                    # Override counts
                    failed_count = len(embedded_chunks)
                    success_count = 0
            except Exception as e:
                logger.warning(f"Could not verify count: {e}")
            
            return {"success": success_count, "failed": failed_count}
            
        except Exception as e:
            logger.error(f"Error in batch storage: {str(e)}")
            import traceback
            traceback.print_exc()
            return {"success": 0, "failed": len(embedded_chunks)}

    def store_document_embeddings(
        self,
        document_data: Dict[str, Any]
    ) -> Dict[str, int]:
        """
        Store all embeddings from a processed document
        
        Args:
            document_data: Document data with embedded_chunks
            
        Returns:
            Stats dict
        """
        if not document_data.get("embedding_complete"):
            logger.warning(f"Document not embedded: {document_data.get('filename')}")
            return {"success": 0, "failed": 0}
        
        embedded_chunks = document_data.get("embedded_chunks", [])
        
        metadata = {
            "filename": document_data.get("filename"),
            "filepath": document_data.get("filepath"),
            "extension": document_data.get("metadata", {}).get("file_type"),
            "processed_at": document_data.get("metadata", {}).get("processed_at")
        }
        
        return self.store_embeddings(embedded_chunks, metadata)
    
    def store_batch(
        self,
        documents: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Store embeddings from multiple documents
        
        Args:
            documents: List of document data
            
        Returns:
            Aggregated stats
        """
        total_success = 0
        total_failed = 0
        processed_docs = 0
        
        logger.info(f"Storing embeddings from {len(documents)} documents...")
        
        for doc in documents:
            if doc.get("embedding_complete"):
                result = self.store_document_embeddings(doc)
                total_success += result["success"]
                total_failed += result["failed"]
                processed_docs += 1
        
        logger.info("="*70)
        logger.info("✓ Batch storage complete")
        logger.info(f"  Documents processed: {processed_docs}")
        logger.info(f"  Chunks stored: {total_success}")
        logger.info(f"  Failed: {total_failed}")
        logger.info("="*70)
        
        return {
            "documents_processed": processed_docs,
            "chunks_stored": total_success,
            "chunks_failed": total_failed
        }
    
    def get_document_count(self) -> int:
        """Get total number of chunks stored"""
        stats = self.client.get_stats(self.class_name)
        return stats.get("document_count", 0)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get vector store statistics"""
        return self.client.get_stats(self.class_name)
    
    def delete_by_filename(self, filename: str) -> int:
        """
        Delete all chunks from a specific file
        
        Args:
            filename: Name of the file
            
        Returns:
            Number of deleted documents
        """
        if not self.client.is_connected():
            return 0
        
        try:
            collection = self.client.get_collection(self.class_name)
            if not collection:
                return 0
            
            # Delete by filename filter
            result = collection.data.delete_many(
                where={
                    "path": ["filename"],
                    "operator": "Equal",
                    "valueText": filename
                }
            )
            
            deleted = result.successful if hasattr(result, 'successful') else 0
            logger.info(f"Deleted {deleted} chunks for file: {filename}")
            return deleted
            
        except Exception as e:
            logger.error(f"Error deleting chunks: {str(e)}")
            return 0
    
    def clear_all(self) -> bool:
        """Delete all documents from the collection"""
        return self.client.delete_collection(self.class_name)
    
    def _parse_datetime(self, dt_string: Optional[str]) -> str:
        """Parse datetime string to RFC3339 format with Z (required by Weaviate)"""
        if not dt_string:
            return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
        
        try:
            # If already has Z, return as is
            if isinstance(dt_string, str) and dt_string.endswith('Z'):
                return dt_string
            
            # Parse the datetime
            if isinstance(dt_string, str):
                if 'T' in dt_string:
                    # Remove existing Z or timezone
                    dt_string = dt_string.replace('Z', '').split('+')[0].split('-')[0:3]
                    dt_string = '-'.join(dt_string) if len(dt_string) > 1 else dt_string[0]
                    
                    dt = datetime.fromisoformat(dt_string)
                else:
                    dt = datetime.fromisoformat(dt_string)
            else:
                dt = dt_string
            
            # Format with Z
            return dt.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
            
        except Exception as e:
            logger.warning(f"Error parsing datetime '{dt_string}': {e}")
            return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"