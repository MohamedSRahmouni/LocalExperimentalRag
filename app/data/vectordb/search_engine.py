"""
Search Engine - Semantic search using Weaviate vector similarity
"""

import logging
from typing import List, Dict, Any, Optional

from .weaviate_client import WeaviateClient

logger = logging.getLogger(__name__)


class SearchEngine:
    """
    Handles semantic search operations using Weaviate vector similarity
    """
    
    def __init__(
        self,
        weaviate_client: WeaviateClient,
        class_name: str = "Ragdocument"
    ):
        """
        Initialize search engine
        
        Args:
            weaviate_client: Weaviate client instance
            class_name: Name of the collection to search
        """
        self.client = weaviate_client
        self.class_name = class_name
    
    def vector_search(
        self,
        query_vector: List[float],
        top_k: int = 5,
        min_score: float = 0.0,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Perform vector similarity search
        
        Args:
            query_vector: Query embedding vector
            top_k: Number of results to return
            min_score: Minimum similarity score (0-1)
            filters: Optional metadata filters
            
        Returns:
            List of search results with scores
        """
        if not self.client.is_connected():
            logger.error("Not connected to Weaviate")
            return []
        
        try:
            collection = self.client.get_collection(self.class_name)
            if not collection:
                logger.error(f"Collection '{self.class_name}' not found")
                return []
            
            # Build query
            query = collection.query.near_vector(
                near_vector=query_vector,
                limit=top_k,
                return_metadata=['distance', 'certainty']
            )
            
            # Add filters if provided
            if filters:
                where_filter = self._build_filters(filters)
                if where_filter:
                    query = query.with_where(where_filter)
            
            # Execute search
            response = query.do()
            
            # Process results
            results = []
            for obj in response.objects:
                # Convert distance to similarity (certainty is already 0-1)
                certainty = obj.metadata.certainty if hasattr(obj.metadata, 'certainty') else 0.5
                
                # Filter by minimum score
                if certainty < min_score:
                    continue
                
                props = obj.properties
                result = {
                    "chunk_id": props.get("chunk_id"),
                    "text": props.get("text"),
                    "score": certainty,
                    "similarity": certainty,
                    "distance": obj.metadata.distance if hasattr(obj.metadata, 'distance') else None,
                    "metadata": {
                        "filename": props.get("filename"),
                        "file_type": props.get("file_type"),
                        "chunk_index": props.get("chunk_index"),
                        "total_chunks": props.get("total_chunks"),
                        "similarity_score": props.get("similarity_score"),
                        "sentence_count": props.get("sentence_count"),
                        "model_type": props.get("model_type")
                    },
                    "text_length": props.get("text_length"),
                    "model": props.get("embedding_model"),
                    "preview": props.get("text", "")[:200] + "..." if len(props.get("text", "")) > 200 else props.get("text", "")
                }
                
                results.append(result)
            
            logger.info(f"Vector search returned {len(results)} results (top_k={top_k})")
            
            return results
            
        except Exception as e:
            logger.error(f"Error in vector search: {str(e)}")
            import traceback
            traceback.print_exc()
            return []
    
    def semantic_search(
        self,
        query_text: str,
        embedder,
        top_k: int = 5,
        min_score: float = 0.5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Perform semantic search using query text
        
        Args:
            query_text: Query text
            embedder: Embedder instance to encode query
            top_k: Number of results
            min_score: Minimum similarity score
            filters: Optional filters
            
        Returns:
            List of search results
        """
        logger.info(f"Semantic search: '{query_text[:100]}...'")
        
        # Encode query
        try:
            query_vector = embedder.encode([query_text], normalize=True)[0]
            query_vector = query_vector.tolist()
        except Exception as e:
            logger.error(f"Error encoding query: {str(e)}")
            return []
        
        # Perform vector search
        return self.vector_search(
            query_vector=query_vector,
            top_k=top_k,
            min_score=min_score,
            filters=filters
        )
    
    def hybrid_search(
        self,
        query_text: str,
        embedder,
        top_k: int = 5,
        min_score: float = 0.5,
        alpha: float = 0.7,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Hybrid search combining vector similarity and keyword search
        
        Args:
            query_text: Query text
            embedder: Embedder instance
            top_k: Number of results
            min_score: Minimum score
            alpha: Weight for vector search (0=keyword only, 1=vector only)
            filters: Optional filters
            
        Returns:
            List of search results
        """
        if not self.client.is_connected():
            return []
        
        logger.info(f"Hybrid search (alpha={alpha}): '{query_text[:100]}...'")
        
        # Encode query
        try:
            query_vector = embedder.encode([query_text], normalize=True)[0].tolist()
        except Exception as e:
            logger.error(f"Error encoding query: {str(e)}")
            return []
        
        try:
            collection = self.client.get_collection(self.class_name)
            if not collection:
                return []
            
            # Hybrid query
            query = collection.query.hybrid(
                query=query_text,
                vector=query_vector,
                alpha=alpha,
                limit=top_k,
                return_metadata=['score', 'explain_score']
            )
            
            # Add filters
            if filters:
                where_filter = self._build_filters(filters)
                if where_filter:
                    query = query.with_where(where_filter)
            
            # Execute
            response = query.do()
            
            results = []
            for obj in response.objects:
                score = obj.metadata.score if hasattr(obj.metadata, 'score') else 0.5
                
                if score < min_score:
                    continue
                
                props = obj.properties
                result = {
                    "chunk_id": props.get("chunk_id"),
                    "text": props.get("text"),
                    "score": score,
                    "similarity": score,
                    "metadata": {
                        "filename": props.get("filename"),
                        "file_type": props.get("file_type"),
                        "chunk_index": props.get("chunk_index"),
                        "model_type": props.get("model_type")
                    },
                    "text_length": props.get("text_length"),
                    "model": props.get("embedding_model"),
                    "preview": props.get("text", "")[:200] + "..." if len(props.get("text", "")) > 200 else props.get("text", "")
                }
                
                results.append(result)
            
            logger.info(f"Hybrid search returned {len(results)} results")
            
            return results
            
        except Exception as e:
            logger.error(f"Error in hybrid search: {str(e)}")
            import traceback
            traceback.print_exc()
            return []
    
    def search_by_filename(
        self,
        filename: str,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Retrieve all chunks from a specific file
        
        Args:
            filename: Name of the file
            limit: Maximum number of chunks to return
            
        Returns:
            List of chunks
        """
        if not self.client.is_connected():
            return []
        
        try:
            collection = self.client.get_collection(self.class_name)
            if not collection:
                return []
            
            response = collection.query.fetch_objects(
                limit=limit,
                filters={
                    "path": ["filename"],
                    "operator": "Equal",
                    "valueText": filename
                }
            )
            
            results = []
            for obj in response.objects:
                props = obj.properties
                results.append({
                    "chunk_id": props.get("chunk_id"),
                    "text": props.get("text"),
                    "metadata": {
                        "filename": props.get("filename"),
                        "chunk_index": props.get("chunk_index"),
                        "total_chunks": props.get("total_chunks")
                    },
                    "chunk_index": props.get("chunk_index", 0)
                })
            
            # Sort by chunk index
            results.sort(key=lambda x: x.get("chunk_index", 0))
            
            logger.info(f"Found {len(results)} chunks for file: {filename}")
            return results
            
        except Exception as e:
            logger.error(f"Error searching by filename: {str(e)}")
            return []
    
    def get_random_samples(self, count: int = 5) -> List[Dict[str, Any]]:
        """
        Get random sample documents
        
        Args:
            count: Number of samples
            
        Returns:
            List of random documents
        """
        if not self.client.is_connected():
            return []
        
        try:
            collection = self.client.get_collection(self.class_name)
            if not collection:
                return []
            
            response = collection.query.fetch_objects(limit=count)
            
            results = []
            for obj in response.objects:
                props = obj.properties
                results.append({
                    "chunk_id": props.get("chunk_id"),
                    "text": props.get("text"),
                    "metadata": {
                        "filename": props.get("filename"),
                        "file_type": props.get("file_type")
                    },
                    "preview": props.get("text", "")[:150] + "..."
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Error getting random samples: {str(e)}")
            return []
    
    def _build_filters(self, filters: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Build Weaviate filter clauses
        
        Args:
            filters: Filter dict (e.g., {"filename": "doc.pdf", "file_type": "pdf"})
            
        Returns:
            Weaviate where filter
        """
        if not filters:
            return None
        
        conditions = []
        
        for key, value in filters.items():
            if key == "filename":
                conditions.append({
                    "path": ["filename"],
                    "operator": "Equal",
                    "valueText": value
                })
            elif key == "file_type":
                conditions.append({
                    "path": ["file_type"],
                    "operator": "Equal",
                    "valueText": value
                })
            elif key == "model_type":
                conditions.append({
                    "path": ["model_type"],
                    "operator": "Equal",
                    "valueText": value
                })
            elif key == "min_length":
                conditions.append({
                    "path": ["text_length"],
                    "operator": "GreaterThanEqual",
                    "valueInt": value
                })
            elif key == "max_length":
                conditions.append({
                    "path": ["text_length"],
                    "operator": "LessThanEqual",
                    "valueInt": value
                })
        
        if not conditions:
            return None
        
        if len(conditions) == 1:
            return conditions[0]
        
        # Multiple conditions - use AND
        return {
            "operator": "And",
            "operands": conditions
        }
    
    def get_similar_chunks(
        self,
        chunk_id: str,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Find chunks similar to a given chunk
        
        Args:
            chunk_id: ID of the reference chunk
            top_k: Number of similar chunks to return
            
        Returns:
            List of similar chunks
        """
        if not self.client.is_connected():
            return []
        
        try:
            collection = self.client.get_collection(self.class_name)
            if not collection:
                return []
            
            # First, get the reference chunk and its vector
            response = collection.query.fetch_objects(
                limit=1,
                filters={
                    "path": ["chunk_id"],
                    "operator": "Equal",
                    "valueText": chunk_id
                },
                include_vector=True
            )
            
            if not response.objects:
                logger.warning(f"Chunk not found: {chunk_id}")
                return []
            
            reference_vector = response.objects[0].vector
            
            # Search for similar chunks
            results = self.vector_search(
                query_vector=reference_vector,
                top_k=top_k + 1  # +1 to exclude reference chunk
            )
            
            # Remove reference chunk from results
            results = [r for r in results if r["chunk_id"] != chunk_id]
            
            return results[:top_k]
            
        except Exception as e:
            logger.error(f"Error finding similar chunks: {str(e)}")
            return []