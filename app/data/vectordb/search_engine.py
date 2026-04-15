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
            
            # ✅ FIX: Nouvelle syntaxe Weaviate v4
            # Build query - NO .do() method anymore!
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
            
            # ✅ FIX: Execute query directly (no .do())
            response = query
            
            # Process results
            results = []
            
            # ✅ FIX: Iterate over response.objects directly
            if hasattr(response, 'objects'):
                objects = response.objects
            else:
                objects = []
            
            for obj in objects:
                # Get certainty/distance
                certainty = 0.5  # Default
                distance = None
                
                if hasattr(obj, 'metadata'):
                    if hasattr(obj.metadata, 'certainty'):
                        certainty = obj.metadata.certainty
                    if hasattr(obj.metadata, 'distance'):
                        distance = obj.metadata.distance
                
                # Filter by minimum score
                if certainty < min_score:
                    continue
                
                # Get properties
                props = obj.properties if hasattr(obj, 'properties') else {}
                
                result = {
                    "chunk_id": props.get("chunk_id"),
                    "text": props.get("text"),
                    "score": certainty,
                    "similarity": certainty,
                    "distance": distance,
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
            
            # ✅ FIX: Hybrid query - nouvelle syntaxe
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
            
            # ✅ FIX: Execute directly
            response = query
            
            results = []
            
            # ✅ FIX: Iterate directly
            if hasattr(response, 'objects'):
                objects = response.objects
            else:
                objects = []
            
            for obj in objects:
                score = 0.5
                if hasattr(obj, 'metadata') and hasattr(obj.metadata, 'score'):
                    score = obj.metadata.score
                
                if score < min_score:
                    continue
                
                props = obj.properties if hasattr(obj, 'properties') else {}
                
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
            
            # ✅ FIX: Nouvelle syntaxe pour fetch_objects
            from weaviate.classes.query import Filter
            
            response = collection.query.fetch_objects(
                limit=limit,
                filters=Filter.by_property("filename").equal(filename)
            )
            
            results = []
            
            if hasattr(response, 'objects'):
                objects = response.objects
            else:
                objects = []
            
            for obj in objects:
                props = obj.properties if hasattr(obj, 'properties') else {}
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
            import traceback
            traceback.print_exc()
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
            
            # ✅ FIX: Nouvelle syntaxe
            response = collection.query.fetch_objects(limit=count)
            
            results = []
            
            if hasattr(response, 'objects'):
                objects = response.objects
            else:
                objects = []
            
            for obj in objects:
                props = obj.properties if hasattr(obj, 'properties') else {}
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
    
    def _build_filters(self, filters: Dict[str, Any]) -> Optional[Any]:
        """
        Build Weaviate filter clauses (v4 syntax)
        
        Args:
            filters: Filter dict
            
        Returns:
            Weaviate Filter object
        """
        if not filters:
            return None
        
        try:
            from weaviate.classes.query import Filter
            
            conditions = []
            
            for key, value in filters.items():
                if key == "filename":
                    conditions.append(Filter.by_property("filename").equal(value))
                elif key == "file_type":
                    conditions.append(Filter.by_property("file_type").equal(value))
                elif key == "model_type":
                    conditions.append(Filter.by_property("model_type").equal(value))
                elif key == "min_length":
                    conditions.append(Filter.by_property("text_length").greater_or_equal(value))
                elif key == "max_length":
                    conditions.append(Filter.by_property("text_length").less_or_equal(value))
            
            if not conditions:
                return None
            
            if len(conditions) == 1:
                return conditions[0]
            
            # Multiple conditions - use AND
            result = conditions[0]
            for cond in conditions[1:]:
                result = result & cond
            
            return result
            
        except Exception as e:
            logger.error(f"Error building filters: {e}")
            return None
    
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
            from weaviate.classes.query import Filter
            
            collection = self.client.get_collection(self.class_name)
            if not collection:
                return []
            
            # First, get the reference chunk and its vector
            response = collection.query.fetch_objects(
                limit=1,
                filters=Filter.by_property("chunk_id").equal(chunk_id),
                include_vector=True
            )
            
            if not hasattr(response, 'objects') or not response.objects:
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