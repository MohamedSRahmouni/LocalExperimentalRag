import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

from .chunk_embedder import ChunkEmbedder
from .batch_processor import BatchEmbeddingProcessor

logger = logging.getLogger(__name__)


class EmbeddingManager:
    """
    Main manager for the embedding pipeline
    Coordinates between chunk embedding and batch processing
    """
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2-VL-2B-Instruct",
        use_gpu: bool = False,
        cache_dir: Optional[str] = None,
        batch_size: int = 8,
        max_workers: int = 2,
        save_intermediate: bool = True,
        output_dir: Optional[str] = None
    ):
        """
        Initialize embedding manager
        
        Args:
            model_name: Embedding model name
            use_gpu: Use GPU
            cache_dir: Model cache directory
            batch_size: Batch size for embedding
            max_workers: Workers for parallel processing
            save_intermediate: Save intermediate results
            output_dir: Output directory for embeddings
        """
        logger.info("Initializing Embedding Manager")
        
        # Initialize embedder
        self.embedder = ChunkEmbedder(
            model_name=model_name,
            use_gpu=use_gpu,
            cache_dir=cache_dir,
            batch_size=batch_size
        )
        
        # Initialize batch processor
        self.batch_processor = BatchEmbeddingProcessor(
            embedder=self.embedder,
            max_workers=max_workers,
            save_intermediate=save_intermediate,
            output_dir=output_dir
        )
        
        logger.info("✓ Embedding Manager initialized")
    
    def is_ready(self) -> bool:
        """Check if embedding system is ready"""
        return self.embedder.is_available()
    
    def embed_single_document(
        self,
        document_data: Dict[str, Any],
        show_progress: bool = True
    ) -> Dict[str, Any]:
        """
        Embed a single document
        
        Args:
            document_data: Document data from DocumentProcessor
            show_progress: Show progress bar
            
        Returns:
            Document with embeddings
        """
        if not self.is_ready():
            raise RuntimeError("Embedding system not ready")
        
        return self.batch_processor.process_document(
            document_data,
            show_progress=show_progress
        )
    
    def embed_multiple_documents(
        self,
        documents: List[Dict[str, Any]],
        parallel: bool = False,
        show_progress: bool = True
    ) -> Dict[str, Any]:
        """
        Embed multiple documents
        
        Args:
            documents: List of document data
            parallel: Use parallel processing
            show_progress: Show progress bars
            
        Returns:
            Batch processing results
        """
        if not self.is_ready():
            raise RuntimeError("Embedding system not ready")
        
        return self.batch_processor.process_batch(
            documents,
            parallel=parallel,
            show_progress=show_progress
        )
    
    def embed_chunks_only(
        self,
        chunks: List[str],
        chunks_metadata: Optional[List[Dict[str, Any]]] = None,
        show_progress: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Embed just chunks without document context
        
        Args:
            chunks: List of text chunks
            chunks_metadata: Optional metadata for chunks
            show_progress: Show progress
            
        Returns:
            List of embedded chunks
        """
        if not self.is_ready():
            raise RuntimeError("Embedding system not ready")
        
        return self.embedder.embed_chunks_batch(
            chunks,
            chunks_metadata=chunks_metadata,
            show_progress=show_progress
        )
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get information about embedding system"""
        return {
            'model_name': self.embedder.model_name,
            'embedding_dim': self.embedder.embedding_dim,
            'device': 'GPU' if self.embedder.use_gpu else 'CPU',
            'batch_size': self.embedder.batch_size,
            'is_ready': self.is_ready(),
            'output_dir': str(self.batch_processor.output_dir)
        }
    
    def clear_cache(self):
        """Clear all caches"""
        self.embedder.clear_cache()
        logger.info("✓ All caches cleared")


# Example usage
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Initialize manager
    manager = EmbeddingManager(
        model_name="Qwen/Qwen2-VL-2B-Instruct",
        use_gpu=False,
        batch_size=8,
        save_intermediate=True,
        output_dir="embeddings_output"
    )
    
    # Check status
    print("\nSystem Info:")
    print(manager.get_system_info())
    
    # Example: Embed some chunks
    test_chunks = [
        "This is a test document about machine learning.",
        "Neural networks are powerful models for pattern recognition.",
        "Deep learning has revolutionized artificial intelligence."
    ]
    
    if manager.is_ready():
        print("\nEmbedding test chunks...")
        embedded = manager.embed_chunks_only(test_chunks)
        
        print(f"\n✓ Embedded {len(embedded)} chunks")
        for i, chunk in enumerate(embedded, 1):
            print(f"\nChunk {i}:")
            print(f"  ID: {chunk['chunk_id'][:16]}...")
            print(f"  Text: {chunk['text'][:50]}...")
            print(f"  Embedding dim: {chunk['embedding_dim']}")
            print(f"  Embedded at: {chunk['embedded_at']}")