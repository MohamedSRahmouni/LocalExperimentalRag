"""
Evaluation Dataset Management
For LangSmith automated evaluation
"""

import os
import json
import logging
from typing import List, Dict, Any
from datetime import datetime

from langsmith import Client

logger = logging.getLogger(__name__)


class EvaluationDatasetManager:
    """Manage evaluation datasets in LangSmith."""
    
    def __init__(self, dataset_name: str = "rag-evaluation-dataset"):
        self.dataset_name = dataset_name
        self.client = Client()
    
    def create_or_get_dataset(self) -> str:
        """Create or retrieve dataset."""
        try:
            # Try to find existing dataset
            datasets = list(self.client.list_datasets(dataset_name=self.dataset_name))
            
            if datasets:
                logger.info(f"✅ Dataset found: {datasets[0].id}")
                return datasets[0].id
            
            # Create new dataset
            dataset = self.client.create_dataset(dataset_name=self.dataset_name)
            logger.info(f"📁 Created dataset: {dataset.id}")
            return dataset.id
            
        except Exception as e:
            logger.error(f"❌ Failed to manage dataset: {e}")
            return None
    
    def upload_examples(
        self,
        examples: List[Dict[str, Any]]
    ):
        """
        Upload question-answer pairs to dataset.
        
        Example format:
        [
            {
                "inputs": {"question": "What is machine learning?"},
                "outputs": {"answer": "Machine learning is..."}
            }
        ]
        """
        dataset_id = self.create_or_get_dataset()
        if not dataset_id:
            return
        
        try:
            self.client.upload_csv(
                csv_data=examples,
                input_keys=["question"],
                output_keys=["answer"],
                description="RAG evaluation questions",
                data_type="kv",
            )
            logger.info(f"✅ Uploaded {len(examples)} examples")
        except Exception as e:
            logger.error(f"❌ Failed to upload examples: {e}")
    
    def run_evaluation(
        self,
        rag_function,
        num_samples: int = 5
    ):
        """Run evaluation on dataset."""
        try:
            from langsmith.evaluation import evaluate
            
            results = evaluate(
                rag_function,
                dataset_name=self.dataset_name,
                evaluators=[],  # Add custom evaluators here
            )
            
            logger.info(f"✅ Evaluation complete: {results}")
            return results
            
        except Exception as e:
            logger.error(f"❌ Evaluation failed: {e}")
            return None