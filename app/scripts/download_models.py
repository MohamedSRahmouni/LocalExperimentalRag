"""
Download required models on first run
"""
import os
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_sentence_transformers_model():
    """Download sentence-transformers model"""
    try:
        logger.info("Downloading sentence-transformers model...")
        from sentence_transformers import SentenceTransformer
        
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
        model = SentenceTransformer(model_name)
        logger.info(f"✓ Downloaded {model_name}")
        return True
    except Exception as e:
        logger.error(f"✗ Failed to download sentence-transformers: {e}")
        return False

def download_nltk_data():
    """Download required NLTK data"""
    try:
        logger.info("Downloading NLTK data...")
        import nltk
        
        resources = ['punkt', 'averaged_perceptron_tagger', 'wordnet']
        for resource in resources:
            try:
                nltk.download(resource, quiet=True)
                logger.info(f"✓ Downloaded NLTK {resource}")
            except Exception as e:
                logger.warning(f"Could not download NLTK {resource}: {e}")
        
        return True
    except Exception as e:
        logger.error(f"✗ Failed to download NLTK data: {e}")
        return False

def download_easyocr_model():
    """Download EasyOCR model"""
    try:
        logger.info("Downloading EasyOCR model (this may take a while)...")
        import easyocr
        
        reader = easyocr.Reader(['en'], gpu=False)
        logger.info("✓ Downloaded EasyOCR English model")
        return True
    except Exception as e:
        logger.error(f"✗ Failed to download EasyOCR: {e}")
        return False

def main():
    """Download all required models"""
    logger.info("="*80)
    logger.info("Downloading required models...")
    logger.info("="*80)
    
    results = {
        'sentence_transformers': download_sentence_transformers_model(),
        'nltk': download_nltk_data(),
        'easyocr': download_easyocr_model()
    }
    
    logger.info("="*80)
    logger.info("Download Summary:")
    for model, success in results.items():
        status = "✓" if success else "✗"
        logger.info(f"{status} {model}")
    
    all_success = all(results.values())
    logger.info("="*80)
    
    if all_success:
        logger.info("✓ All models downloaded successfully!")
    else:
        logger.warning("⚠ Some models failed to download. Check logs above.")
    
    return all_success

if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)