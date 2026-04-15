import logging
import re
import unicodedata
from typing import List, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)




class TextCleaner:
    """Handles text cleaning and normalization"""

    def __init__(self):
        # Common stopwords (you can expand this or use NLTK)
        self.common_junk = [
            r'http\S+',  # Remove URLs
            r'www\.\S+',
        ]

    def clean_text(self, text: str) -> str:
        """Main text cleaning pipeline"""
        if not text or not isinstance(text, str):
            return ""
        
        try:
            # 1. Normalize unicode characters
            text = self.normalize_unicode(text)
            
            # 2. Remove extra whitespace
            text = self.remove_extra_whitespace(text)
            
            # 3. Remove special characters (but keep punctuation)
            text = self.remove_special_chars(text)
            
            # 4. Normalize case (optional - can be removed if case matters)
            text = text.lower()
            
            # 5. Remove URLs and emails if needed
            text = self.remove_urls_emails(text)
            
            # 6. Fix common encoding issues
            text = self.fix_encoding_issues(text)
            
            return text.strip()
        
        except Exception as e:
            logger.error(f"Error cleaning text: {str(e)}")
            return ""
        
    def normalize_unicode(self, text: str) -> str:
        """Normalize unicode characters to standard form"""
        # NFKD: compatibility decomposition
        text = unicodedata.normalize('NFKD', text)
        # Remove non-ASCII characters or keep them based on requirement
        # text = text.encode('ascii', 'ignore').decode('ascii')  # Strict ASCII
        return text
    
    def remove_extra_whitespace(self, text: str) -> str:
        """Remove extra spaces, tabs, newlines"""
        text = re.sub(r'\s+', ' ', text)
        text = '\n'.join(line.strip() for line in text.split('\n'))
        text = re.sub(r'\n\s*\n', '\n\n', text)
        return text
    
    def remove_special_chars(self, text: str) -> str:
        """Remove or replace special characters"""
        # Remove zero-width characters
        text = re.sub(r'[\u200b\u200c\u200d\ufeff]', '', text)
        # Remove control characters except newlines and tabs
        text = ''.join(char for char in text if unicodedata.category(char)[0] != 'C' or char in '\n\t')
        return text
    
    def remove_urls_emails(self, text: str) -> str:
        """Remove URLs and email addresses"""
        # Remove URLs
        text = re.sub(r'http\S+|www\.\S+', '', text)
        # Optionally remove emails (comment out if you need to keep them)
        # text = re.sub(r'\S+@\S+', '', text)
        return text
    
    def fix_encoding_issues(self, text: str) -> str:
        """Fix common encoding problems"""
        replacements = {
            'â€™': "'",
            'â€œ': '"',
            'â€': '"',
            'â€"': '-',
            'â€"': '--',
            'Ã©': 'é',
            'Ã¨': 'è',
            'Ã ': 'à',
        }
        
        for wrong, right in replacements.items():
            text = text.replace(wrong, right)
        
        return text
    
    def remove_duplicates(self, texts: List[str]) -> List[str]:
        """Remove duplicate text chunks while preserving order"""
        seen = set()
        unique_texts = []
        
        for text in texts:
            # Normalize for comparison
            normalized = self.clean_text(text).lower().strip()
            
            if normalized and normalized not in seen:
                seen.add(normalized)
                unique_texts.append(text)
        
        logger.info(f"Removed {len(texts) - len(unique_texts)} duplicate texts")
        return unique_texts
    
    def is_valid_text(self, text: str, min_length: int = 5) -> bool:  # ✅ Changed from 10 to 5
        """
        Check if text is valid and meaningful
        
        Args:
            text: Text to validate
            min_length: Minimum length (lowered to 5 for tolerance)
            
        Returns:
            True if valid, False otherwise
        """
        if not text or not isinstance(text, str):
            return False
        
        cleaned = text.strip()
        
        # Very permissive minimum length
        if len(cleaned) < min_length:
            logger.debug(f"Text too short: {len(cleaned)} < {min_length}")
            return False
        
        # Must contain SOME alphabetic or numeric characters
        if not any(c.isalnum() for c in cleaned):
            logger.debug("Text contains no alphanumeric characters")
            return False
        
        # ✅ REMOVED: Repetition check for short texts
        if len(cleaned) >= 50:  # Only check repetition for longer texts
            if self.is_repetitive(cleaned):
                logger.debug("Text is repetitive")
                return False
        
        return True

    def is_repetitive(self, text: str, threshold: float = 0.7) -> bool:
        """Detect if text is overly repetitive"""
        words = text.split()
        if len(words) < 10:
            return False
        
        unique_words = set(words)
        repetition_ratio = len(unique_words) / len(words)
        
        return repetition_ratio < (1 - threshold)