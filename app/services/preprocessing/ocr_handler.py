import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from PIL import Image
import cv2
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OCRHandler:
    """Handles OCR for images and scanned documents using EasyOCR"""
    
    def __init__(self, lang: str = 'en', use_gpu: bool = False):
        """
        Initialize OCR Handler with EasyOCR
        
        Args:
            lang: Language code ('en', 'ch_sim', 'fr', 'de', 'ko', 'ja', etc.)
            use_gpu: Whether to use GPU acceleration
        """
        self.supported_formats = {'.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif'}
        self.lang = lang
        self.use_gpu = use_gpu
        self.reader = None
        
        # Language mapping for EasyOCR
        lang_map = {
            'en': 'en',
            'ch': 'ch_sim',
            'chinese': 'ch_sim',
            'french': 'fr',
            'german': 'de',
            'korean': 'ko',
            'japan': 'ja',
            'japanese': 'ja',
            'spanish': 'es',
            'portuguese': 'pt',
            'russian': 'ru',
            'arabic': 'ar',
            'hindi': 'hi'
        }
        
        ocr_lang = lang_map.get(lang.lower(), 'en')
        
        try:
            import easyocr
            
            logger.info(f"Initializing EasyOCR with language: {ocr_lang}")
            logger.info("First run will download models (~100-200MB)...")
            
            self.reader = easyocr.Reader(
                [ocr_lang],
                gpu=use_gpu,
                verbose=False,
                download_enabled=True
            )
            
            logger.info("✓ EasyOCR initialized successfully")
            logger.info(f"  Language: {ocr_lang}")
            logger.info(f"  GPU: {'Enabled' if use_gpu else 'CPU'}")
            
        except ImportError as e:
            logger.error(f"EasyOCR not installed: {str(e)}")
            logger.error("Install with: pip install easyocr")
            raise
        except Exception as e:
            logger.error(f"Error initializing EasyOCR: {str(e)}")
            raise
    
    def extract_text_from_image(self, image_path: str, preprocess: bool = True) -> Dict[str, Any]:
        """Extract text from image using EasyOCR"""
        
        if self.reader is None:
            return {
                'success': False,
                'error': 'OCR engine not initialized',
                'text': ''
            }
        
        try:
            image_path = Path(image_path)
            
            if not image_path.exists():
                return {
                    'success': False,
                    'error': f'Image file not found: {image_path}',
                    'text': ''
                }
            
            logger.info(f"Extracting text from: {image_path.name}")
            
            # Read image
            if preprocess:
                img_array = self._preprocess_image(str(image_path))
            else:
                img_array = cv2.imread(str(image_path))
            
            if img_array is None:
                return {
                    'success': False,
                    'error': 'Could not read image file',
                    'text': ''
                }
            
            # Perform OCR
            result = self.reader.readtext(img_array, detail=1, paragraph=False)
            
            # Extract text and confidence
            text_lines = []
            confidence_scores = []
            bounding_boxes = []
            
            for detection in result:
                bbox, text, confidence = detection
                text_lines.append(text)
                confidence_scores.append(confidence)
                bounding_boxes.append(bbox)
            
            # Join text
            full_text = '\n'.join(text_lines)
            avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0
            
            logger.info(f"✓ Extracted {len(text_lines)} lines, {len(full_text)} chars, "
                       f"{avg_confidence*100:.1f}% confidence")
            
            return {
                'success': True,
                'text': full_text,
                'confidence': avg_confidence * 100,
                'word_count': len(full_text.split()),
                'char_count': len(full_text),
                'line_count': len(text_lines),
                'bounding_boxes': bounding_boxes,
                'confidence_scores': confidence_scores,
                'filename': image_path.name
            }
        
        except Exception as e:
            error_msg = f"OCR error on {image_path}: {str(e)}"
            logger.error(error_msg)
            return {
                'success': False,
                'error': error_msg,
                'text': ''
            }
    
    def _preprocess_image(self, image_path: str) -> np.ndarray:
        """Preprocess image for better OCR"""
        img = cv2.imread(image_path)
        
        if img is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        # Convert to grayscale
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        
        # Denoise
        denoised = cv2.fastNlMeansDenoising(gray, None, h=10, templateWindowSize=7, searchWindowSize=21)
        
        # Adaptive threshold
        binary = cv2.adaptiveThreshold(
            denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        
        return binary
    
    def is_scanned_pdf_page(self, text: str, threshold: int = 50) -> bool:
        """Check if PDF page is scanned"""
        if not text:
            return True
        meaningful_chars = sum(1 for c in text.strip() if c.isalnum())
        return meaningful_chars < threshold
    
    def extract_text_from_pdf_image(self, pdf_path: str, page_number: int, dpi: int = 300) -> str:
        """Convert PDF page to image and OCR"""
        
        if self.reader is None:
            return ""
        
        try:
            from pdf2image import convert_from_path
            
            images = convert_from_path(
                pdf_path,
                first_page=page_number + 1,
                last_page=page_number + 1,
                dpi=dpi
            )
            
            if not images:
                return ""
            
            # Save temp
            temp_path = f"temp_page_{page_number}_{os.getpid()}.png"
            images[0].save(temp_path, 'PNG')
            
            # OCR
            result = self.extract_text_from_image(temp_path, preprocess=True)
            
            # Cleanup
            if Path(temp_path).exists():
                Path(temp_path).unlink()
            
            return result.get('text', '')
            
        except Exception as e:
            logger.error(f"Error OCR PDF page {page_number}: {str(e)}")
            return ""
    
    def get_supported_languages(self) -> List[str]:
        """Get supported languages"""
        return ['en', 'ch_sim', 'fr', 'de', 'ko', 'ja', 'es', 'pt', 'ru', 'ar', 'hi']