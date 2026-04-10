import os
import re
from pathlib import Path
from typing import Optional, Dict, Any, List
import logging
import pdfplumber
import docx
import csv
import email
from email import policy
from email.parser import BytesParser
import extract_msg
from PIL import Image

from .ocr_handler import OCRHandler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FileHandler:
    """Handles different file types and extracts text"""
    
    SUPPORTED_EXTENSIONS = {
        '.pdf', '.doc', '.docx', '.csv', '.txt', '.eml', '.msg',
        '.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif'
    }
    
    IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif'}
    
    def __init__(self, lang: str = 'en', use_gpu: bool = False):
        """
        Initialize File Handler
        
        Args:
            lang: OCR language code ('en', 'ch', 'french', 'german', etc.)
            use_gpu: Whether to use GPU for OCR
        """
        self.lang = lang
        self.use_gpu = use_gpu
        self.ocr_handler = OCRHandler(lang=lang, use_gpu=use_gpu)
        
        self.extractors = {
            '.pdf': self.extract_from_pdf,
            '.docx': self.extract_from_docx,
            '.doc': self.extract_from_doc,
            '.txt': self.extract_from_txt,
            '.csv': self.extract_from_csv,
            '.eml': self.extract_from_eml,
            '.msg': self.extract_from_msg,
            '.png': self.extract_from_image,
            '.jpg': self.extract_from_image,
            '.jpeg': self.extract_from_image,
            '.tiff': self.extract_from_image,
            '.bmp': self.extract_from_image,
            '.gif': self.extract_from_image,
        }
    
    def extract_text(self, file_path: str) -> Dict[str, Any]:
        """Extract text from file based on extension"""
        file_path = Path(file_path)
        
        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            return {'success': False, 'error': 'File not found', 'text': ''}
        
        extension = file_path.suffix.lower()
        
        if extension not in self.SUPPORTED_EXTENSIONS:
            logger.error(f"Unsupported file type: {extension}")
            return {'success': False, 'error': f'Unsupported file type: {extension}', 'text': ''}
        
        try:
            extractor = self.extractors.get(extension)
            result = extractor(file_path)
            
            # Handle string return (legacy) or dict return
            if isinstance(result, str):
                text = result
                extra_data = {}
            else:
                text = result.get('text', '')
                extra_data = {k: v for k, v in result.items() if k != 'text'}
            
            return {
                'success': True,
                'text': text,
                'filename': file_path.name,
                'extension': extension,
                'size': file_path.stat().st_size,
                **extra_data
            }
        
        except Exception as e:
            logger.error(f"Error extracting text from {file_path}: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'text': '',
                'filename': file_path.name
            }
    
    def extract_from_pdf(self, file_path: Path) -> Dict[str, Any]:
        """
        Extract text from PDF using pdfplumber
        Detects scanned pages and uses OCR when needed
        """
        text_content = []
        is_scanned = False
        ocr_pages = []
        
        try:
            with pdfplumber.open(file_path) as pdf:
                total_pages = len(pdf.pages)
                logger.info(f"Processing PDF with {total_pages} pages: {file_path.name}")
                
                for page_num, page in enumerate(pdf.pages):
                    try:
                        # Extract text from page
                        page_text = page.extract_text()
                        
                        # Check if page is scanned (little/no text)
                        if self.ocr_handler.is_scanned_pdf_page(page_text or ''):
                            logger.info(f"Page {page_num + 1} appears to be scanned, using OCR")
                            is_scanned = True
                            ocr_pages.append(page_num + 1)
                            
                            # Use OCR on this page
                            ocr_text = self.ocr_handler.extract_text_from_pdf_image(
                                str(file_path), 
                                page_num
                            )
                            
                            if ocr_text:
                                text_content.append(ocr_text)
                        else:
                            # Use extracted text
                            if page_text:
                                text_content.append(page_text)
                                
                                # Also extract tables if present
                                tables = page.extract_tables()
                                for table in tables:
                                    table_text = self._format_table(table)
                                    if table_text:
                                        text_content.append(table_text)
                    
                    except Exception as e:
                        logger.warning(f"Error processing page {page_num + 1}: {str(e)}")
                        continue
            
            full_text = '\n\n'.join(text_content)
            
            return {
                'text': full_text,
                'is_scanned': is_scanned,
                'ocr_pages': ocr_pages,
                'total_pages': total_pages
            }
        
        except Exception as e:
            logger.error(f"Error reading PDF {file_path}: {str(e)}")
            raise
    
    def _format_table(self, table: List[List]) -> str:
        """Format extracted table data into readable text"""
        if not table:
            return ""
        
        formatted_rows = []
        for row in table:
            # Filter out None values and join cells
            cells = [str(cell).strip() if cell else '' for cell in row]
            row_text = ' | '.join(cells)
            if row_text.strip():
                formatted_rows.append(row_text)
        
        return '\n'.join(formatted_rows)
    
    def extract_from_image(self, file_path: Path) -> Dict[str, Any]:
        """Extract text from image using OCR"""
        logger.info(f"Extracting text from image: {file_path.name}")
        
        result = self.ocr_handler.extract_text_from_image(str(file_path))
        
        if not result['success']:
            raise Exception(result.get('error', 'OCR failed'))
        
        return {
            'text': result['text'],
            'ocr_confidence': result.get('confidence', 0),
            'is_image': True,
            'word_count': result.get('word_count', 0)
        }
    
    def extract_from_docx(self, file_path: Path) -> str:
        """Extract text from DOCX file"""
        try:
            doc = docx.Document(file_path)
            text_content = []
            
            # Extract paragraphs
            for para in doc.paragraphs:
                if para.text.strip():
                    text_content.append(para.text)
            
            # Extract tables
            for table in doc.tables:
                for row in table.rows:
                    row_text = ' | '.join(cell.text.strip() for cell in row.cells)
                    if row_text.strip():
                        text_content.append(row_text)
            
            # Extract headers and footers
            for section in doc.sections:
                header = section.header
                footer = section.footer
                
                for para in header.paragraphs:
                    if para.text.strip():
                        text_content.append(para.text)
                
                for para in footer.paragraphs:
                    if para.text.strip():
                        text_content.append(para.text)
            
            return '\n\n'.join(text_content)
        
        except Exception as e:
            logger.error(f"Error reading DOCX {file_path}: {str(e)}")
            raise
    
    def extract_from_doc(self, file_path: Path) -> str:
        """Extract text from DOC file (legacy format)"""
        try:
            # Try using textract if available
            import textract
            text = textract.process(str(file_path)).decode('utf-8')
            return text
        except ImportError:
            logger.warning("textract not installed. Trying alternative method...")
            try:
                # Alternative: use antiword if available
                import subprocess
                result = subprocess.run(
                    ['antiword', str(file_path)],
                    capture_output=True,
                    text=True
                )
                return result.stdout
            except Exception:
                logger.warning("antiword not available. Using basic extraction...")
                # Fallback: basic binary extraction
                with open(file_path, 'rb') as file:
                    content = file.read()
                    text = content.decode('latin-1', errors='ignore')
                    # Clean up extracted text
                    text = ''.join(char for char in text if char.isprintable() or char in '\n\t')
                    return text
        except Exception as e:
            logger.error(f"Error reading DOC {file_path}: {str(e)}")
            raise
    
    def extract_from_txt(self, file_path: Path) -> str:
        """Extract text from TXT file"""
        try:
            # Try different encodings
            encodings = ['utf-8', 'utf-16', 'latin-1', 'cp1252', 'iso-8859-1']
            
            for encoding in encodings:
                try:
                    with open(file_path, 'r', encoding=encoding) as file:
                        return file.read()
                except (UnicodeDecodeError, UnicodeError):
                    continue
            
            # If all encodings fail, read as binary and ignore errors
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
                logger.warning(f"Using fallback encoding for {file_path}")
                return file.read()
        
        except Exception as e:
            logger.error(f"Error reading TXT {file_path}: {str(e)}")
            raise
    
    def extract_from_csv(self, file_path: Path) -> str:
        """Extract text from CSV file"""
        try:
            text_content = []
            
            # Try different encodings
            encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
            
            for encoding in encodings:
                try:
                    with open(file_path, 'r', encoding=encoding, newline='') as file:
                        # Detect delimiter
                        sample = file.read(1024)
                        file.seek(0)
                        sniffer = csv.Sniffer()
                        try:
                            dialect = sniffer.sniff(sample)
                            delimiter = dialect.delimiter
                        except:
                            delimiter = ','
                        
                        csv_reader = csv.reader(file, delimiter=delimiter)
                        
                        for row_num, row in enumerate(csv_reader):
                            # Join row cells with pipe separator
                            row_text = ' | '.join(str(cell).strip() for cell in row if cell)
                            if row_text:
                                text_content.append(row_text)
                        
                        return '\n'.join(text_content)
                
                except UnicodeDecodeError:
                    continue
            
            return '\n'.join(text_content)
        
        except Exception as e:
            logger.error(f"Error reading CSV {file_path}: {str(e)}")
            raise
    
    def extract_from_eml(self, file_path: Path) -> str:
        """Extract text from EML email file"""
        try:
            with open(file_path, 'rb') as file:
                msg = BytesParser(policy=policy.default).parse(file)
            
            text_content = []
            
            # Extract headers
            text_content.append(f"From: {msg.get('From', '')}")
            text_content.append(f"To: {msg.get('To', '')}")
            text_content.append(f"Subject: {msg.get('Subject', '')}")
            text_content.append(f"Date: {msg.get('Date', '')}")
            text_content.append("\n" + "="*50 + "\n")
            
            # Extract body
            if msg.is_multipart():
                for part in msg.walk():
                    content_type = part.get_content_type()
                    
                    if content_type == 'text/plain':
                        try:
                            body = part.get_content()
                            text_content.append(body)
                        except Exception as e:
                            logger.warning(f"Error extracting email part: {str(e)}")
                    
                    elif content_type == 'text/html':
                        # Extract text from HTML if plain text not available
                        try:
                            from bs4 import BeautifulSoup
                            html_body = part.get_content()
                            soup = BeautifulSoup(html_body, 'html.parser')
                            text_content.append(soup.get_text())
                        except ImportError:
                            logger.warning("BeautifulSoup not installed, skipping HTML parsing")
                        except Exception as e:
                            logger.warning(f"Error parsing HTML: {str(e)}")
            else:
                try:
                    body = msg.get_content()
                    text_content.append(body)
                except Exception as e:
                    logger.warning(f"Error extracting email body: {str(e)}")
            
            return '\n\n'.join(text_content)
        
        except Exception as e:
            logger.error(f"Error reading EML {file_path}: {str(e)}")
            raise
    
    def extract_from_msg(self, file_path: Path) -> str:
        """Extract text from MSG Outlook file"""
        try:
            msg = extract_msg.Message(str(file_path))
            
            text_content = []
            
            # Extract headers
            text_content.append(f"From: {msg.sender or ''}")
            text_content.append(f"To: {msg.to or ''}")
            text_content.append(f"Subject: {msg.subject or ''}")
            text_content.append(f"Date: {msg.date or ''}")
            text_content.append("\n" + "="*50 + "\n")
            
            # Extract body
            if msg.body:
                text_content.append(msg.body)
            
            # Extract attachments info (but don't process them for now)
            if msg.attachments:
                text_content.append(f"\n\nAttachments: {len(msg.attachments)}")
                for attachment in msg.attachments:
                    text_content.append(f"  - {attachment.longFilename or attachment.shortFilename}")
            
            msg.close()
            
            return '\n\n'.join(text_content)
        
        except Exception as e:
            logger.error(f"Error reading MSG {file_path}: {str(e)}")
            raise