"""
File utility functions
"""

from pathlib import Path
from app.core.config import settings


def allowed_file(filename: str) -> bool:
    """
    Check if file extension is allowed
    
    Args:
        filename: Name of the file to check
        
    Returns:
        True if file extension is in ALLOWED_EXTENSIONS, False otherwise
    """
    if not filename:
        return False
    
    file_extension = Path(filename).suffix.lower()
    return file_extension in settings.ALLOWED_EXTENSIONS