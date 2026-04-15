"""
File utility functions
"""

from pathlib import Path
from app.core.config import settings


def allowed_file(filename: str) -> bool:
    """
    Check if file extension is allowed
    
    Args:
        filename: Name of the file
        
    Returns:
        True if allowed, False otherwise
    """
    return Path(filename).suffix.lower() in settings.ALLOWED_EXTENSIONS