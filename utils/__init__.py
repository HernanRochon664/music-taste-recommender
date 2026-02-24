"""
Utility modules for the project
"""

from .logger import get_logger, setup_logger
from .download_embeddings import (
    download_embeddings_from_hf,
    ensure_embeddings_available
)

__all__ = [
    'get_logger',
    'setup_logger',
    'download_embeddings_from_hf',
    'ensure_embeddings_available'
]