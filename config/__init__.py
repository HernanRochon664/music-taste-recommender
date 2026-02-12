"""
Configuration module for Music Taste Recommender

Provides access to:
- YAML-based settings (settings.yaml)
- Business strategies (business_config.py)
- Genre mappings (genre_mapping.py)
"""

from .config_loader import Config, get_config
from .business_config import (
    STRATEGIES,
    RecommendationStrategy,
    get_strategy
)
from .genre_mapping import (
    GENRE_GROUPS,
    FINAL_GENRES,
    MIN_TRACKS_PER_GENRE,
    MAX_TRACKS_PER_GENRE
)

# Global config instance
config = get_config()

__all__ = [
    'config',
    'get_config',
    'Config',
    'STRATEGIES',
    'RecommendationStrategy',
    'get_strategy',
    'GENRE_GROUPS',
    'FINAL_GENRES',
    'MIN_TRACKS_PER_GENRE',
    'MAX_TRACKS_PER_GENRE'
]