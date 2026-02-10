"""
Configuration for Streamlit demo
"""

from pathlib import Path

# Paths
EMBEDDINGS_PATH = Path("data/embeddings/track_embeddings.npy")
TRACK_IDS_PATH = Path("data/embeddings/track_ids.npy")
DATASET_PATH = Path("data/processed/spotify_clean_balanced.csv")

# Demo settings
DEFAULT_STRATEGY = "balanced"
N_RECOMMENDATIONS = 10
N_SEED_TRACKS_TO_SHOW = 100  # Show top N popular tracks in selector

# Styling
GENRE_COLORS = {
    'Rock': '#e74c3c',
    'Pop': '#3498db',
    'Classical': '#9b59b6',
    'Hip Hop': '#f39c12',
    'Electronic': '#1abc9c',
    'Jazz': '#34495e',
    'Latin': '#e67e22',
    'Country': '#16a085',
    'Soundtrack': '#95a5a6'
}

# Strategy descriptions
STRATEGY_INFO = {
    'conservative': {
        'emoji': '🛡️',
        'name': 'Conservative',
        'description': 'Prioritizes user satisfaction. Best for new users or churn-sensitive segments.',
        'weights': 'Relevance: 70% | Diversity: 30%'
    },
    'balanced': {
        'emoji': '⚖️',
        'name': 'Balanced',
        'description': 'Optimal balance between satisfaction and exploration. General purpose.',
        'weights': 'Relevance: 50% | Diversity: 50%'
    },
    'discovery': {
        'emoji': '🔍',
        'name': 'Discovery',
        'description': 'Promotes catalog exploration. Higher engagement potential.',
        'weights': 'Relevance: 40% | Diversity: 60%'
    },
    'aggressive': {
        'emoji': '🚀',
        'name': 'Aggressive',
        'description': 'Maximum exploration. Best for power users seeking novelty.',
        'weights': 'Relevance: 30% | Diversity: 70%'
    }
}