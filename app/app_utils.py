"""
Utility functions for Streamlit demo
"""

import pandas as pd
import streamlit as st
from pathlib import Path
import sys

# Add parent to path
sys.path.append(str(Path(__file__).parent.parent))

from src.recommender import MusicRecommender


@st.cache_resource
def load_recommender(embeddings_path, track_ids_path, dataset_path):
    """
    Load recommender system (cached)

    Note: Strategy is NOT cached - it's changed dynamically via change_strategy()
    """
    recommender = MusicRecommender(
        str(embeddings_path),
        str(track_ids_path),
        str(dataset_path),
        strategy='balanced'
    )
    return recommender


@st.cache_data
def load_dataset(dataset_path):
    """
    Load dataset (cached)
    """
    return pd.read_csv(dataset_path)


def get_popular_tracks(df, n=100):
    """
    Get most popular tracks for seed selection
    """
    return df.nlargest(n, 'popularity')[['track_id', 'name', 'general_genre', 'popularity']]


def format_track_name(row):
    """
    Format track name for display
    """
    return f"{row['name']} | {row['general_genre']} (popularity: {row['popularity']:.0f})"