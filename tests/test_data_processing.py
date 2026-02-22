"""
Tests for data processing pipeline
"""

import pytest
import pandas as pd
import numpy as np
from src.data_processing import DatasetProcessor
from config.genre_mapping import GENRE_GROUPS


class TestDatasetProcessor:
    """Test suite for DatasetProcessor"""

    @pytest.fixture
    def processor(self):
        """Create a DatasetProcessor instance"""
        return DatasetProcessor(
            genre_mapping=GENRE_GROUPS,
            min_tracks_per_genre=10,
            max_tracks_per_genre=100,
            random_state=64
        )

    def test_parse_genres_valid(self, processor):
        """Test parsing valid genre strings"""
        genre_string = "['pop', 'rock', 'indie']"
        result = processor.parse_genres(genre_string)

        assert result == ['pop', 'rock', 'indie']

    def test_parse_genres_empty(self, processor):
        """Test parsing empty genre string"""
        genre_string = "[]"
        result = processor.parse_genres(genre_string)

        assert result == []

    def test_parse_genres_invalid(self, processor):
        """Test parsing invalid genre string"""
        genre_string = "not a list"
        result = processor.parse_genres(genre_string)

        assert result == []

    def test_map_to_general_genre_rock(self, processor):
        """Test mapping rock genres"""
        genres = ['rock', 'classic rock']
        result = processor.map_to_general_genre(genres)

        assert result == 'Rock'

    def test_map_to_general_genre_pop(self, processor):
        """Test mapping pop genres"""
        genres = ['pop', 'dance pop']
        result = processor.map_to_general_genre(genres)

        assert result == 'Pop'

    def test_map_to_general_genre_unknown(self, processor):
        """Test mapping unknown genres"""
        genres = ['unknown-genre-xyz']
        result = processor.map_to_general_genre(genres)

        assert result == 'Other'

    def test_map_to_general_genre_empty(self, processor):
        """Test mapping empty genre list"""
        genres = []
        result = processor.map_to_general_genre(genres)

        assert result is None

    def test_filter_valid_genres(self, processor):
        """Test filtering valid genres"""
        # Create sample dataframe
        df = pd.DataFrame({
            'general_genre': ['Rock', 'Pop', None, 'Other', 'Jazz'],
            'track_id': range(5)
        })

        result = processor.filter_valid_genres(df)

        # Should keep Rock, Pop, Jazz (3 tracks)
        assert len(result) == 3
        assert 'Other' not in result['general_genre'].values
        assert result['general_genre'].isna().sum() == 0