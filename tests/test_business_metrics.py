"""
Tests for business metrics calculation
"""

import pytest
import numpy as np
from src.business_metrics import BusinessMetrics


class TestBusinessMetrics:
    """Test suite for business metrics"""

    def test_relevance_score_perfect_match(self):
        """Test relevance with identical embeddings"""
        # Create identical embeddings
        embeddings_dict = {
            'track1': np.array([1.0, 0.0, 0.0]),
            'track2': np.array([1.0, 0.0, 0.0]),
            'track3': np.array([1.0, 0.0, 0.0])
        }

        recommended = ['track2', 'track3']
        history = ['track1']

        score = BusinessMetrics.relevance_score(
            recommended,
            history,
            embeddings_dict
        )

        # Perfect match should give score = 1.0
        assert score == pytest.approx(1.0, abs=0.01)

    def test_relevance_score_orthogonal(self):
        """Test relevance with orthogonal embeddings"""
        embeddings_dict = {
            'track1': np.array([1.0, 0.0, 0.0]),
            'track2': np.array([0.0, 1.0, 0.0]),
            'track3': np.array([0.0, 0.0, 1.0])
        }

        recommended = ['track2', 'track3']
        history = ['track1']

        score = BusinessMetrics.relevance_score(
            recommended,
            history,
            embeddings_dict
        )

        # Orthogonal vectors should give score = 0.0
        assert score == pytest.approx(0.0, abs=0.01)

    def test_diversity_score_no_overlap(self):
        """Test diversity when all genres are different"""
        recommended_genres = ['Rock', 'Jazz', 'Classical']
        history_genres = ['Pop', 'Hip Hop']

        score = BusinessMetrics.diversity_score(
            recommended_genres,
            history_genres
        )

        # All genres are new → diversity = 1.0
        assert score == 1.0

    def test_diversity_score_complete_overlap(self):
        """Test diversity when all genres are the same"""
        recommended_genres = ['Rock', 'Rock', 'Rock']
        history_genres = ['Rock']

        score = BusinessMetrics.diversity_score(
            recommended_genres,
            history_genres
        )

        # All same genre → diversity = 0.0
        assert score == 0.0

    def test_diversity_score_partial_overlap(self):
        """Test diversity with 50% overlap"""
        recommended_genres = ['Rock', 'Jazz']
        history_genres = ['Rock']

        score = BusinessMetrics.diversity_score(
            recommended_genres,
            history_genres
        )

        # 1 new genre out of 2 unique → diversity = 0.5
        assert score == 0.5

    def test_composite_score_balanced(self):
        """Test composite score with balanced weights"""
        relevance = 0.8
        diversity = 0.6

        result = BusinessMetrics.composite_business_score(
            relevance,
            diversity,
            w_relevance=0.5,
            w_diversity=0.5
        )

        expected = 0.5 * 0.8 + 0.5 * 0.6  # = 0.7
        assert result['composite_score'] == pytest.approx(expected, abs=0.01)
        assert result['penalty_applied'] == False

    def test_composite_score_with_penalty(self):
        """Test composite score applies penalty for low relevance"""
        relevance = 0.4  # Below threshold of 0.5
        diversity = 0.8

        result = BusinessMetrics.composite_business_score(
            relevance,
            diversity,
            w_relevance=0.6,
            w_diversity=0.4,
            relevance_threshold=0.5
        )

        # Should apply penalty
        assert result['penalty_applied'] == True
        assert result['penalty_factor'] < 1.0

        # Composite should be lower due to penalty
        no_penalty_score = 0.6 * 0.4 + 0.4 * 0.8
        assert result['composite_score'] < no_penalty_score

    def test_diversity_score_empty_recommendations(self):
        """Test diversity with empty recommendations"""
        score = BusinessMetrics.diversity_score([], ['Rock'])
        assert score == 0.0