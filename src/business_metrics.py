"""
Business metrics for recommendation system
Phase 1: Relevance + Diversity
"""
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from typing import Dict, List, Optional

from config.config_loader import get_config


class BusinessMetrics:
    """
    Calculates business metrics for recommendations

    Phase 1 implements:
    - Relevance Score
    - Diversity Score
    """

    @staticmethod
    def relevance_score(
        recommended_track_ids: List[str],
        user_history_ids: List[str],
        embeddings_dict: Dict[str, np.ndarray]
    ) -> float:
        """
        Measures how similar recommendations are to the user's history

        Args:
            recommended_track_ids: IDs of recommended tracks
            user_history_ids: IDs of tracks in user's history
            embeddings_dict: Dict {track_id: embedding_vector}

        Returns:
            Score 0-1 (greater = more relevant)
        """
        # Get embeddings
        rec_embeddings = np.array([
            embeddings_dict[tid] for tid in recommended_track_ids
            if tid in embeddings_dict
        ])

        hist_embeddings = np.array([
            embeddings_dict[tid] for tid in user_history_ids
            if tid in embeddings_dict
        ])

        if len(rec_embeddings) == 0 or len(hist_embeddings) == 0:
            return 0.0

        # Cosine similarity between each recommendation and the history
        similarities = cosine_similarity(rec_embeddings, hist_embeddings)

        # Average of the best similarities (max per recommendation)
        max_similarities = similarities.max(axis=1)
        relevance = max_similarities.mean()

        return float(relevance)

    @staticmethod
    def diversity_score(
        recommended_genres: List[str],
        user_history_genres: List[str]
    ) -> float:
        """
        Measures diversity of genres in recommendations vs history

        Args:
            recommended_genres: Genres of recommended tracks
            user_history_genres: Genres in user's history

        Returns:
            Score 0-1 (0=no diversity, 1=all new genres)
        """
        rec_genres = set(recommended_genres)
        hist_genres = set(user_history_genres)

        if len(rec_genres) == 0:
            return 0.0

        # New genres introduced
        new_genres = rec_genres - hist_genres

        diversity = len(new_genres) / len(rec_genres)

        return float(diversity)

    @staticmethod
    def composite_business_score(
        relevance: float,
        diversity: float,
        w_relevance: Optional[float] = None,
        w_diversity: Optional[float] = None,
        relevance_threshold: Optional[float] = None
    ) -> Dict[str, float]:
        """
        Composite business score with penalty for low relevance

        Args:
            relevance: Relevance score (0-1)
            diversity: Diversity score (0-1)
            w_relevance: Weight of relevance
            w_diversity: Weight of diversity
            relevance_threshold: Penalize if relevance < threshold

        Returns:
            Dict with score and components
        """
        # Load defaults from config if not provided
        cfg = get_config()
        if w_relevance is None:
            w_relevance = cfg.get('recommender.strategies.balanced.w_relevance', 0.6)
        if w_diversity is None:
            w_diversity = cfg.get('recommender.strategies.balanced.w_diversity', 0.4)
        if relevance_threshold is None:
            relevance_threshold = cfg.get('recommender.relevance_threshold', 0.5)

        # Penalty if relevance very low
        if relevance < relevance_threshold:
            penalty = relevance / relevance_threshold  # Penalizes proportionally
        else:
            penalty = 1.0

        # Composite Score
        score = (w_relevance * relevance + w_diversity * diversity) * penalty

        return {
            'composite_score': float(score),
            'relevance': float(relevance),
            'diversity': float(diversity),
            'penalty_applied': penalty < 1.0,
            'penalty_factor': float(penalty)
        }

    @staticmethod
    def evaluate_recommendations(
        recommendations_df: pd.DataFrame,
        user_history_df: pd.DataFrame,
        embeddings_dict: Dict[str, np.ndarray],
        strategy_weights: Dict[str, float] = None
    ) -> Dict[str, float]:
        """
        Evaluation of a set of recommendations

        Args:
            recommendations_df: DataFrame with columns [track_id, general_genre]
            user_history_df: DataFrame with columns [track_id, general_genre]
            embeddings_dict: Dict of embeddings
            strategy_weights: Dict with w_relevance, w_diversity

        Returns:
            Dict with all metrics
        """
        if strategy_weights is None:
            cfg = get_config()
            strategy_weights = {
                'w_relevance': cfg.get('recommender.strategies.balanced.w_relevance', 0.6),
                'w_diversity': cfg.get('recommender.strategies.balanced.w_diversity', 0.4)
            }

        # Calculate individual metrics
        relevance = BusinessMetrics.relevance_score(
            recommendations_df['track_id'].tolist(),
            user_history_df['track_id'].tolist(),
            embeddings_dict
        )

        diversity = BusinessMetrics.diversity_score(
            recommendations_df['general_genre'].tolist(),
            user_history_df['general_genre'].tolist()
        )

        # Composite Score
        composite = BusinessMetrics.composite_business_score(
            relevance,
            diversity,
            w_relevance=strategy_weights['w_relevance'],
            w_diversity=strategy_weights['w_diversity']
        )

        return composite


def format_metrics_report(metrics: Dict[str, float]) -> str:
    """
    Format metrics for console display
    """
    report = f"""
    📊 BUSINESS METRICS REPORT
    {'='*50}

    🎯 Composite Score:     {metrics['composite_score']:.3f}

    Component Scores:
    - Relevance:          {metrics['relevance']:.3f}
    - Diversity:          {metrics['diversity']:.3f}

    {'⚠️ Penalty Applied' if metrics['penalty_applied'] else '✅ No Penalty'}
    """
    return report