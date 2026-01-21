"""
Recommendation system based on similarity with business metrics
"""
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Tuple
import pickle

from .business_metrics import BusinessMetrics
from config.business_config import RecommendationStrategy, get_strategy


class MusicRecommender:
    """
    Recomendation system with similarity based on optimization by business metrics
    """

    def __init__(
        self,
        embeddings_path: str,
        track_ids_path: str,
        dataset_path: str,
        strategy: str = 'balanced'
    ):
        """
        Args:
            embeddings_path: Path to track_embeddings.npy
            track_ids_path: Path to track_ids.npy
            dataset_path: Path to the CSV dataset
            strategy: Name of the strategy ('balanced', 'retention', 'discovery')
        """
        print("Loading recommendation system...")

        # Load embeddings
        self.embeddings = np.load(embeddings_path)
        self.track_ids = np.load(track_ids_path, allow_pickle=True)

        # Load dataset
        self.df = pd.read_csv(dataset_path)

        # Create dict for fast lookup
        self.track_id_to_idx = {tid: idx for idx, tid in enumerate(self.track_ids)}
        self.embeddings_dict = {
            tid: self.embeddings[idx]
            for tid, idx in self.track_id_to_idx.items()
        }

        # Load strategy
        self.strategy = get_strategy(strategy)

        print(f"✅ System loaded:")
        print(f"   - {len(self.track_ids):,} tracks")
        print(f"   - Strategy: {self.strategy.name}")
        print(f"   - Weights: Relevance={self.strategy.w_relevance}, "
            f"Diversity={self.strategy.w_diversity}")

    def get_track_info(self, track_id: str) -> pd.Series:
        """Get information of a track"""
        return self.df[self.df['track_id'] == track_id].iloc[0]

    def find_similar_tracks(
        self,
        track_id: str,
        n_recommendations: int = 10,
        exclude_ids: List[str] = None
    ) -> pd.DataFrame:
        """
        Finds similar tracks based on embeddings

        Args:
            track_id: ID of the base track
            n_recommendations: Number of recommendations
            exclude_ids: IDs to exclude (e.g., user history)

        Returns:
            DataFrame with recommendations and similarity scores
        """
        if track_id not in self.track_id_to_idx:
            raise ValueError(f"Track ID {track_id} not found in dataset")

        # Embedding of the base track
        idx = self.track_id_to_idx[track_id]
        query_embedding = self.embeddings[idx].reshape(1, -1)

        # Calculate similarities
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]

        # Order by similarity (excluding the track itself)
        similar_indices = np.argsort(similarities)[::-1]

        # Filter
        recommendations = []
        for sim_idx in similar_indices:
            sim_track_id = self.track_ids[sim_idx]

            # Exclude the track itself
            if sim_track_id == track_id:
                continue

            # Exclude tracks from history if specified
            if exclude_ids and sim_track_id in exclude_ids:
                continue

            recommendations.append({
                'track_id': sim_track_id,
                'similarity': similarities[sim_idx]
            })

            if len(recommendations) >= n_recommendations:
                break

        # Create DataFrame with complete information
        rec_df = pd.DataFrame(recommendations)
        rec_df = rec_df.merge(
            self.df[['track_id', 'name', 'general_genre', 'popularity', 'artist_popularity']],
            on='track_id'
        )

        return rec_df

    def recommend_for_user_history(
        self,
        user_history_ids: List[str],
        n_recommendations: int = 10,
        evaluate: bool = True
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Generates recommendations based on user history

        Args:
            user_history_ids: List of track_ids from the user history
            n_recommendations: Number of recommendations
            evaluate: If True, evaluates with business metrics

        Returns:
            (recommendations_df, metrics_dict)
        """
        # Calculate average embedding of the history
        history_embeddings = np.array([
            self.embeddings_dict[tid]
            for tid in user_history_ids
            if tid in self.embeddings_dict
        ])

        if len(history_embeddings) == 0:
            raise ValueError("No valid tracks in user history")

        user_profile_embedding = history_embeddings.mean(axis=0).reshape(1, -1)

        # Calculate similarities with all tracks
        similarities = cosine_similarity(user_profile_embedding, self.embeddings)[0]

        # Order by similarity
        similar_indices = np.argsort(similarities)[::-1]

        # Filter (exclude tracks from history)
        recommendations = []
        for sim_idx in similar_indices:
            sim_track_id = self.track_ids[sim_idx]

            if sim_track_id in user_history_ids:
                continue

            recommendations.append({
                'track_id': sim_track_id,
                'similarity': similarities[sim_idx]
            })

            if len(recommendations) >= n_recommendations:
                break

        # DataFrame with complete information
        rec_df = pd.DataFrame(recommendations)
        rec_df = rec_df.merge(
            self.df[['track_id', 'name', 'general_genre', 'popularity', 'artist_popularity']],
            on='track_id'
        )

        # Evaluate with business metrics
        metrics = {}
        if evaluate:
            user_history_df = self.df[self.df['track_id'].isin(user_history_ids)]

            metrics = BusinessMetrics.evaluate_recommendations(
                rec_df,
                user_history_df,
                self.embeddings_dict,
                strategy_weights={
                    'w_relevance': self.strategy.w_relevance,
                    'w_diversity': self.strategy.w_diversity
                }
            )

        return rec_df, metrics

    def change_strategy(self, strategy_name: str):
        """Changes the recommendation strategy"""
        self.strategy = get_strategy(strategy_name)
        print(f"✅ Strategy changed to: {self.strategy.name}")
        print(f"   Weights: Relevance={self.strategy.w_relevance}, "
            f"Diversity={self.strategy.w_diversity}")