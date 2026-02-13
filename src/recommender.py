"""
Recommendation system based on similarity with business metrics
"""
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Tuple, Optional
import pickle

from .business_metrics import BusinessMetrics
from config.business_config import RecommendationStrategy, get_strategy
from config.config_loader import get_config


class MusicRecommender:
    """
    Recomendation system with similarity based on optimization by business metrics
    """

    def __init__(
        self,
        embeddings_path: str,
        track_ids_path: str,
        dataset_path: str,
        strategy: str = 'balanced',
        artist_popularity_threshold: Optional[int] = None
    ):
        """
        Args:
            embeddings_path: Path to track_embeddings.npy
            track_ids_path: Path to track_ids.npy
            dataset_path: Path to the CSV dataset
            strategy: Name of the strategy ('balanced', 'retention', 'discovery')
            artist_popularity_threshold: Threshold to filter duplicates by artist popularity
        """
        print("Loading recommendation system...")

        # Load config for duplicate filtering threshold
        cfg = get_config()
        if artist_popularity_threshold is None:
            artist_popularity_threshold = cfg.get('recommender.duplicate_filter.artist_popularity_threshold', 10)

        self.artist_popularity_threshold = artist_popularity_threshold

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
        print(f"   - Duplicate filter threshold: {self.artist_popularity_threshold}")

    def get_track_info(self, track_id: str) -> pd.Series:
        """Get information of a track"""
        return self.df[self.df['track_id'] == track_id].iloc[0]

    def find_similar_tracks(
        self,
        track_id: str,
        n_recommendations: int = 10,
        n_candidates: int = 100,
        exclude_ids: List[str] = None
    ) -> pd.DataFrame:
        """
        Find similar tracks with re-ranking based on embeddings

        Uses a quota-based approach to balance relevance and diversity:
        - Separates candidates into same-genre and different-genre pools
        - Applies strategy-specific quotas
        - Orders by similarity within each pool

        Args:
            track_id: ID of the seed track
            n_recommendations: Number of final recommendations
            n_candidates: Number of candidates to consider for re-ranking
            exclude_ids: Track IDs to exclude (e.g., user history)

        Returns:
            DataFrame with recommendations and scores
        """
        if track_id not in self.track_id_to_idx:
            raise ValueError(f"Track ID {track_id} not found in dataset")

        # Get embedding of seed track
        idx = self.track_id_to_idx[track_id]
        query_embedding = self.embeddings[idx].reshape(1, -1)

        # Calculate similarities with all tracks
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]

        # Get top candidates sorted by similarity
        similar_indices = np.argsort(similarities)[::-1]

        # Get seed track info for filtering
        seed_info = self.get_track_info(track_id)
        seed_genre = seed_info['general_genre']
        seed_name = seed_info['name'].lower()
        seed_artist_popularity = seed_info.get('artist_popularity', 0)

        # Collect candidates with filtering
        candidates = []
        for sim_idx in similar_indices:
            sim_track_id = self.track_ids[sim_idx]

            # Exclude the seed track itself (by ID)
            if sim_track_id == track_id:
                continue

            # Exclude tracks from user history if specified
            if exclude_ids and sim_track_id in exclude_ids:
                continue

            track_info = self.get_track_info(sim_track_id)
            track_name = track_info['name'].lower()
            track_artist_popularity = track_info.get('artist_popularity', 0)

            # Filter duplicates (same track, different versions)
            # Allow if names are similar BUT artist popularity differs significantly
            # (likely different artists with same song name, e.g., covers)
            if seed_name in track_name or track_name in seed_name:
                artist_pop_diff = abs(seed_artist_popularity - track_artist_popularity)

                # If artist popularity is very similar, likely same artist/version
                if artist_pop_diff < self.artist_popularity_threshold:
                    continue

            track_genre = track_info['general_genre']

            candidates.append({
                'track_id': sim_track_id,
                'similarity': similarities[sim_idx],
                'genre': track_genre,
                'is_same_genre': track_genre == seed_genre
            })

            if len(candidates) >= n_candidates:
                break

        # Separate candidates by genre
        same_genre = [c for c in candidates if c['is_same_genre']]
        diff_genre = [c for c in candidates if not c['is_same_genre']]

        # Sort each pool by similarity (descending)
        same_genre.sort(key=lambda x: x['similarity'], reverse=True)
        diff_genre.sort(key=lambda x: x['similarity'], reverse=True)

        # Determine quotas based on strategy
        # Higher w_diversity = more different genres
        if self.strategy.w_diversity >= 0.6:  # Aggressive (30% same, 70% different)
            n_same = max(1, int(n_recommendations * 0.3))
            n_diff = n_recommendations - n_same
        elif self.strategy.w_diversity >= 0.5:  # Discovery (40% same, 60% different)
            n_same = max(2, int(n_recommendations * 0.4))
            n_diff = n_recommendations - n_same
        elif self.strategy.w_diversity >= 0.4:  # Balanced (50% same, 50% different)
            n_same = int(n_recommendations * 0.5)
            n_diff = n_recommendations - n_same
        else:  # Conservative/Retention (70% same, 30% different)
            n_same = int(n_recommendations * 0.7)
            n_diff = n_recommendations - n_same

        # Select according to quotas
        selected_same = same_genre[:n_same]
        selected_diff = diff_genre[:n_diff]

        # Fill remaining slots if one pool is exhausted
        total_selected = len(selected_same) + len(selected_diff)
        if total_selected < n_recommendations:
            # Need more tracks, take from remaining candidates
            remaining = same_genre[n_same:] + diff_genre[n_diff:]
            remaining.sort(key=lambda x: x['similarity'], reverse=True)
            needed = n_recommendations - total_selected
            selected_same.extend(remaining[:needed])

        # Combine and sort by similarity for presentation
        final_recommendations = selected_same + selected_diff
        final_recommendations.sort(key=lambda x: x['similarity'], reverse=True)
        final_recommendations = final_recommendations[:n_recommendations]

        # Calculate scores for visualization
        for rec in final_recommendations:
            diversity_component = 0.0 if rec['is_same_genre'] else 1.0
            combined_score = (
                self.strategy.w_relevance * rec['similarity'] +
                self.strategy.w_diversity * diversity_component
            )
            rec['diversity_component'] = diversity_component
            rec['combined_score'] = combined_score

        # Create DataFrame with complete information
        rec_df = pd.DataFrame([
            {
                'track_id': rec['track_id'],
                'similarity': rec['similarity'],
                'diversity_component': rec['diversity_component'],
                'combined_score': rec['combined_score']
            }
            for rec in final_recommendations
        ])

        rec_df = rec_df.merge(
            self.df[['track_id', 'name', 'general_genre', 'popularity', 'artist_popularity']],
            on='track_id'
        )

        return rec_df

    def recommend_for_user_history(
        self,
        user_history_ids: List[str],
        n_recommendations: int = 10,
        n_candidates: int = 100,
        evaluate: bool = True
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Generates recommendations based on user history with re-ranking

        Process:
        1. Get top-N candidates by similarity (N >> n_recommendations)
        2. For each candidate, calculate diversity score
        3. Re-rank combining similarity and diversity according to strategy
        4. Return top-K final recommendations

        Args:
            user_history_ids: List of track_ids from user history
            n_recommendations: Number of final recommendations
            n_candidates: Number of candidates to consider for re-ranking
            evaluate: If True, evaluates with business metrics

        Returns:
            (recommendations_df, metrics_dict)
        """
        # 1. Calculate embedding average of the user history (user profile)
        history_embeddings = np.array([
            self.embeddings_dict[tid]
            for tid in user_history_ids
            if tid in self.embeddings_dict
        ])

        if len(history_embeddings) == 0:
            raise ValueError("No valid tracks in user history")

        user_profile_embedding = history_embeddings.mean(axis=0).reshape(1, -1)

        # 2. Calculate similarities with all tracks
        similarities = cosine_similarity(user_profile_embedding, self.embeddings)[0]

        # 3. Get top-N candidates by similarity
        candidate_indices = np.argsort(similarities)[::-1]

        # Filter candidates (exclude history)
        candidates = []
        for idx in candidate_indices:
            track_id = self.track_ids[idx]

            if track_id in user_history_ids:
                continue

            candidates.append({
                'idx': idx,
                'track_id': track_id,
                'similarity': similarities[idx]
            })

            if len(candidates) >= n_candidates:
                break

        # 4. Get genres from history (to calculate diversity)
        user_history_df = self.df[self.df['track_id'].isin(user_history_ids)]
        user_history_genres = set(user_history_df['general_genre'].tolist())

        # 5. Re-ranking: combine similarity + diversity
        for candidate in candidates:
            track_id = candidate['track_id']
            track_info = self.get_track_info(track_id)
            track_genre = track_info['general_genre']

            # Calculate diversity: 1 if new genre, 0 if known genre
            is_new_genre = 1.0 if track_genre not in user_history_genres else 0.0

            # Combined score according to strategy
            combined_score = (
                self.strategy.w_relevance * candidate['similarity'] +
                self.strategy.w_diversity * is_new_genre
            )

            candidate['diversity_component'] = is_new_genre
            candidate['combined_score'] = combined_score

        # 6. Order by combined score and take top-K
        candidates_sorted = sorted(
            candidates,
            key=lambda x: x['combined_score'],
            reverse=True
        )

        top_recommendations = candidates_sorted[:n_recommendations]

        # 7. Create DataFrame with complete information
        rec_df = pd.DataFrame([
            {
                'track_id': rec['track_id'],
                'similarity': rec['similarity'],
                'diversity_component': rec['diversity_component'],
                'combined_score': rec['combined_score']
            }
            for rec in top_recommendations
        ])

        rec_df = rec_df.merge(
            self.df[['track_id', 'name', 'general_genre', 'popularity', 'artist_popularity']],
            on='track_id'
        )

        # 8. Evaluate with business metrics
        metrics = {}
        if evaluate:
            metrics = BusinessMetrics.evaluate_recommendations(
                rec_df,
                user_history_df,
                self.embeddings_dict,
                strategy_weights={
                    'w_relevance': self.strategy.w_relevance,
                    'w_diversity': self.strategy.w_diversity
                }
            )

            # Add additional info
            metrics['avg_similarity'] = rec_df['similarity'].mean()
            metrics['avg_combined_score'] = rec_df['combined_score'].mean()
            metrics['new_genres_count'] = rec_df['diversity_component'].sum()

        return rec_df, metrics

    def change_strategy(self, strategy_name: str):
        """Changes the recommendation strategy"""
        self.strategy = get_strategy(strategy_name)
        print(f"✅ Strategy changed to: {self.strategy.name}")
        print(f"   Weights: Relevance={self.strategy.w_relevance}, "
            f"Diversity={self.strategy.w_diversity}")