"""
User and behavior simulator for recommendation evaluation
"""
import pandas as pd
import numpy as np
from typing import List, Optional
from dataclasses import dataclass

from config.config_loader import get_config
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class SimulatedUser:
    """User simulated with track history"""
    user_id: int
    favorite_genre: str
    history_track_ids: List[str]
    history: pd.DataFrame  # DataFrame with tracks from history

class UserSimulator:
    """
    Generates simulated users with realistic listening histories
    """

    def __init__(self, tracks_df: pd.DataFrame, seed: int = 42):
        """
        Args:
            tracks_df: Complete tracks DataFrame
            seed: Random seed for reproducibility
        """
        self.tracks_df = tracks_df
        self.rng = np.random.RandomState(seed)

        # Calculate genre distribution (for realistic sampling)
        self.genre_distribution = (
            tracks_df['general_genre']
            .value_counts(normalize=True)
            .to_dict()
        )

    def generate_users(
        self,
        n_users: Optional[int] = None,
        tracks_per_user: Optional[int] = None,
        favorite_genre_ratio: Optional[float] = None
    ) -> List[SimulatedUser]:
        """
        Generates simulated users with realistic listening histories

        Args:
            n_users: Number of users to generate
            tracks_per_user: Tracks in each user's history
            favorite_genre_ratio: % of tracks from favorite genre (0.7 = 70%)

        Returns:
            SimulatedUser list
        """
        # Load defaults from config if not provided
        cfg = get_config()
        if n_users is None:
            n_users = cfg.get('evaluation.n_simulated_users', 1000)
        if tracks_per_user is None:
            tracks_per_user = cfg.get('evaluation.tracks_per_user', 20)
        if favorite_genre_ratio is None:
            favorite_genre_ratio = cfg.get('evaluation.favorite_genre_ratio', 0.7)
        users = []

        logger.info(f"Generating {n_users} simulated users...")

        for user_id in range(n_users):
            # Select favorite genre (weighted by real distribution)
            genres = list(self.genre_distribution.keys())
            probs = list(self.genre_distribution.values())
            fav_genre = self.rng.choice(genres, p=probs)

            # Calculate how many tracks of each type
            n_fav = int(tracks_per_user * favorite_genre_ratio)
            n_other = tracks_per_user - n_fav

            # Sample tracks from favorite genre
            fav_tracks = self.tracks_df[
                self.tracks_df['general_genre'] == fav_genre
            ]

            if len(fav_tracks) < n_fav:
                # If there aren't enough, take what's available
                sampled_fav = fav_tracks
                n_fav = len(fav_tracks)
                n_other = tracks_per_user - n_fav
            else:
                sampled_fav = fav_tracks.sample(n=n_fav, random_state=user_id)

            # Sample tracks from other genres
            other_tracks = self.tracks_df[
                self.tracks_df['general_genre'] != fav_genre
            ]
            sampled_other = other_tracks.sample(n=n_other, random_state=user_id)

            # Combine history
            history = pd.concat([sampled_fav, sampled_other], ignore_index=True)

            # Create user
            user = SimulatedUser(
                user_id=user_id,
                favorite_genre=fav_genre,
                history_track_ids=history['track_id'].tolist(),
                history=history
            )

            users.append(user)

        logger.info(f"✅ {len(users)} simulated users generated")
        self._print_distribution_summary(users)

        return users

    def _print_distribution_summary(self, users: List[SimulatedUser]):
        """Logs summary of favorite genre distribution"""
        fav_genres = [u.favorite_genre for u in users]
        distribution = pd.Series(fav_genres).value_counts()

        logger.debug("📊 Favorite Genre Distribution:")
        for genre, count in distribution.items():
            pct = (count / len(users)) * 100
            logger.debug(f"   {genre:15s}: {count:4d} ({pct:5.1f}%)")