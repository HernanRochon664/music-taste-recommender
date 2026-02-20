"""
Data processing logic for dataset preparation

Handles:
- Genre parsing and mapping
- Audio features filtering
- Dataset balancing
"""

import pandas as pd
import ast
from typing import Optional, Dict, List
from pathlib import Path

from config.config_loader import get_config
from config.genre_mapping import MIN_TRACKS_PER_GENRE, MAX_TRACKS_PER_GENRE
from utils.logger import get_logger

logger = get_logger(__name__)


class DatasetProcessor:
    """
    Processes raw Spotify dataset into clean, balanced format
    """

    def __init__(
        self,
        genre_mapping: Dict[str, List[str]],
        min_tracks_per_genre: Optional[int] = None,
        max_tracks_per_genre: Optional[int] = None,
        random_state: Optional[int] = None
    ):
        """
        Args:
            genre_mapping: Dict mapping general genres to specific genres
            min_tracks_per_genre: Minimum tracks to include a genre
            max_tracks_per_genre: Maximum tracks to keep per genre
            random_state: Random seed for reproducibility
        """
        # Load defaults from configuration if not provided
        cfg = get_config()
        if min_tracks_per_genre is None:
            min_tracks_per_genre = cfg.get('data_processing.min_tracks_per_genre', MIN_TRACKS_PER_GENRE)
        if max_tracks_per_genre is None:
            max_tracks_per_genre = cfg.get('data_processing.max_tracks_per_genre', MAX_TRACKS_PER_GENRE)
        if random_state is None:
            random_state = cfg.get('data_processing.random_state', 42)

        self.genre_mapping = genre_mapping
        self.min_tracks = min_tracks_per_genre
        self.max_tracks = max_tracks_per_genre
        self.random_state = random_state

        # Audio features to validate
        self.audio_features = [
            'acousticness', 'danceability', 'energy',
            'instrumentalness', 'liveness', 'loudness',
            'speechiness', 'tempo', 'valence', 'key'
        ]

        # Final columns to keep
        self.final_columns = [
            'track_id', 'name',
            'acousticness', 'danceability', 'energy',
            'instrumentalness', 'key', 'liveness', 'loudness',
            'speechiness', 'tempo', 'valence',
            'general_genre', 'genres_list',
            'popularity', 'artist_popularity',
            'duration_ms', 'explicit', 'album_release_date'
        ]

    @staticmethod
    def parse_genres(genre_string) -> List[str]:
        """
        Parse genre string to list

        Args:
            genre_string: String representation of list (e.g., "['pop', 'rock']")

        Returns:
            List of genres or empty list if parsing fails
        """
        try:
            return ast.literal_eval(genre_string)
        except:
            return []

    def map_to_general_genre(self, genre_list: List[str]) -> Optional[str]:
        """
        Map specific genres to general category

        Args:
            genre_list: List of specific genres

        Returns:
            General genre category or None
        """
        if not genre_list or genre_list == []:
            return None

        for general, specifics in self.genre_mapping.items():
            for genre in genre_list:
                if genre.lower() in specifics:
                    return general

        return 'Other'

    def load_and_parse(self, filepath: str) -> pd.DataFrame:
        """
        Load raw dataset and parse genres

        Args:
            filepath: Path to raw CSV

        Returns:
            DataFrame with parsed genres
        """
        logger.info(f"📂 Loading dataset from {filepath}...")
        df = pd.read_csv(filepath)
        logger.debug(f"   Loaded: {df.shape}")

        logger.info("🔍 Parsing genres...")
        df['genres_list'] = df['genres'].apply(self.parse_genres)
        df['general_genre'] = df['genres_list'].apply(self.map_to_general_genre)

        return df

    def filter_valid_genres(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter tracks with valid genre mappings

        Args:
            df: DataFrame with general_genre column

        Returns:
            Filtered DataFrame
        """
        logger.info("🎯 Filtering valid genres...")

        # Show initial distribution
        logger.debug("   Initial genre distribution:")
        genre_dist = df['general_genre'].value_counts()
        for genre, count in genre_dist.head(15).items():
            logger.debug(f"      {genre:20s}: {count:,}")

        # Filter
        df_filtered = df[df['general_genre'].notna()].copy()
        df_filtered = df_filtered[df_filtered['general_genre'] != 'Other']

        logger.debug(f"   ✅ Tracks after filtering: {len(df_filtered):,}")

        return df_filtered

    def filter_complete_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter tracks with complete audio features

        Args:
            df: DataFrame with audio feature columns

        Returns:
            DataFrame with complete features
        """
        logger.info("🎵 Filtering complete audio features...")

        initial_count = len(df)
        df_clean = df.dropna(subset=self.audio_features)

        logger.debug(f"   Tracks with complete features: {len(df_clean):,}")
        logger.debug(f"   Dropped: {initial_count - len(df_clean):,}")

        return df_clean

    def balance_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Balance dataset by genre

        Args:
            df: DataFrame with general_genre column

        Returns:
            Balanced DataFrame
        """
        logger.info("⚖️  Balancing dataset...")
        logger.debug(f"   Min tracks per genre: {self.min_tracks:,}")
        logger.debug(f"   Max tracks per genre: {self.max_tracks:,}")

        balanced_dfs = []
        discarded_genres = []

        for genre in sorted(df['general_genre'].unique()):
            genre_df = df[df['general_genre'] == genre]
            n_tracks = len(genre_df)

            if n_tracks < self.min_tracks:
                logger.debug(f"   ❌ {genre:20s}: {n_tracks:6,} tracks (< min, discarding)")
                discarded_genres.append(genre)
                continue
            elif n_tracks > self.max_tracks:
                genre_df = genre_df.sample(
                    n=self.max_tracks,
                    random_state=self.random_state
                )
                logger.debug(f"   📉 {genre:20s}: {n_tracks:6,} → {self.max_tracks:6,} tracks")
            else:
                logger.debug(f"   ✅ {genre:20s}: {n_tracks:6,} tracks (kept)")

            balanced_dfs.append(genre_df)

        df_balanced = pd.concat(balanced_dfs, ignore_index=True)

        # Shuffle
        df_balanced = df_balanced.sample(
            frac=1,
            random_state=self.random_state
        ).reset_index(drop=True)

        logger.debug(f"   Final balanced dataset: {len(df_balanced):,} tracks")
        logger.debug(f"   Genres included: {df_balanced['general_genre'].nunique()}")

        if discarded_genres:
            logger.debug(f"   Discarded genres: {', '.join(discarded_genres)}")

        return df_balanced

    def select_final_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Select final columns for output

        Args:
            df: Full DataFrame

        Returns:
            DataFrame with selected columns
        """
        logger.info("📋 Selecting final columns...")
        df_final = df[self.final_columns].copy()
        logger.debug(f"   Columns: {len(self.final_columns)}")

        return df_final

    def process(
        self,
        input_path: str,
        output_path: str,
        show_stats: bool = True
    ) -> pd.DataFrame:
        """
        Complete processing pipeline

        Args:
            input_path: Path to raw CSV
            output_path: Path to save processed CSV
            show_stats: Whether to print final statistics

        Returns:
            Processed DataFrame
        """
        logger.info("="*60)
        logger.info("DATASET PROCESSING PIPELINE")
        logger.info("="*60)

        # Step 1: Load and parse
        df = self.load_and_parse(input_path)

        # Step 2: Filter valid genres
        df = self.filter_valid_genres(df)

        # Step 3: Filter complete features
        df = self.filter_complete_features(df)

        # Step 4: Balance
        df = self.balance_dataset(df)

        # Step 5: Select columns
        df_final = self.select_final_columns(df)

        # Step 6: Save
        logger.info("💾 Saving processed dataset...")
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df_final.to_csv(output_path, index=False)

        logger.info(f"   ✅ Saved to: {output_path}")
        logger.info(f"   Shape: {df_final.shape}")
        logger.info(f"   Size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")

        # Step 7: Stats
        if show_stats:
            self._print_statistics(df_final)

        logger.info("="*60)
        logger.info("✅ PROCESSING COMPLETE")
        logger.info("="*60)

        return df_final

    def _print_statistics(self, df: pd.DataFrame):
        """Log final dataset statistics"""
        logger.debug("")
        logger.debug("="*60)
        logger.debug("FINAL DATASET STATISTICS")
        logger.debug("="*60)

        logger.debug("📊 Genre distribution:")
        genre_dist = df['general_genre'].value_counts().sort_index()
        for genre, count in genre_dist.items():
            pct = (count / len(df)) * 100
            logger.debug(f"   {genre:20s}: {count:6,} ({pct:5.1f}%)")

        logger.debug("🎵 Audio features statistics:")
        logger.debug(df[self.audio_features].describe().round(3).to_string())