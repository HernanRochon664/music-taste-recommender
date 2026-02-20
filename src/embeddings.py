"""
Generation of hybrid embeddings Pipeline for tracks from Spotify
"""
import pandas as pd
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from config.config_loader import get_config
from sklearn.preprocessing import StandardScaler
import pickle
from typing import Dict, Tuple
from tqdm import tqdm

from .utils import (
    one_hot_encode_key,
    normalize_audio_features,
    combine_embeddings,
    l2_normalize
)

from utils.logger import get_logger

logger = get_logger(__name__)


class TrackEmbeddingGenerator:
    """
    Hybrid embeddings generator for tracks

    Combines:
    - Numeric audio features (6 features + 12 one-hot key)
    - Embeddings per genre (sentence-transformers)
    """

    def __init__(
        self,
        model_name: str = None,
        w_audio: float = None,
        w_genre: float = None
    ):
        """
        Args:
            model_name: Model of sentence-transformers
            w_audio: Weight for audio features
            w_genre: Weight for genre embeddings
        """
        # Load defaults from config when not provided
        cfg = get_config()
        if model_name is None:
            model_name = cfg.get('embeddings.model_name', 'all-MiniLM-L6-v2')
        if w_audio is None:
            w_audio = cfg.get('embeddings.audio_weight', 10.0)
        if w_genre is None:
            w_genre = cfg.get('embeddings.genre_weight', 1.0)

        self.model_name = model_name
        self.w_audio = w_audio
        self.w_genre = w_genre

        # load sentence-transformers model
        logger.info(f"Loading model {self.model_name}...")
        self.sentence_model = SentenceTransformer(self.model_name)

        # Scaler for audio features
        self.scaler = None

        # Batch size for encoding from config
        self.batch_size = cfg.get('embeddings.batch_size', 256)

        # Features to use
        self.audio_feature_cols = [
            'acousticness', 'danceability', 'energy',
            'instrumentalness', 'speechiness', 'valence'
        ]

    def prepare_audio_features(
        self,
        df: pd.DataFrame,
        fit_scaler: bool = True
    ) -> np.ndarray:
        """
        Prepares audio features: normalizes + one-hot encodes key

        Args:
            df: DataFrame with audio feature columns
            fit_scaler: If True, trains the scaler
        Returns:
            Array of shape (n_samples, 18)
        """
        # Extract 6 audio features
        audio_raw = df[self.audio_feature_cols].values

        # Normalize
        audio_normalized, self.scaler = normalize_audio_features(
            audio_raw,
            scaler=self.scaler,
            fit=fit_scaler
        )

        # One-hot encode key
        keys = df['key'].values
        key_encoded = one_hot_encode_key(keys)

        # Concatenate
        audio_features_full = np.concatenate(
            [audio_normalized, key_encoded],
            axis=1
        )

        return audio_features_full

    def generate_genre_embeddings(self, genres: pd.Series) -> np.ndarray:
        """
        Generates genre embeddings using sentence-transformers

        Args:
            genres: Series with genres (e.g., "Rock", "Pop")

        Returns:
            Array of shape (n_samples, 384)
        """
        logger.debug("Generating genre embeddings...")

        # Convert to list
        genre_list = genres.tolist()

        # Generate embeddings in batch
        genre_embeddings = self.sentence_model.encode(
            genre_list,
            show_progress_bar=True,
            batch_size=self.batch_size
        )

        return genre_embeddings

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Generates hybrid embeddings for the entire dataset

        Args:
            df: DataFrame with columns: audio features, key, general_genre

        Returns:
            Array of embeddings (n_samples, 402)
        """
        logger.info("=== GENERATING HYBRID EMBEDDINGS ===")

        # 1. Prepare audio features
        logger.info("Step 1/3: Processing audio features...")
        audio_features = self.prepare_audio_features(df, fit_scaler=True)
        logger.debug(f"Audio features shape: {audio_features.shape}")

        # 2. Generate genre embeddings
        logger.info("Step 2/3: Generating genre embeddings...")
        genre_embeddings = self.generate_genre_embeddings(df['general_genre'])
        logger.debug(f"Genre embeddings shape: {genre_embeddings.shape}")

        # 3. Combine
        logger.info("Step 3/3: Combining embeddings...")
        combined_embeddings = combine_embeddings(
            audio_features,
            genre_embeddings,
            w_audio=self.w_audio,
            w_genre=self.w_genre,
            normalize=True
        )
        logger.debug(f"Combined embeddings shape: {combined_embeddings.shape}")

        logger.info("✅ Embeddings generated successfully!")

        return combined_embeddings

    def save_embeddings(
        self,
        embeddings: np.ndarray,
        track_ids: pd.Series,
        output_dir: Path
    ):
        """
        Saves embeddings and metadata

        Args:
            embeddings: Array of embeddings
            track_ids: Series with the track_ids
            output_dir: Directory where to save the embeddings
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save embeddings
        embeddings_path = output_dir / "track_embeddings.npy"
        np.save(embeddings_path, embeddings)
        logger.info(f"✅ Embeddings saved in: {embeddings_path}")

        # Save track_ids (for mapping later)
        ids_path = output_dir / "track_ids.npy"
        np.save(ids_path, track_ids.values)
        logger.info(f"✅ Track IDs saved in: {ids_path}")

        # Save scaler
        scaler_path = output_dir / "scaler.pkl"
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        logger.info(f"✅ Scaler saved in: {scaler_path}")

        # Save metadata
        metadata = {
            'model_name': self.model_name,
            'w_audio': self.w_audio,
            'w_genre': self.w_genre,
            'n_samples': len(embeddings),
            'embedding_dim': embeddings.shape[1],
            'audio_features': self.audio_feature_cols
        }

        metadata_path = output_dir / "metadata.pkl"
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
        logger.info(f"✅ Metadata saved in: {metadata_path}")

def generate_all_embeddings(
    data_path: str,
    output_dir: str,
    w_audio: float = None,
    w_genre: float = None
) -> Dict:
    """
    Main function for generating embeddings of the whole dataset

    Args:
        data_path: Path to the dataset CSV
        output_dir: Directory where to save embeddings
        w_audio: Weight for audio features
        w_genre: Weight for genre embeddings

    Returns:
        Dict with process information
    """
    # Load defaults from config
    cfg = get_config()
    if w_audio is None:
        w_audio = cfg.get('embeddings.audio_weight', 10.0)
    if w_genre is None:
        w_genre = cfg.get('embeddings.genre_weight', 1.0)

    # Load dataset
    logger.info(f"Loading dataset from {data_path}...")
    df = pd.read_csv(data_path)
    logger.debug(f"Dataset loaded: {df.shape}")

    # Create generator
    generator = TrackEmbeddingGenerator(
        w_audio=w_audio,
        w_genre=w_genre
    )

    # Generate embeddings
    embeddings = generator.fit_transform(df)

    # Save
    generator.save_embeddings(
        embeddings,
        df['track_id'],
        Path(output_dir)
    )

    return {
        'n_tracks': len(df),
        'embedding_dim': embeddings.shape[1],
        'output_dir': output_dir
    }