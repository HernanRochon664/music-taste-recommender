"""
Generation of hybrid embeddings Pipeline for tracks from Spotify
"""
import pandas as pd
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
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


class TrackEmbeddingGenerator:
    """
    Hybrid embeddings generator for tracks

    Combines:
    - Numeric audio features (6 features + 12 one-hot key)
    - Embeddings per genre (sentence-transformers)
    """

    def __init__(
        self,
        model_name: str = 'all-MiniLM-L6-v2',
        w_audio: float = 10.0,
        w_genre: float = 1.0
    ):
        """
        Args:
            model_name: Model of sentence-transformers
            w_audio: Weight for audio features
            w_genre: Weight for genre embeddings
        """
        self.model_name = model_name
        self.w_audio = w_audio
        self.w_genre = w_genre

        # load sentence-transformers model
        print(f"Loading model {model_name}...")
        self.sentence_model = SentenceTransformer(model_name)

        # Scaler for audio features
        self.scaler = None

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
        print("Generating genre embeddings...")

        # Convert to list
        genre_list = genres.tolist()

        # Generate embeddings in batch
        genre_embeddings = self.sentence_model.encode(
            genre_list,
            show_progress_bar=True,
            batch_size=256
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
        print("\n=== GENERATING HYBRID EMBEDDINGS ===\n")

        # 1. Prepare audio features
        print("1. Processing audio features...")
        audio_features = self.prepare_audio_features(df, fit_scaler=True)
        print(f"   Audio features shape: {audio_features.shape}")

        # 2. Generate genre embeddings
        print("\n2. Generating genre embeddings...")
        genre_embeddings = self.generate_genre_embeddings(df['general_genre'])
        print(f"   Genre embeddings shape: {genre_embeddings.shape}")

        # 3. Combine
        print("\n3. Combining embeddings...")
        combined_embeddings = combine_embeddings(
            audio_features,
            genre_embeddings,
            w_audio=self.w_audio,
            w_genre=self.w_genre,
            normalize=True
        )
        print(f"   Combined embeddings shape: {combined_embeddings.shape}")

        print("\n✅ Embeddings generated successfully!")

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
        print(f"✅ Embeddings saved in: {embeddings_path}")

        # Save track_ids (for mapping later)
        ids_path = output_dir / "track_ids.npy"
        np.save(ids_path, track_ids.values)
        print(f"✅ Track IDs saved in: {ids_path}")

        # Save scaler
        scaler_path = output_dir / "scaler.pkl"
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        print(f"✅ Scaler saved in: {scaler_path}")

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
        print(f"✅ Metadata saved in: {metadata_path}")

def generate_all_embeddings(
    data_path: str,
    output_dir: str,
    w_audio: float = 10.0,
    w_genre: float = 1.0
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
    # Load dataset
    print(f"Loading dataset from {data_path}...")
    df = pd.read_csv(data_path)
    print(f"Dataset loaded: {df.shape}")

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