"""
Utilities for processing embeddings
"""
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import List, Tuple


def one_hot_encode_key(keys: np.ndarray) -> np.ndarray:
    """
    One-hot encode 'key' column (0-11)

    Args:
        keys: Array of keys (values 0-11)

    Returns:
        Array of shape (n_samples, 12) with one-hot encoding
    """
    n_samples = len(keys)
    one_hot = np.zeros((n_samples, 12))

    for i, key in enumerate(keys):
        if not np.isnan(key):
            one_hot[i, int(key)] = 1

    return one_hot


def normalize_audio_features(
    features: np.ndarray,
    scaler: StandardScaler = None,
    fit: bool = True
) -> Tuple[np.ndarray, StandardScaler]:
    """
    Normalise audio features using StandardScaler

    Args:
        features: Array of features (n_samples, n_features)
        scaler: Pre-trained scaler (optional)
        fit: If True, fit the scaler. If False, only transform

    Returns:
        features_normalized, scaler
    """
    if scaler is None:
        scaler = StandardScaler()

    if fit:
        features_normalized = scaler.fit_transform(features)
    else:
        features_normalized = scaler.transform(features)

    return features_normalized, scaler


def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    """
    L2 Normalization (each vector will have norm = 1)

    Args:
        vectors: Array of shape (n_samples, n_dims)

    Returns:
        Normalized vectors
    """
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    # Avoid division by zero
    norms = np.where(norms == 0, 1, norms)
    return vectors / norms


def combine_embeddings(
    audio_features: np.ndarray,
    genre_embeddings: np.ndarray,
    w_audio: float = 10.0,
    w_genre: float = 1.0,
    normalize: bool = True
) -> np.ndarray:
    """
    Combine audio and genre embeddings with weights

    Args:
        audio_features: (n_samples, n_audio_dims)
        genre_embeddings: (n_samples, 384)
        w_audio: Weight for audio features
        w_genre: Weight for genre embeddings
        normalize: If True, applies L2 normalization at the end

    Returns:
        Combined embeddings (n_samples, n_audio_dims + 384)
    """
    # weight features
    audio_weighted = audio_features * w_audio
    genre_weighted = genre_embeddings * w_genre

    # Concatenate
    combined = np.concatenate([audio_weighted, genre_weighted], axis=1)

    # Normalize
    if normalize:
        combined = l2_normalize(combined)

    return combined