"""
Download embeddings from Hugging Face Hub on first run
"""

from pathlib import Path
from typing import Optional, Dict
from .logger import get_logger

from huggingface_hub import hf_hub_download

logger = get_logger(__name__)

def download_embeddings_from_hf(
    repo_id: str,
    repo_type: str,
    embeddings_dir: Path,
    force_download: bool = False,
    token: Optional[str] = None
) -> Dict[str, str]:
    """
    Download embeddings from Hugging Face Hub if not present locally.

    Args:
        repo_id: Hugging Face repo ID
        embeddings_dir: Local directory to store embeddings
        force_download: Force re-download even if files exist
        token: Hugging Face access token (optional)

    Returns:
        Dict with paths to downloaded files
    """

    embeddings_dir = Path(embeddings_dir)
    embeddings_dir.mkdir(parents=True, exist_ok=True)

    files = [
        "spotify_clean_balanced.csv",
        "track_embeddings.npy",
        "track_ids.npy",
        "scaler.pkl",
        "metadata.pkl"
    ]

    downloaded_paths: Dict[str, str] = {}

    logger.info("🔽 Downloading embeddings from Hugging Face...")

    for filename in files:
        if filename.endswith('.csv'):
            # CSV to processed/
            local_path = embeddings_dir.parent / "processed" / filename
            local_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            # Embeddings to embeddings/
            local_path = embeddings_dir / filename

        if local_path.exists() and not force_download:
            logger.info(f"  ✅ {filename} already exists locally")
            downloaded_paths[filename] = str(local_path)
            continue

        try:
            logger.info(f"  📥 Downloading {filename}...")
            downloaded_file = hf_hub_download(
                repo_id=repo_id,
                repo_type=repo_type,
                filename=filename,
                cache_dir=str(Path(embeddings_dir).parent / ".cache"),
                local_dir=str(local_path.parent),
                local_dir_use_symlinks=False,
                token=token
            )

            downloaded_paths[filename] = str(local_path)
            logger.info(f"  ✅ {filename} downloaded")

        except Exception as e:
            logger.error(f"  ❌ Error downloading {filename}: {e}")
            raise

    logger.info("✅ All embeddings downloaded successfully!")

    return downloaded_paths


def ensure_embeddings_available(
    repo_id: str,
    repo_type: str,
    embeddings_dir: str = "data/embeddings",
    token: Optional[str] = None
) -> bool:
    """
    Ensure embeddings are available (download if needed)

    Args:
        repo_id: Hugging Face repo ID
        embeddings_dir: Local embeddings directory
        token: Hugging Face access token (optional)

    Returns:
        True if embeddings are available
    """
    try:
        download_embeddings_from_hf(
            repo_id=repo_id,
            repo_type=repo_type,
            embeddings_dir=Path(embeddings_dir),
            token=token
        )
        return True

    except Exception as e:
        logger.error(f"Failed to download embeddings: {e}")
        return False