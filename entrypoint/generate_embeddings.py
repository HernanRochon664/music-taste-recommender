"""
Script for generating embeddings of the entire dataset
"""
from pathlib import Path
import sys

# Add src to the path
sys.path.append(str(Path(__file__).parent.parent))

from config.config_loader import get_config
from src.embeddings import generate_all_embeddings


if __name__ == "__main__":
    # Load config
    cfg = get_config()

    # Paths
    root_dir = Path(__file__).parent.parent
    data_path = Path(cfg.get('paths.processed_data', root_dir / "data" / "processed" / "spotify_clean_balanced.csv"))
    output_dir = Path(cfg.get('paths.embeddings_dir', root_dir / "data" / "embeddings"))

    # Parameters (weights)
    W_AUDIO = cfg.get('embeddings.audio_weight', 10.0)  # Greater emphasis on audio features
    W_GENRE = cfg.get('embeddings.genre_weight', 1.0)   # Normal weight for genre
    print("="*60)
    print("GENERATION OF HYBRID EMBEDDINGS")
    print("="*60)
    print(f"\nDataset: {data_path}")
    print(f"Output: {output_dir}")
    print(f"\nWeights:")
    print(f"  Audio features: {W_AUDIO}")
    print(f"  Genre: {W_GENRE}")
    print("\n" + "="*60 + "\n")

    # Verify that the file exists
    if not data_path.exists():
        print(f"❌ ERROR: File not found: {data_path}")
        print(f"   Check the path or adjust it in the script")
        sys.exit(1)

    # Generate embeddings
    try:
        result = generate_all_embeddings(
            data_path=str(data_path),
            output_dir=str(output_dir),
            w_audio=W_AUDIO,
            w_genre=W_GENRE
        )

        print("\n" + "="*60)
        print("✅ PROCESS COMPLETED SUCCESSFULLY")
        print("="*60)
        print(f"Tracks processed: {result['n_tracks']:,}")
        print(f"Embedding dimension: {result['embedding_dim']}")
        print(f"Saved to: {result['output_dir']}")
    except Exception as e:
        print(f"\n❌ ERROR during the generation:")
        print(f"   {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)