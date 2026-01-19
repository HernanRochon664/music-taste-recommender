"""
Script for generating embeddings of the entire dataset
"""
from pathlib import Path
import sys

# Add src to the path
sys.path.append(str(Path(__file__).parent.parent))

from src.embeddings import generate_all_embeddings


if __name__ == "__main__":
    # Paths
    root_dir = Path(__file__).parent.parent
    data_path = root_dir / "data" / "processed" / "spotify_clean_balanced.csv"
    output_dir = root_dir / "data" / "embeddings"

    # Parameters
    W_AUDIO = 10.0  # Greater emphasis on audio features (because there are only 18 dims)
    W_GENRE = 1.0   # Normal weight for genre (384 dims)
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