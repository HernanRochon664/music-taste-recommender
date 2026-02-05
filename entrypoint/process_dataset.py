"""
Dataset processing entrypoint

Usage:
    python entrypoint/process_dataset.py

Input:  data/raw/music_data.csv
Output: data/processed/spotify_clean_balanced.csv
"""

import sys
from pathlib import Path

# Add app to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data_processing import DatasetProcessor
from config.genre_mapping import (
    GENRE_GROUPS,
    MIN_TRACKS_PER_GENRE,
    MAX_TRACKS_PER_GENRE
)


# Configuration
RAW_DATA_PATH = "data/raw/music_data.csv"
OUTPUT_PATH = "data/processed/spotify_clean_balanced.csv"


def main():
    """Run dataset processing pipeline"""

    # Check input file exists
    raw_path = Path(RAW_DATA_PATH)
    if not raw_path.exists():
        print(f"❌ ERROR: Input file not found: {RAW_DATA_PATH}")
        print(f"\n📥 Please download the dataset from:")
        print(f"   https://www.kaggle.com/datasets/olegfostenko/almost-a-million-spotify-tracks")
        print(f"\n   And place it in: {RAW_DATA_PATH}")
        sys.exit(1)

    # Initialize processor
    processor = DatasetProcessor(
        genre_mapping=GENRE_GROUPS,
        min_tracks_per_genre=MIN_TRACKS_PER_GENRE,
        max_tracks_per_genre=MAX_TRACKS_PER_GENRE,
        random_state=64
    )

    # Process
    try:
        df_final = processor.process(
            input_path=RAW_DATA_PATH,
            output_path=OUTPUT_PATH,
            show_stats=True
        )

        print(f"\n✅ Success! Processed {len(df_final):,} tracks")

    except Exception as e:
        print(f"\n❌ ERROR during processing:")
        print(f"   {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()