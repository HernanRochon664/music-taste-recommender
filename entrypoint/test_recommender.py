"""
Test script of recommendation system
"""
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from config.config_loader import get_config
from src.recommender import MusicRecommender
from src.user_simulator import UserSimulator
from src.business_metrics import format_metrics_report
from utils.logger import get_logger
import pandas as pd

logger = get_logger(__name__)


if __name__ == "__main__":
    logger.info("="*60)
    logger.info("TEST OF RECOMMENDATION SYSTEM")
    logger.info("="*60)

    # Paths (from config)
    cfg = get_config()
    root_dir = Path(__file__).parent.parent
    embeddings_dir = Path(cfg.get('paths.embeddings_dir', root_dir / "data" / "embeddings"))
    embeddings_path = embeddings_dir / "track_embeddings.npy"
    track_ids_path = embeddings_dir / "track_ids.npy"
    dataset_path = Path(cfg.get('paths.processed_data', root_dir / "data" / "processed" / "spotify_clean_balanced.csv"))

    # Load system
    logger.info("\n1️⃣ Loading system...")
    recommender = MusicRecommender(
        embeddings_path,
        track_ids_path,
        dataset_path,
        strategy='balanced'
    )

    # Load dataset for simulation
    logger.info("\n2️⃣ Generating test user...")
    df = pd.read_csv(dataset_path)
    simulator = UserSimulator(df, seed=42)
    test_users = simulator.generate_users(n_users=5, tracks_per_user=20)

    # Test with first user
    test_user = test_users[0]

    logger.info(f"\n📋 Test user:")
    logger.info(f"   ID: {test_user.user_id}")
    logger.info(f"   Favorite genre: {test_user.favorite_genre}")
    logger.info(f"   Tracks in history: {len(test_user.history)}")

    logger.info("   History (first 5 tracks):")
    for idx, row in test_user.history.head().iterrows():
        logger.info(f"      - {row['name'][:40]:40s} | {row['general_genre']}")

    # Generate recommendations
    logger.info("\n3️⃣ Generating recommendations...")
    recommendations, metrics = recommender.recommend_for_user_history(
        test_user.history_track_ids,
        n_recommendations=10
    )

    # Show recommendations
    logger.info("\n🎵 RECOMMENDATIONS:")
    logger.info("="*80)
    for idx, row in recommendations.iterrows():
        logger.info(f"{idx+1:2d}. {row['name'][:40]:40s} | {row['general_genre']:12s} | "
            f"Sim: {row['similarity']:.3f}")

    # Show metrics
    logger.info(format_metrics_report(metrics))

    # Test different strategies
    logger_info_msg = "\n4️⃣ Comparing strategies..."
    logger.info(logger_info_msg)
    logger.info("="*60)

    strategies = ['balanced', 'retention', 'discovery']

    for strategy_name in strategies:
        recommender.change_strategy(strategy_name)
        _, metrics = recommender.recommend_for_user_history(
            test_user.history_track_ids,
            n_recommendations=10
        )

        logger.info(f"\n{strategy_name.upper():12s} | "
            f"Relevance: {metrics['relevance']:.3f} | "
            f"Diversity: {metrics['diversity']:.3f} | "
            f"Composite: {metrics['composite_score']:.3f}")

    logger.info("\n" + "="*60)
    logger.info("✅ TEST COMPLETED")
    logger.info("="*60)