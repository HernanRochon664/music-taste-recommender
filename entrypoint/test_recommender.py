"""
Test script of recommendation system
"""
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from src.recommender import MusicRecommender
from src.user_simulator import UserSimulator
from src.business_metrics import format_metrics_report
import pandas as pd


if __name__ == "__main__":
    print("="*60)
    print("TEST OF RECOMMENDATION SYSTEM")
    print("="*60)

    # Paths
    root_dir = Path(__file__).parent.parent
    embeddings_path = root_dir / "data" / "embeddings" / "track_embeddings.npy"
    track_ids_path = root_dir / "data" / "embeddings" / "track_ids.npy"
    dataset_path = root_dir / "data" / "processed" / "spotify_clean_balanced.csv"

    # Load system
    print("\n1️⃣ Loading system...")
    recommender = MusicRecommender(
        embeddings_path,
        track_ids_path,
        dataset_path,
        strategy='balanced'
    )

    # Load dataset for simulation
    print("\n2️⃣ Generating test user...")
    df = pd.read_csv(dataset_path)
    simulator = UserSimulator(df, seed=42)
    test_users = simulator.generate_users(n_users=5, tracks_per_user=20)

    # Test with first user
    test_user = test_users[0]

    print(f"\n📊 Test user:")
    print(f"   ID: {test_user.user_id}")
    print(f"   Favorite genre: {test_user.favorite_genre}")
    print(f"   Tracks in history: {len(test_user.history)}")

    print("\n   History (first 5 tracks):")
    for idx, row in test_user.history.head().iterrows():
        print(f"      - {row['name'][:40]:40s} | {row['general_genre']}")

    # Generate recommendations
    print("\n3️⃣ Generating recommendations...")
    recommendations, metrics = recommender.recommend_for_user_history(
        test_user.history_track_ids,
        n_recommendations=10
    )

    # Show recommendations
    print("\n🎵 RECOMMENDATIONS:")
    print("="*80)
    for idx, row in recommendations.iterrows():
        print(f"{idx+1:2d}. {row['name'][:40]:40s} | {row['general_genre']:12s} | "
            f"Sim: {row['similarity']:.3f}")

    # Show metrics
    print(format_metrics_report(metrics))

    # Test different strategies
    print("\n4️⃣ Comparing strategies...")
    print("="*60)

    strategies = ['balanced', 'retention', 'discovery']

    for strategy_name in strategies:
        recommender.change_strategy(strategy_name)
        _, metrics = recommender.recommend_for_user_history(
            test_user.history_track_ids,
            n_recommendations=10
        )

        print(f"\n{strategy_name.upper():12s} | "
            f"Relevance: {metrics['relevance']:.3f} | "
            f"Diversity: {metrics['diversity']:.3f} | "
            f"Composite: {metrics['composite_score']:.3f}")

    print("\n" + "="*60)
    print("✅ TEST COMPLETED")
    print("="*60)