"""
Music Discovery Engine - Streamlit Demo

A business-driven recommendation system demonstrating
configurable trade-offs between relevance and diversity.
"""

import streamlit as st
import sys
from pathlib import Path

# Add parent to path
sys.path.append(str(Path(__file__).parent.parent))

from app_config import *
from app_utils import load_recommender, load_dataset, get_popular_tracks
from components.track_selector import render_track_selector
from components.metrics_display import render_metrics
from components.recommendations_table import render_recommendations

# Check and download embeddings if needed
from utils.download_embeddings import ensure_embeddings_available
from config import config

# Page config (must be first Streamlit command)
st.set_page_config(
    page_title="Music Discovery Engine",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Embeddings check
HF_REPO_ID = config.get('huggingface.repo_id')
HF_TYPE = config.get('huggingface.repo_type')
EMBEDDINGS_DIR = Path(config.get('paths.embeddings_dir'))
embeddings_path = EMBEDDINGS_DIR / "track_embeddings.npy"

# Download embeddings if not present
if not embeddings_path.exists():
    st.info("🔽 First run detected. Downloading embeddings from Hugging Face...")
    st.caption(f"Repository: {HF_REPO_ID}")

    # Get token if available
    token = None
    try:
        if 'HF_TOKEN' in st.secrets:
            token = st.secrets['HF_TOKEN']
    except Exception:
        pass

    with st.spinner("⏳ Downloading embeddings. This will take 1-2 minutes..."):
        success = ensure_embeddings_available(
            repo_id=HF_REPO_ID,
            repo_type=HF_TYPE,
            embeddings_dir=str(EMBEDDINGS_DIR),
            token=token
        )

    if success:
        st.success("✅ Embeddings downloaded successfully! Reloading app...")
        st.rerun()
    else:
        st.error(f"❌ Failed to download embeddings from {HF_REPO_ID}")
        st.error("Please check:")
        st.code("""
        1. Repository exists and is public (or you have access)
        2. Files are named correctly:
            - track_embeddings.npy
            - track_ids.npy
            - scaler.pkl
            - metadata.pkl
        """)
        st.stop()

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)


def main():
    # Header
    st.markdown('<p class="main-header">🎵 Music Taste Recommender</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Business-Driven Recommendation System</p>', unsafe_allow_html=True)

    # Sidebar - Strategy selection
    with st.sidebar:
        st.markdown("## ⚙️ Configuration")

        strategy_key = st.radio(
            "Select Recommendation Strategy:",
            options=list(STRATEGY_INFO.keys()),
            format_func=lambda x: f"{STRATEGY_INFO[x]['emoji']} {STRATEGY_INFO[x]['name']}",
            index=1  # Default to 'balanced'
        )

        strategy_info = STRATEGY_INFO[strategy_key]

        st.markdown(f"**{strategy_info['description']}**")
        st.caption(strategy_info['weights'])

        st.markdown("---")

        # About
        with st.expander("ℹ️ About this demo"):
            st.markdown("""
            This demo showcases a hybrid recommendation system that:

            - Combines audio features + genre embeddings
            - Implements re-ranking for business objectives
            - Provides configurable strategies

            **Tech Stack:**
            - sentence-transformers (embeddings)
            - scikit-learn (similarity)
            - Streamlit (interface)
            """)

        st.markdown("---")
        st.caption("💡 Tip: Try different strategies to see how recommendations change!")

    # Load data and model
    with st.spinner("Loading recommendation system..."):
        df = load_dataset(DATASET_PATH)
        recommender = load_recommender(
            EMBEDDINGS_PATH,
            TRACK_IDS_PATH,
            DATASET_PATH,
            strategy=strategy_key
        )

    # Track selection
    selected_track_id = render_track_selector(df, N_SEED_TRACKS_TO_SHOW)

    if selected_track_id is None:
        st.info("👆 Please select a track to get recommendations")
        return

    # Strategy selection (already in sidebar)
    st.markdown("### ⚙️ Step 2: Review Strategy")
    st.success(f"{strategy_info['emoji']} Using **{strategy_info['name']}** strategy")

    # Generate button
    if st.button("🚀 Generate Recommendations", type="primary", use_container_width=True):

        with st.spinner("Generating recommendations..."):
            # Get seed track info
            seed_track = df[df['track_id'] == selected_track_id].iloc[0]

            # Generate recommendations
            recommendations = recommender.find_similar_tracks(
                selected_track_id,
                n_recommendations=N_RECOMMENDATIONS,
                n_candidates=100
            )

            # Add evaluation metrics
            # Calculate diversity (% of tracks that are NOT the seed genre)
            seed_genre = seed_track['general_genre']
            rec_genres = recommendations['general_genre'].tolist()

            different_genre_count = sum(1 for g in rec_genres if g != seed_genre)
            diversity = different_genre_count / len(rec_genres)

            # Calculate relevance (from similarity)
            relevance = recommendations['similarity'].mean()

            # Calculate composite
            from config.business_config import get_strategy
            strategy_obj = get_strategy(strategy_key)
            composite = (
                strategy_obj.w_relevance * relevance +
                strategy_obj.w_diversity * diversity
            )

            metrics_display = {
                'relevance': relevance,
                'diversity': diversity,
                'composite_score': composite
            }

        # Display results
        st.markdown("---")

        # Metrics
        render_metrics(metrics_display, strategy_info)

        st.markdown("---")

        # Recommendations
        render_recommendations(recommendations, seed_track.to_dict())

        # Download option
        st.markdown("---")
        csv = recommendations.to_csv(index=False)
        st.download_button(
            label="📥 Download Recommendations (CSV)",
            data=csv,
            file_name=f"recommendations_{seed_track['name'][:20]}.csv",
            mime="text/csv"
        )


if __name__ == "__main__":
    main()