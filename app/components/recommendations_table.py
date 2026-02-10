"""
Recommendations display component
"""

import streamlit as st
import plotly.express as px
from app.app_config import GENRE_COLORS


def render_recommendations(recommendations_df, seed_track_info):
    """
    Render recommendations table and visualizations

    Args:
        recommendations_df: DataFrame with recommendations
        seed_track_info: Dict with seed track metadata
    """
    st.markdown("### 🎵 Top 10 Recommendations")

    # Seed track info
    with st.expander("📀 Seed Track Info", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Track:** {seed_track_info['name']}")
            st.write(f"**Genre:** {seed_track_info['general_genre']}")
        with col2:
            st.write(f"**Popularity:** {seed_track_info['popularity']:.0f}")
            st.write(f"**Artist Popularity:** {seed_track_info.get('artist_popularity', 'N/A')}")

    # Recommendations table
    display_df = recommendations_df[[
        'name', 'general_genre', 'similarity', 'popularity'
    ]].copy()

    display_df.columns = ['Track', 'Genre', 'Similarity', 'Popularity']
    display_df.index = range(1, len(display_df) + 1)
    display_df['Similarity'] = display_df['Similarity'].apply(lambda x: f"{x:.3f}")
    display_df['Popularity'] = display_df['Popularity'].apply(lambda x: f"{x:.0f}")

    st.dataframe(
        display_df,
        use_container_width=True,
        height=400
    )

    # Genre distribution
    st.markdown("#### 📊 Genre Distribution in Recommendations")

    genre_counts = recommendations_df['general_genre'].value_counts()

    col1, col2 = st.columns([2, 1])

    with col1:
        # Pie chart
        fig = px.pie(
            values=genre_counts.values,
            names=genre_counts.index,
            color=genre_counts.index,
            color_discrete_map=GENRE_COLORS,
            hole=0.4
        )
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Stats
        st.markdown("**Statistics:**")
        st.write(f"Unique genres: {len(genre_counts)}")
        st.write(f"Seed genre: {seed_track_info['general_genre']}")

        same_genre_count = genre_counts.get(seed_track_info['general_genre'], 0)
        st.write(f"Same as seed: {same_genre_count}/10")

        new_genres = 10 - same_genre_count
        st.write(f"New genres: {new_genres}/10")