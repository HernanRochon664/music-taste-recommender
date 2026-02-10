"""
Track selection component
"""

import streamlit as st


def render_track_selector(df, n_tracks=100):
    """
    Render track selection widget

    Args:
        df: DataFrame with tracks
        n_tracks: Number of popular tracks to show

    Returns:
        Selected track_id or None
    """
    st.markdown("### 🎵 Step 1: Select a Seed Track")

    # Get popular tracks
    popular_tracks = df.nlargest(n_tracks, 'popularity')

    # Create options
    track_options = {}
    for _, row in popular_tracks.iterrows():
        display_name = f"{row['name'][:40]:40s} | {row['general_genre']:12s} | ⭐ {row['popularity']:.0f}"
        track_options[display_name] = row['track_id']

    # Selection methods
    selection_method = st.radio(
        "Selection method:",
        ["Search by name", "Browse popular tracks"],
        horizontal=True
    )

    if selection_method == "Search by name":
        # Search box
        search_query = st.text_input(
            "🔍 Search track name:",
            placeholder="e.g., Bohemian Rhapsody"
        )

        if search_query:
            # Filter tracks by search
            matches = df[df['name'].str.contains(search_query, case=False, na=False)]

            if len(matches) > 0:
                matches = matches.nlargest(20, 'popularity')

                search_options = {}
                for _, row in matches.iterrows():
                    display_name = f"{row['name'][:40]:40s} | {row['general_genre']:12s}"
                    search_options[display_name] = row['track_id']

                selected_display = st.selectbox(
                    f"Found {len(matches)} matches:",
                    options=list(search_options.keys())
                )

                return search_options[selected_display]
            else:
                st.warning("No tracks found. Try a different search term.")
                return None
        else:
            st.info("👆 Enter a track name to search")
            return None

    else:
        # Browse popular
        selected_display = st.selectbox(
            f"Choose from top {n_tracks} popular tracks:",
            options=list(track_options.keys())
        )

        return track_options[selected_display]