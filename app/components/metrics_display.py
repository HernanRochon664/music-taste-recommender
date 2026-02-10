"""
Metrics visualization component
"""

import streamlit as st
import plotly.graph_objects as go


def render_metrics(metrics, strategy_info):
    """
    Render business metrics with visualizations

    Args:
        metrics: Dict with relevance, diversity, composite_score
        strategy_info: Dict with strategy metadata
    """
    st.markdown("### 📊 Business Metrics")

    # Display strategy info
    st.info(
        f"{strategy_info['emoji']} **{strategy_info['name']} Strategy** | "
        f"{strategy_info['weights']}"
    )

    # Metric cards
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            label="🎯 Relevance",
            value=f"{metrics['relevance']:.3f}",
            help="Similarity to seed track (higher = more similar)"
        )

    with col2:
        st.metric(
            label="🌈 Diversity",
            value=f"{metrics['diversity']:.3f}",
            help="Genre diversity (higher = more exploration)"
        )

    with col3:
        st.metric(
            label="⭐ Composite",
            value=f"{metrics['composite_score']:.3f}",
            help="Weighted combination based on strategy"
        )

    # Visualization
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=['Relevance', 'Diversity', 'Composite'],
        y=[metrics['relevance'], metrics['diversity'], metrics['composite_score']],
        marker_color=['#3498db', '#e74c3c', '#2ecc71'],
        text=[f"{metrics['relevance']:.3f}",
            f"{metrics['diversity']:.3f}",
            f"{metrics['composite_score']:.3f}"],
        textposition='auto',
    ))

    fig.update_layout(
        title="Metric Scores",
        yaxis_title="Score",
        yaxis_range=[0, 1],
        height=300,
        showlegend=False
    )

    st.plotly_chart(fig, use_container_width=True)

    # Interpretation
    with st.expander("📖 How to interpret these metrics"):
        st.markdown("""
        **Relevance Score (0-1)**
        - Measures how similar recommendations are to the seed track
        - Higher = recommendations match user's taste more closely
        - Business impact: Higher satisfaction, lower churn

        **Diversity Score (0-1)**
        - Percentage of recommendations from new genres
        - Higher = more catalog exploration
        - Business impact: Increased engagement, artist discovery

        **Composite Score (0-1)**
        - Weighted combination based on strategy
        - Balances satisfaction vs. exploration
        - Business impact: Optimized for business objective
        """)