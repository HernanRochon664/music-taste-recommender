"""
Configuration of business strategies for recommendations
"""

from dataclasses import dataclass


@dataclass
class RecommendationStrategy:
    """Recommendation strategy with configurable weights"""
    name: str
    description: str
    w_relevance: float
    w_diversity: float
    w_novelty: float = 0.0

    def __post_init__(self):
        # Validate that weights sum to ~1.0
        total = self.w_relevance + self.w_diversity + self.w_novelty
        if abs(total - 1.0) > 0.01:
            raise ValueError(f"Weights must sum to 1.0, got {total}")


# Predefined strategies
STRATEGIES = {
    'balanced': RecommendationStrategy(
        name='Balanced',
        description='Balance between satisfaction and exploration',
        w_relevance=0.50,
        w_diversity=0.50
    ),

    'retention': RecommendationStrategy(
        name='Retention-Focused',
        description='Prioritizes user satisfaction (lower churn risk)',
        w_relevance=0.60,
        w_diversity=0.40
    ),

    'discovery': RecommendationStrategy(
        name='Discovery-Focused',
        description='Promotes catalog exploration (higher engagement)',
        w_relevance=0.40,
        w_diversity=0.60
    ),

    'conservative': RecommendationStrategy(
        name='Conservative',
        description='Comfort zone (new or sensitive users)',
        w_relevance=0.70,
        w_diversity=0.30
    ),

    'aggressive': RecommendationStrategy(
        name='Aggressive Discovery',
        description='Maximum exploration (power users)',
        w_relevance=0.30,
        w_diversity=0.70
    )
}


def get_strategy(strategy_name: str = 'balanced') -> RecommendationStrategy:
    """Obtains a recommendation strategy by name"""
    if strategy_name not in STRATEGIES:
        raise ValueError(f"Unknown strategy: {strategy_name}. Available: {list(STRATEGIES.keys())}")
    return STRATEGIES[strategy_name]