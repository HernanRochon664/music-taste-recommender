# 🏗️ System Architecture

## Overview

The Music Taste Recommender uses a **two-stage recommendation approach**:

1. **Candidate Retrieval**: Fast similarity search over 217K tracks
2. **Re-ranking**: Business-driven re-ordering based on strategy

---

## Data Processing Pipeline

Before the recommendation system can operate, raw data must be processed through a multi-stage pipeline:

### Pipeline Stages
```
┌───────────────────────────────────────────────────────────────┐
│                RAW DATA (Kaggle - 900K tracks)                │
└───────────────────────────────┬───────────────────────────────┘
                                │
                                ▼
┌───────────────────────────────────────────────────────────────┐
│                    STAGE 1: GENRE PARSING                     │
│                                                               │
│  • Parse genre strings to lists                               │
│  • Map specific genres to 9 general categories                │
│  • Filter unmapped genres                                     │
└───────────────────────────────┬───────────────────────────────┘
                                │
                                ▼
┌───────────────────────────────────────────────────────────────┐
│                  STAGE 2: QUALITY FILTERING                   │
│                                                               │
│  • Remove tracks with missing audio features                  │
│  • Validate data completeness                                 │
│  • Result: ~270K tracks with complete data                    │
└───────────────────────────────┬───────────────────────────────┘
                                │
                                ▼
┌───────────────────────────────────────────────────────────────┐
│                  STAGE 3: DATASET BALANCING                   │
│                                                               │
│  • Discard genres with < 10K tracks                           │
│  • Downsample genres with > 40K tracks                        │
│  • Result: 217K tracks, 9 genres (balanced)                   │
└───────────────────────────────┬───────────────────────────────┘
                                │
                                ▼
┌───────────────────────────────────────────────────────────────┐
│                PROCESSED DATASET (217K tracks)                │
│                                                               │
│  • 9 balanced genres                                          │
│  • Complete audio features                                    │
│  • Ready for embedding generation                             │
└───────────────────────────────────────────────────────────────┘
```

### Configuration-Driven Design

The data processing pipeline is configuration-driven for maintainability:

**Config Layer** (`config/genre_mapping.py`):
- Genre taxonomy (15 general categories)
- Balancing parameters (min/max tracks)
- Reusable across different processing runs

**Logic Layer** (`src/data_processing.py`):
- `DatasetProcessor` class with modular methods
- Each stage is independently testable
- Reusable for different datasets/parameters

**Orchestration Layer** (`entrypoint/process_dataset.py`):
- Coordinates pipeline execution
- Handles I/O and error reporting
- CLI interface for reproducibility

**Example Usage:**
```python
from src.data_processing import DatasetProcessor
from config.genre_mapping import GENRE_GROUPS

processor = DatasetProcessor(
    genre_mapping=GENRE_GROUPS,
    min_tracks_per_genre=10_000,
    max_tracks_per_genre=40_000
)

df = processor.process(
    input_path="data/raw/spotify_tracks.csv",
    output_path="data/processed/clean_balanced.csv"
)
```

### Genre Mapping Strategy

The system uses a **hierarchical genre taxonomy**:

**Level 1 (Specific):** Spotify's granular genres (5,888 unique)
- Examples: "nu metal", "pov: indie", "deep groove house"

**Level 2 (General):** Our 15 mapped categories
- Examples: "Metal", "Pop", "Electronic"

**Level 3 (Final):** 9 categories after balancing
- Retained: Genres with ≥10K tracks
- Discarded: R&B (9K), Folk (7K), Metal (7K), etc.

**Rationale:**
- Balances diversity with statistical robustness
- Each genre has sufficient samples for embedding quality
- Prevents model bias toward over-represented genres

---

## Component Diagram
```
┌───────────────────────────────────────────────────────────────┐
│                           USER INPUT                          │
│                    (Track history or seed)                    │
└───────────────────────────────┬───────────────────────────────┘
                                │
                                ▼
┌───────────────────────────────────────────────────────────────┐
│                      EMBEDDING GENERATOR                      │
│  ┌────────────────┐  ┌─────────────────┐  ┌────────────────┐  │
│  │ Audio Features │  │ Genre Embedding │  │ Normalization  │  │
│  │   (6 + 12 OH)  │→ │  (Transformer)  │→ │   & Combine    │  │
│  └────────────────┘  └─────────────────┘  └────────────────┘  │
│                               ↓                               │
│                        [402-dim vector]                       │
└───────────────────────────────┬───────────────────────────────┘
                                │
                                ▼
┌───────────────────────────────────────────────────────────────┐
│                     RECOMMENDATION ENGINE                     │
│                                                               │
│  Stage 1: RETRIEVAL                                           │
│  ┌─────────────────────────────────────────────┐              │
│  │ Cosine Similarity Search                    │              │
│  │ → Top-100 candidates                        │              │
│  └─────────────────────────────────────────────┘              │
│                         ↓                                     │
│  Stage 2: RE-RANKING                                          │
│  ┌─────────────────────────────────────────────┐              │
│  │ For each candidate:                         │              │
│  │   • Calculate diversity score               │              │
│  │   • Combine: w_rel*sim + w_div*diversity    │              │
│  │ → Sort by combined score                    │              │
│  │ → Return top-10                             │              │
│  └─────────────────────────────────────────────┘              │
└───────────────────────────────┬───────────────────────────────┘
                                │
                                ▼
┌───────────────────────────────────────────────────────────────┐
│                       BUSINESS METRICS                        │
│                                                               │
│  • Relevance Score (cosine similarity)                        │
│  • Diversity Score (new genres %)                             │
│  • Composite Score (strategy-weighted)                        │
└───────────────────────────────────────────────────────────────┘
```

---

## Data Flow

### 1. Embedding Generation (Offline)
```python
Input: 217K tracks with audio features + genres
    ↓
Process:
  - Normalize audio features (StandardScaler)
  - One-hot encode 'key' (12 classes)
  - Generate genre embeddings (sentence-transformers)
  - Weighted concatenation
  - L2 normalization
    ↓
Output: 402-dim embeddings stored in .npy
```

### 2. Recommendation (Online)
```python
Input: User history (list of track_ids)
    ↓
Step 1: Create user profile
  user_embedding = mean(embeddings[history_ids])
    ↓
Step 2: Retrieve candidates
  similarities = cosine_similarity(user_embedding, all_embeddings)
  top_100_candidates = argsort(similarities)[:100]
    ↓
Step 3: Re-rank
  for candidate in top_100:
      diversity = 1 if new_genre else 0
      score = w_rel * similarity + w_div * diversity
  top_10 = sort_by(score)[:10]
    ↓
Output: 10 recommendations with scores
```

---

## Key Design Decisions

### Why Hybrid Embeddings?

**Problem:** Audio features alone don't capture genre semantics.

**Solution:** Combine:
- **Audio features**: Capture musical characteristics
- **Genre embeddings**: Capture semantic relationships

**Result:** Jazz and Classical have high audio similarity but different genre embeddings.

### Why Re-ranking?

**Alternative:** Directly optimize embedding space for diversity.

**Why Re-ranking is better:**
1. **Flexibility**: Easy to change strategy without re-training
2. **Interpretability**: Clear separation of relevance vs. diversity
3. **Performance**: Fast cosine search + lightweight re-ranking

### Why Weighted Combination?

**Problem:** 384 genre dims >> 18 audio dims → genre dominates

**Solution:** Weight audio features 10x:
```python
audio_weighted = audio * 10.0
genre_weighted = genre * 1.0
combined = concat([audio_weighted, genre_weighted])
```

**Result:** Balanced influence of both components.

---

## Strategy Configuration

Strategies are defined in `config/business_config.py`:
```python
@dataclass
class RecommendationStrategy:
    w_relevance: float  # Weight for similarity
    w_diversity: float  # Weight for diversity
```

**Examples:**
- **Conservative** (0.7, 0.3): Prioritize satisfaction
- **Balanced** (0.5, 0.5): Equal weight
- **Aggressive** (0.3, 0.7): Maximize exploration

**Implementation:**
```python
score = strategy.w_rel * similarity + strategy.w_div * diversity
```

---

## Scalability Considerations

### Current Implementation
- **Dataset**: 217K tracks (in-memory)
- **Embeddings**: ~350 MB (NumPy array)
- **Latency**: ~50ms per recommendation (on laptop CPU)

### Production Scaling

For millions of tracks:

1. **Vector Database**:
   - Use Faiss, Milvus, or Pinecone
   - Enable ANN (approximate nearest neighbors)

2. **Distributed Embeddings**:
   - Shard embeddings across nodes
   - Parallel similarity search

3. **Caching**:
   - Cache user profiles
   - Pre-compute popular recommendations

4. **Batch Processing**:
   - Generate recommendations offline for popular patterns
   - Real-time only for unique queries

---

## Testing Strategy

### Unit Tests
- Business metrics calculation
- Embedding normalization
- Strategy weight validation

### Integration Tests
- End-to-end recommendation flow
- Embedding generation pipeline

### Evaluation
- 500 simulated users
- 5 business strategies
- Genre-specific analysis

---

## Future Enhancements

1. **Real User Feedback Loop**
   - Incorporate click-through rate
   - A/B testing framework

2. **Temporal Dynamics**
   - Decay old history
   - Trending tracks boost

3. **Context-Aware**
   - Time of day
   - User mood/activity

4. **Multi-Armed Bandits**
   - Adaptive strategy selection
   - Explore-exploit optimization