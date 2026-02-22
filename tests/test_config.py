"""
Tests for configuration loading
"""

import pytest
from config import config, get_config


def test_config_singleton():
    """Test that config is a singleton"""
    config1 = get_config()
    config2 = get_config()
    assert config1 is config2


def test_config_has_required_keys():
    """Test that config has all required top-level keys"""
    required_keys = ['project', 'paths', 'embeddings', 'recommender', 'logging']

    for key in required_keys:
        assert config.get(key) is not None, f"Missing required config key: {key}"


def test_config_paths():
    """Test that all paths are defined"""
    paths = config.paths

    assert 'raw_data' in paths
    assert 'processed_data' in paths
    assert 'embeddings_dir' in paths

    # Check they're strings
    assert isinstance(paths['raw_data'], str)
    assert isinstance(paths['processed_data'], str)
    assert isinstance(paths['embeddings_dir'], str)


def test_config_embeddings():
    """Test embeddings configuration"""
    emb_config = config.embeddings_config

    assert emb_config['model_name'] == 'all-MiniLM-L6-v2'
    assert emb_config['embedding_dim'] == 402
    assert emb_config['audio_weight'] == 10.0
    assert emb_config['genre_weight'] == 1.0


def test_config_recommender():
    """Test recommender configuration"""
    rec_config = config.recommender_config

    assert rec_config['n_candidates'] == 100
    assert rec_config['n_recommendations'] == 10
    assert rec_config['default_strategy'] == 'balanced'


def test_config_get_with_default():
    """Test getting non-existent key with default"""
    value = config.get('non.existent.key', 'default_value')
    assert value == 'default_value'


def test_config_dot_notation():
    """Test dot notation access"""
    project_name = config.get('project.name')
    assert project_name == 'music-taste-recommender'

    model_name = config.get('embeddings.model_name')
    assert model_name == 'all-MiniLM-L6-v2'