"""
Configuration loader for YAML settings

Provides centralized access to all configuration parameters.
"""

import yaml
from pathlib import Path
from typing import Dict, Any
import os


class Config:
    """
    Configuration manager for the project

    Loads settings from YAML and provides easy access.
    """

    _instance = None
    _config = None

    def __new__(cls):
        """Singleton pattern - only one config instance"""
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        """Load configuration on first instantiation"""
        if self._config is None:
            self.load_config()

    def load_config(self, config_path: str = None):
        """
        Load configuration from YAML file

        Args:
            config_path: Path to YAML config file.
                        If None, uses default location.
        """
        if config_path is None:
            # Default path relative to this file
            config_dir = Path(__file__).parent
            config_path = config_dir / "settings.yaml"

        with open(config_path, 'r') as f:
            self._config = yaml.safe_load(f)

    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation

        Args:
            key_path: Path to config value (e.g., 'paths.raw_data')
            default: Default value if key not found

        Returns:
            Configuration value or default

        Examples:
            >>> config = Config()
            >>> config.get('paths.raw_data')
            'data/raw/...'
            >>> config.get('embeddings.model_name')
            'all-MiniLM-L6-v2'
        """
        keys = key_path.split('.')
        value = self._config

        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default

    def get_all(self) -> Dict[str, Any]:
        """Get entire configuration dictionary"""
        return self._config.copy()

    # Convenience methods for common configs

    @property
    def paths(self) -> Dict[str, str]:
        """Get all paths configuration"""
        return self.get('paths', {})

    @property
    def embeddings_config(self) -> Dict[str, Any]:
        """Get embeddings configuration"""
        return self.get('embeddings', {})

    @property
    def recommender_config(self) -> Dict[str, Any]:
        """Get recommender configuration"""
        return self.get('recommender', {})

    @property
    def data_processing_config(self) -> Dict[str, Any]:
        """Get data processing configuration"""
        return self.get('data_processing', {})

    @property
    def logging_config(self) -> Dict[str, Any]:
        """Get logging configuration"""
        return self.get('logging', {})

    @property
    def hf_config(self) -> Dict[str, Any]:
        """Get Hugging Face configuration"""
        return self.get('huggingface', {})

# Global config instance
config = Config()


# Convenience function
def get_config() -> Config:
    """Get global configuration instance"""
    return config