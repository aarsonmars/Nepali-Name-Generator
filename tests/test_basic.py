"""
Basic tests for the Nepali Name Generator.
"""

import pytest
from src.config import ModelConfig
from src.data import create_datasets
from src.models import Bigram


def test_model_config():
    """Test ModelConfig initialization."""
    config = ModelConfig(vocab_size=100, block_size=10)
    assert config.vocab_size == 100
    assert config.block_size == 10
    assert config.n_layer == 4  # default value


def test_bigram_model():
    """Test Bigram model initialization."""
    config = ModelConfig(vocab_size=55, block_size=20)
    model = Bigram(config)
    assert model.get_block_size() == 1


def test_data_loading():
    """Test data loading from files."""
    # This would require actual data files to test properly
    # For now, just test that the function exists
    from src.data import create_datasets
    assert callable(create_datasets)


if __name__ == "__main__":
    pytest.main([__file__])
