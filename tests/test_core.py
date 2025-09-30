"""
Test suite for DLLM sampling system core functionality.
"""

import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch

from dllm_sampling.core import SamplingConfig, DLLMSampler


class TestSamplingConfig:
    """Test SamplingConfig dataclass."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = SamplingConfig()
        
        assert config.temperature == 1.0
        assert config.top_k == 50
        assert config.top_p == 0.9
        assert config.repetition_penalty == 1.0
        assert config.batch_size == 1
        assert config.max_length == 512
        assert config.use_cache is True
        assert config.num_workers == 1
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = SamplingConfig(
            temperature=0.8,
            top_k=40,
            top_p=0.95,
            batch_size=4
        )
        
        assert config.temperature == 0.8
        assert config.top_k == 40
        assert config.top_p == 0.95
        assert config.batch_size == 4


class TestDLLMSampler:
    """Test DLLMSampler class."""
    
    @pytest.fixture
    def config(self):
        return SamplingConfig(temperature=1.0, top_k=50, top_p=0.9)
    
    @pytest.fixture
    def sampler(self, config):
        return DLLMSampler(config)
    
    @pytest.fixture
    def logits(self):
        """Create test logits tensor."""
        torch.manual_seed(42)
        return torch.randn(2, 1000)  # batch_size=2, vocab_size=1000
    
    def test_sampler_initialization(self, sampler):
        """Test sampler initialization."""
        assert sampler.config.temperature == 1.0
        assert sampler.device.type in ['cuda', 'cpu']
        assert sampler.metrics['total_tokens'] == 0