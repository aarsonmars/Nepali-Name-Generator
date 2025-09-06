"""
Configuration classes for the makemore project.
"""

from dataclasses import dataclass
from typing import List, Optional
import torch

@dataclass
class ModelConfig:
    """Configuration for neural network models."""
    block_size: int = None  # length of the input sequences of integers
    vocab_size: int = None  # the input integers are in range [0 .. vocab_size -1]
    # parameters below control the sizes of each model slightly differently
    n_layer: int = 4
    n_embd: int = 64
    n_embd2: int = 64
    n_head: int = 4

@dataclass
class DataConfig:
    """Configuration for data loading and preprocessing."""
    input_file: str = 'data/names.txt'  # default input file
    input_files: Optional[List[str]] = None  # multiple input files
    test_split_ratio: float = 0.1  # fraction of data for testing (0.1 = 10%)
    max_test_size: int = 1000  # maximum test set size
    encoding: str = 'utf-8'  # file encoding

@dataclass
class TrainingConfig:
    """Configuration for training parameters."""
    max_steps: int = 5000  # max number of optimization steps (-1 for infinite)
    batch_size: int = 32  # batch size during optimization
    learning_rate: float = 5e-4  # learning rate
    weight_decay: float = 0.01  # weight decay
    betas: tuple = (0.9, 0.99)  # Adam optimizer betas
    eps: float = 1e-8  # Adam optimizer epsilon
    gradient_clip: Optional[float] = None  # gradient clipping value
    eval_interval: int = 500  # evaluate every N steps
    sample_interval: int = 200  # sample every N steps
    save_interval: int = 500  # save model every N steps

@dataclass
class SamplingConfig:
    """Configuration for text generation/sampling."""
    temperature: float = 1.0  # sampling temperature
    top_k: int = -1  # top-k sampling (-1 means no top-k)
    max_new_tokens: int = 100  # maximum tokens to generate
    do_sample: bool = True  # whether to sample or use greedy decoding
    num_samples: int = 10  # number of samples to generate

@dataclass
class SystemConfig:
    """Configuration for system-level parameters."""
    device: str = 'cpu'  # device to use for compute
    seed: int = 3407  # random seed
    num_workers: int = 4  # number of data workers
    work_dir: str = 'models/out'  # output working directory
    resume: bool = False  # resume training from checkpoint
    sample_only: bool = False  # just sample, don't train
    model_type: str = 'transformer'  # model type: bigram|mlp|rnn|gru|bow|transformer

@dataclass
class Config:
    """Main configuration class that combines all configs."""
    model: ModelConfig = None
    data: DataConfig = None
    training: TrainingConfig = None
    sampling: SamplingConfig = None
    system: SystemConfig = None

    def __post_init__(self):
        if self.model is None:
            self.model = ModelConfig()
        if self.data is None:
            self.data = DataConfig()
        if self.training is None:
            self.training = TrainingConfig()
        if self.sampling is None:
            self.sampling = SamplingConfig()
        if self.system is None:
            self.system = SystemConfig()

# Convenience function to create default config
def get_default_config():
    """Get a default configuration."""
    return Config(
        model=ModelConfig(),
        data=DataConfig(),
        training=TrainingConfig(),
        sampling=SamplingConfig(),
        system=SystemConfig()
    )

# Configuration presets for different use cases
def get_quick_train_config():
    """Configuration for quick training/testing."""
    return Config(
        training=TrainingConfig(
            max_steps=1000,
            batch_size=16,
            eval_interval=100,
            sample_interval=50
        ),
        system=SystemConfig(
            work_dir='models/quick_train'
        )
    )

def get_production_config():
    """Configuration for production training."""
    return Config(
        training=TrainingConfig(
            max_steps=10000,
            batch_size=64,
            learning_rate=3e-4,
            eval_interval=500,
            save_interval=1000
        ),
        system=SystemConfig(
            device='cuda' if torch.cuda.is_available() else 'cpu',
            work_dir='models/production'
        )
    )
