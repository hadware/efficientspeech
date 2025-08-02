from dataclasses import dataclass

from ogmios.dataset.commons import AcousticStats, DatasetFolder


@dataclass
class OgmiosConfig:
    pass


@dataclass
class ModelConfig:
    """Parameters for the model architecture"""
    depth: int = 2  # Encoder depth. Default for tiny, small & base.
    block_depth: int = 2  # Decoder block depth. Default for tiny & small. Base:  3
    n_blocks: int = 2  # Decoder blocks. Default for tiny. Small & base: 3.
    reduction: int = 4  # Embed dim reduction factor. Default for tiny. Small: 2. Base: 1.
    head: int = 1  # Number of transformer encoder head. Default for tiny & small. Base: 2.
    embed_dim: int = 128  # Embedding or feature dim. To be reduced by --reduction.
    kernel_size: int = 3  # Conv1d kernel size (Encoder). Default for tiny & small. Base is 5.
    decoder_kernel_size: int = 5  # Conv1d kernel size (Decoder). Default for tiny, small & base: 5.
    expansion: int = 1  # MixFFN expansion. Default for tiny & small. Base: 2.

@dataclass
class DatasetParams:
    """Parameters from the dataset that are also used for the model architecture"""
    pitch_stats: AcousticStats
    energy_stats: AcousticStats
    dim_alphabet: int
    n_mels: int
