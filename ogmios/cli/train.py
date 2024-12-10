'''
EfficientSpeech: An On-Device Text to Speech Model
https://ieeexplore.ieee.org/abstract/document/10094639
Rowel Atienza
Apache 2.0 License

Usage:
    python3 train.py
'''

import datetime
import logging
from pathlib import Path
from typing import Literal

import torch
import yaml
from lightning import Trainer
from lightning.pytorch.loggers import TensorBoardLogger
from tap import Tap

from ogmios.datamodule import OgmiosDataModule
from ogmios.model import EfficientSpeech
from ogmios.utils import get_args


def print_args(args):
    opt_log = '--------------- Options ---------------\n'
    opt = vars(args)
    for k, v in opt.items():
        opt_log += f'{str(k)}: {str(v)}\n'
    opt_log += '---------------------------------------\n'
    print(opt_log)
    return opt_log


class TrainCommandParser(Tap):
    config: Path  # Path to processing config file (yaml)
    verbose: bool = False

    accelerator: Literal['gpu', 'cpu'] = 'gpu'
    devices: int = 1
    iter: int = 1
    threads: int = 24
    precision: Literal["bf16-mixed", "16-mixed", 16, 32, 64] = 16

    num_workers: int = 4
    max_epochs: int = 5000
    warmup_epochs: int = 50
    weight_decay: float = 1e-5  # Optimizer weight decay
    lr: float = 1e-3  # Learning rate for AdamW
    val_every_epoch: int = 5  # Run val every N epochs

    batch_size: int = 128  # Batch size
    depth: int = 2  # Encoder depth. Default for tiny, small & base.
    block_depth: int = 2 # Decoder block depth. Default for tiny & small. Base:  3
    n_blocks: int = 2 # Decoder blocks. Default for tiny. Small & base: 3.
    reduction: int = 4 # Embed dim reduction factor. Default for tiny. Small: 2. Base: 1.
    head : int = 1 # Number of transformer encoder head. Default for tiny & small. Base: 2.
    embed_dim: int = 128 # Embedding or feature dim. To be reduced by --reduction.
    kernel_size: int = 3 # Conv1d kernel size (Encoder). Default for tiny & small. Base is 5.
    decoder_kernel_size: int = 5 # Conv1d kernel size (Decoder). Default for tiny, small & base: 5.
    expansion: int = 1 # MixFFN expansion. Default for tiny & small. Base: 2.

    infer_device: Literal['cpu', 'gpu'] = 'cpu'
    hifigan_checkpoint: str = "LJ_V2/generator_v2"



if __name__ == "__main__":
    args = TrainCommandParser().parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    config = yaml.load(open(args.config, "r"), Loader=yaml.FullLoader)

    args.num_workers *= args.devices
    torch.set_float32_matmul_precision('high')

    datamodule = OgmiosDataModule(preprocess_config=config,
                                  batch_size=args.batch_size,
                                  num_workers=args.num_workers)

    model = EfficientSpeech(preprocess_config=config,
                            lr=args.lr,
                            weight_decay=args.weight_decay,
                            max_epochs=args.max_epochs,
                            depth=args.depth,
                            n_blocks=args.n_blocks,
                            block_depth=args.block_depth,
                            reduction=args.reduction,
                            head=args.head,
                            embed_dim=args.embed_dim,
                            kernel_size=args.kernel_size,
                            decoder_kernel_size=args.decoder_kernel_size,
                            expansion=args.expansion,
                            hifigan_checkpoint=args.hifigan_checkpoint,
                            infer_device=args.infer_device,
                            verbose=args.verbose)

    if args.verbose:
        print_args(args)

    tb_logger = TensorBoardLogger("tb_logs", name="ogmios")

    trainer = Trainer(accelerator=args.accelerator,
                      devices=args.devices,
                      precision=args.precision,
                      check_val_every_n_epoch=args.val_every_epoch,
                      max_epochs=args.max_epochs,
                      logger=tb_logger)

    start_time = datetime.datetime.now()
    trainer.fit(model, datamodule=datamodule)
    elapsed_time = datetime.datetime.now() - start_time
    print(f"Training time: {elapsed_time}")
