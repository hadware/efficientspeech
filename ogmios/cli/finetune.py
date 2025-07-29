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
from ogmios.dataset.commons import PreprocessingConfig, DatasetFolder
from ogmios.trainer import EfficientSpeech


def print_args(args):
    opt_log = '--------------- Options ---------------\n'
    opt = vars(args)
    for k, v in opt.items():
        opt_log += f'{str(k)}: {str(v)}\n'
    opt_log += '---------------------------------------\n'
    print(opt_log)
    return opt_log


class FineTuneCommandParser(Tap):
    config: Path  # Path to processing config file (yaml)
    checkpoint: Path # Path to model checkpoint
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

    hifigan_onnx_path: Path = Path("hifigan/hifigan_16k_light.onnx")


if __name__ == "__main__":
    args = FineTuneCommandParser().parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    config = yaml.load(open(args.config, "r"), Loader=yaml.FullLoader)
    preprocessing_config = PreprocessingConfig(**config["preprocessing"])
    dataset_folder = DatasetFolder(root_path=Path(config["dataset"]["root_path"]),
                                   ds_name=config["dataset"].get("name"))
    args.num_workers *= args.devices
    torch.set_float32_matmul_precision('high')

    datamodule = OgmiosDataModule(dataset_folder=dataset_folder,
                                  preprocess_config=preprocessing_config,
                                  batch_size=args.batch_size,
                                  num_workers=args.num_workers)

    model = EfficientSpeech.load_from_checkpoint(checkpoint_path=args.checkpoint,
                                                 map_location=args.accelerator)
    model.lr = args.lr
    model.weight_decay = args.weight_decay
    model.max_epochs = args.max_epochs


    if args.verbose:
        print_args(args)

    tb_logger = TensorBoardLogger("tb_logs", name=f"ogmios_ft_{dataset_folder.name}")

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
