import datetime
import logging
from pathlib import Path
from typing import Literal

import torch
import yaml
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from tap import Tap

from ogmios.config import ModelConfig, DatasetParams
from ogmios.datamodule import OgmiosDataModule
from ogmios.dataset.commons import PreprocessingConfig, DatasetFolder
from ogmios.hifigan import HifiganOnnxModel
from ogmios.lightning import EfficientSpeech


def print_args(args):
    opt_log = '--------------- Options ---------------\n'
    opt = vars(args)
    for k, v in opt.items():
        opt_log += f'{str(k)}: {str(v)}\n'
    opt_log += '---------------------------------------\n'
    print(opt_log)
    return opt_log


MODEL_CONFIGS = {
    "tiny": ModelConfig(depth=2,
                        block_depth=2,
                        n_blocks=2,
                        reduction=4,
                        head=1,
                        embed_dim=128,
                        kernel_size=3,
                        decoder_kernel_size=5,
                        expansion=1),
    "small": ModelConfig(depth=2,
                         block_depth=2,
                         n_blocks=3,
                         reduction=2,
                         head=1,
                         embed_dim=256,
                         kernel_size=3,
                         decoder_kernel_size=5,
                         expansion=1),
    "base": ModelConfig(depth=2,
                        block_depth=3,
                        n_blocks=3,
                        reduction=1,
                        head=2,
                        embed_dim=128,
                        kernel_size=5,
                        decoder_kernel_size=5,
                        expansion=2),
}


class TrainCommandParser(Tap):
    config: Path  # Path to processing config file (yaml)
    verbose: bool = False

    accelerator: Literal['gpu', 'cpu'] = 'gpu'
    devices: int = 1
    iter: int = 1
    threads: int = 24
    precision: Literal["bf16-mixed", "16-mixed", 16, 32, 64] = "16-mixed"

    num_workers: int = 4
    max_epochs: int = 5000
    warmup_epochs: int = 50
    weight_decay: float = 1e-5  # Optimizer weight decay
    lr: float = 1e-3  # Learning rate for AdamW
    val_every_epoch: int = 5  # Run val every N epochs
    batch_size: int = 128  # Batch size

    model_config: Literal["tiny", "small", "base"] = "tiny"

    hifigan_onnx_path: Path = Path("hifigan/hifigan_16k_light.onnx")


if __name__ == "__main__":
    args = TrainCommandParser().parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    config = yaml.load(open(args.config, "r"), Loader=yaml.FullLoader)
    preprocess_cfg = PreprocessingConfig(**config["preprocessing"])
    dataset_folder = DatasetFolder(root_path=Path(config["dataset"]["root_path"]),
                                   ds_name=config["dataset"].get("name"))
    args.num_workers *= args.devices
    torch.set_float32_matmul_precision('high')
    model_config = MODEL_CONFIGS[args.model_config]
    dataset_params = DatasetParams(pitch_stats=dataset_folder.stats["pitch"],
                                   energy_stats=dataset_folder.stats["energy"],
                                   dim_alphabet=len(dataset_folder.phonemes),
                                   n_mels=preprocess_cfg.mel.n_mel_channels)

    datamodule = OgmiosDataModule(dataset_folder=dataset_folder,
                                  preprocess_config=preprocess_cfg,
                                  batch_size=args.batch_size,
                                  num_workers=args.num_workers)

    hifigan_onnx = HifiganOnnxModel(args.hifigan_onnx_path)
    pl_model = EfficientSpeech(model_config=model_config,
                               dataset_params=dataset_params,
                               hifigan_onnx=hifigan_onnx,
                               sampling_rate=preprocess_cfg.sampling_rate,
                               lr=args.lr,
                               weight_decay=args.weight_decay,
                               max_epochs=args.max_epochs)

    if args.verbose:
        print_args(args)

    tb_logger = TensorBoardLogger("tb_logs", name=f"ogmios_{dataset_folder.name}")

    checkpoint_callback = ModelCheckpoint(
        dirpath=tb_logger.log_dir,  # Dossier où sauvegarder
        filename="last",  # Toujours le même nom de fichier
        save_last=True,  # Optionnel, pour garantir une dernière sauvegarde aussi
        save_weights_only=False,  # Sauvegarder tout le modèle (poids et architecture)
        save_on_train_epoch_end=True,
    )

    trainer = Trainer(accelerator=args.accelerator,
                      devices=args.devices,
                      precision=args.precision,
                      check_val_every_n_epoch=args.val_every_epoch,
                      max_epochs=args.max_epochs,
                      logger=tb_logger,
                      log_every_n_steps=25,
                      callbacks=[checkpoint_callback])

    start_time = datetime.datetime.now()
    trainer.fit(pl_model, datamodule=datamodule)
    elapsed_time = datetime.datetime.now() - start_time
    print(f"Training time: {elapsed_time}")
