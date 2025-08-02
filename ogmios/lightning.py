'''
EfficientSpeech: An On-Device Text to Speech Model
https://ieeexplore.ieee.org/abstract/document/10094639
Rowel Atienza
Apache 2.0 License
2023
'''

import math
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from lightning import LightningModule
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from ogmios.config import ModelConfig, DatasetParams
from ogmios.datamodule import OgmiosBatch
from ogmios.hifigan import HifiganOnnxModel
from ogmios.layers import Phoneme2Mel, MelDecoder, PhonemeEncoder
from ogmios.layers.networks import Phone2MelPredictions
from ogmios.utils import plot_spectrogram_to_numpy


# chatgpt
def get_lr_scheduler(optimizer, warmup_steps, total_steps, min_lr=0):
    """
    Create a learning rate scheduler with linear warm-up and cosine learning rate decay.

    Args:
        optimizer (torch.optim.Optimizer): The optimizer for which to create the scheduler.
        warmup_steps (int): The number of warm-up steps.
        total_steps (int): The total number of steps.
        min_lr (float, optional): The minimum learning rate at the end of the decay. Default: 0.

    Returns:
        torch.optim.lr_scheduler.LambdaLR: The learning rate scheduler.
    """

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            # Linear warm-up
            return float(current_step) / float(max(1, warmup_steps))
        else:
            # Cosine learning rate decay
            progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return max(min_lr, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = LambdaLR(optimizer, lr_lambda)
    return scheduler


class EfficientSpeech(LightningModule):
    def __init__(self,
                 model_config: ModelConfig,
                 dataset_params: DatasetParams,
                 sampling_rate: int,
                 lr: float = 1e-3,
                 weight_decay: float = 1e-6,
                 max_epochs: int = 5000,
                 hifigan_onnx: Optional[HifiganOnnxModel] = None,
                 ):
        super().__init__()
        self.save_hyperparameters(ignore=["hifigan_onnx"])
        self.sampling_rate = sampling_rate
        self.lr = lr
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.hifigan = hifigan_onnx

        phoneme_encoder = PhonemeEncoder(alphabet_dim=dataset_params.dim_alphabet,
                                         pitch_stats=dataset_params.pitch_stats,
                                         energy_stats=dataset_params.energy_stats,
                                         depth=model_config.depth,
                                         reduction=model_config.reduction,
                                         head=model_config.head,
                                         embed_dim=model_config.embed_dim,
                                         kernel_size=model_config.kernel_size,
                                         expansion=model_config.expansion, )

        mel_decoder = MelDecoder(dim=model_config.embed_dim // model_config.reduction,
                                 n_mel_channels=dataset_params.n_mels,
                                 kernel_size=model_config.decoder_kernel_size,
                                 n_blocks=model_config.n_blocks,
                                 block_depth=model_config.block_depth)

        self.phoneme2mel = Phoneme2Mel(encoder=phoneme_encoder,
                                       decoder=mel_decoder)

    def forward(self, x):
        return self.phoneme2mel(x, train=self.training)

    def loss(self, y_hat, y, x):
        pitch_pred = y_hat["pitch"]
        energy_pred = y_hat["energy"]
        duration_pred = y_hat["duration"]
        mel_pred = y_hat["mel"]

        phoneme_mask = x["phoneme_mask"]
        mel_mask = x["mel_mask"]

        pitch = x["pitch"]
        energy = x["energy"]
        duration = x["duration"]
        mel = y["mel"]

        mel_mask = ~mel_mask
        mel_mask = mel_mask.unsqueeze(-1)
        target = mel.masked_select(mel_mask)
        pred = mel_pred.masked_select(mel_mask)
        mel_loss = nn.L1Loss()(pred, target)

        phoneme_mask = ~phoneme_mask

        pitch_pred = pitch_pred[:, :pitch.shape[-1]]
        pitch_pred = torch.squeeze(pitch_pred)
        pitch = pitch.masked_select(phoneme_mask)
        pitch_pred = pitch_pred.masked_select(phoneme_mask)
        pitch_loss = nn.MSELoss()(pitch_pred, pitch)

        energy_pred = energy_pred[:, :energy.shape[-1]]
        energy_pred = torch.squeeze(energy_pred)
        energy = energy.masked_select(phoneme_mask)
        energy_pred = energy_pred.masked_select(phoneme_mask)
        energy_loss = nn.MSELoss()(energy_pred, energy)

        duration_pred = duration_pred[:, :duration.shape[-1]]
        duration_pred = torch.squeeze(duration_pred)
        duration = duration.masked_select(phoneme_mask)
        duration_pred = duration_pred.masked_select(phoneme_mask)
        duration = torch.log(duration.float() + 1)
        duration_pred = torch.log(duration_pred.float() + 1)
        duration_loss = nn.MSELoss()(duration_pred, duration)

        return mel_loss, pitch_loss, energy_loss, duration_loss

    def training_step(self, batch: OgmiosBatch, batch_idx: int):
        x, y = batch
        y_hat = self.forward(x)

        mel_loss, pitch_loss, energy_loss, duration_loss = self.loss(y_hat, y, x)
        loss = (10. * mel_loss) + (2. * pitch_loss) + (2. * energy_loss) + duration_loss

        self.log("mel_loss", mel_loss, on_step=False, on_epoch=True)
        self.log("pitch_loss", pitch_loss, on_step=False, on_epoch=True)
        self.log("energy_loss", energy_loss, on_step=False, on_epoch=True)
        self.log("dur_loss", duration_loss, on_step=False, on_epoch=True)
        self.log("loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def on_train_epoch_end(self):
        self.log("lr", self.scheduler.get_last_lr()[0], on_epoch=True, prog_bar=True)

    def predict_step(self, batch: OgmiosBatch, batch_idx: int = 0, dataloader_idx: int = 0) -> Phone2MelPredictions:
        return self.phoneme2mel(batch, train=False)

    def synth_wav(self, mel: np.ndarray):
        mel = mel.transpose(1, 0).astype(np.float32)
        wav = self.hifigan.synth(mel)
        return wav

    def log_predictions(self, index: int,
                        predictions: Phone2MelPredictions,
                        mels_gt: torch.Tensor,
                        mel_lens_gt: torch.Tensor):
        mel_len = predictions["mel_len"][index].int()
        mel_pred = predictions["mel"][index][:mel_len, :]

        mel_len_gt = mel_lens_gt[index]
        mel_gt = mels_gt[index, :mel_len_gt, :]

        wav_pred = self.synth_wav(mel_pred.cpu().numpy())
        wav_resynth = self.synth_wav(mel_gt.cpu().numpy())

        self.logger.experiment.add_image(
            f"mel/pred_{i}",
            plot_spectrogram_to_numpy(mel_pred),
            global_step=self.global_step,
            dataformats="HWC"
        )

        self.logger.experiment.add_image(
            f"mel/target_{i}",
            plot_spectrogram_to_numpy(mel_gt),
            global_step=self.global_step,
            dataformats="HWC"
        )

        self.logger.experiment.add_audio(
            tag=f"wav/predicted_{i}",
            snd_tensor=wav_pred,
            global_step=self.global_step,
            sample_rate=self.sampling_rate)

        # then logging resynthesis of ground truth mel (through hifigan)
        self.logger.experiment.add_audio(
            tag=f"wav/resynth_{i}",
            snd_tensor=wav_resynth,
            global_step=self.global_step,
            sample_rate=self.sampling_rate)

    def validation_step(self, batch, batch_idx):
        if batch_idx != 0 or self.current_epoch < 1:
            return

        x, y = batch
        mels_gt = y["mel"]
        mel_lens_gt = x["mel_len"]
        predictions = self.predict_step(x)

        mel_loss, pitch_loss, energy_loss, duration_loss = self.loss(predictions, y, x)
        self.log("val_mel_loss", mel_loss)
        self.log("val_pitch_loss", pitch_loss)
        self.log("val_energy_loss", energy_loss)
        self.log("val_dur_loss", duration_loss)

        for i in range(10):
            self.log_predictions(i, predictions, mels_gt, mel_lens_gt)

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.scheduler = get_lr_scheduler(optimizer, 50, self.max_epochs, min_lr=0)

        return [optimizer], [self.scheduler]
