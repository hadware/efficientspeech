'''
EfficientSpeech: An On-Device Text to Speech Model
https://ieeexplore.ieee.org/abstract/document/10094639
Rowel Atienza
Apache 2.0 License
2023
'''

import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from lightning import LightningModule
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from ogmios.datamodule import OgmiosBatch
from ogmios.dataset.commons import DatasetFolder, PreprocessingConfig
from ogmios.hifigan import HifiganOnnxModel
from ogmios.layers import PhonemeEncoder, MelDecoder, Phoneme2Mel
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
                 dataset_folder: DatasetFolder,
                 preprocess_config: PreprocessingConfig,
                 lr: float = 1e-3,
                 weight_decay: float = 1e-6,
                 max_epochs: int = 5000,
                 depth: int = 2,
                 n_blocks: int = 2,
                 block_depth: int = 2,
                 reduction: int = 4,
                 head: int = 1,
                 embed_dim: int = 128,
                 kernel_size: int = 3,
                 decoder_kernel_size: int = 3,
                 expansion: int = 1,
                 hifigan_onnx_path: Path = None):
        super().__init__()

        self.save_hyperparameters()

        phoneme_encoder = PhonemeEncoder(alphabet_dim=len(dataset_folder.phonemes),
                                         pitch_stats=dataset_folder.stats["pitch"],
                                         energy_stats=dataset_folder.stats["energy"],
                                         depth=depth,
                                         reduction=reduction,
                                         head=head,
                                         embed_dim=embed_dim,
                                         kernel_size=kernel_size,
                                         expansion=expansion)

        mel_decoder = MelDecoder(dim=embed_dim // reduction,
                                 n_mel_channels=preprocess_config.mel.n_mel_channels,
                                 kernel_size=decoder_kernel_size,
                                 n_blocks=n_blocks,
                                 block_depth=block_depth)

        self.phoneme2mel = Phoneme2Mel(encoder=phoneme_encoder,
                                       decoder=mel_decoder)

        self.hifigan = HifiganOnnxModel(hifigan_onnx_path)

        self.training_step_outputs = []

    def forward(self, x):
        return self.phoneme2mel(x, train=self.training)

    def predict_step(self, batch, batch_idx=0, dataloader_idx=0):
        batch_mels, batch_mel_lens, batch_durations = self.phoneme2mel(batch, train=False)
        batch_mels = batch_mels.transpose(1, 2)
        batch_mel_lens = batch_mel_lens.int()
        predictions = []
        with torch.no_grad():
            for mel, mel_len, duration in zip(batch_mels, batch_mel_lens, batch_durations):
                mel = mel[:, :mel_len]
                wav = self.hifigan.synth(mel.cpu().numpy().astype(np.float32))
                predictions.append((wav, mel))

        return predictions

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

        losses = {"loss": loss,
                  "mel_loss": mel_loss,
                  "pitch_loss": pitch_loss,
                  "energy_loss": energy_loss,
                  "duration_loss": duration_loss}
        self.training_step_outputs.append(losses)

        return loss

    def on_train_epoch_end(self):
        avg_loss = torch.stack([x["loss"] for x in self.training_step_outputs]).mean()
        avg_mel_loss = torch.stack([x["mel_loss"] for x in self.training_step_outputs]).mean()
        avg_pitch_loss = torch.stack([x["pitch_loss"] for x in self.training_step_outputs]).mean()
        avg_energy_loss = torch.stack(
            [x["energy_loss"] for x in self.training_step_outputs]).mean()
        avg_duration_loss = torch.stack(
            [x["duration_loss"] for x in self.training_step_outputs]).mean()
        self.log("mel", avg_mel_loss, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("pitch", avg_pitch_loss, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("energy", avg_energy_loss, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("dur", avg_duration_loss, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("loss", avg_loss, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("lr", self.scheduler.get_last_lr()[0], on_epoch=True, prog_bar=True, sync_dist=True)
        self.training_step_outputs.clear()

    def validation_step(self, batch, batch_idx):

        if batch_idx != 0 or self.current_epoch < 1:
            return

        # first logging predictions from eval batch
        x, y = batch
        mels_gt = y["mel"]
        mel_lens_gt = x["mel_len"]
        for i, (wav, mels_pred) in enumerate(self.predict_step(x)):
            mels_pred = mels_pred.cpu().numpy()

            self.logger.experiment.add_image(
                f"mel/pred_{i}",
                plot_spectrogram_to_numpy(mels_pred),
                global_step=self.global_step,
                dataformats="HWC"
            )
            self.logger.experiment.add_audio(
                tag=f"wav/predicted_{i}",
                snd_tensor=wav,
                global_step=self.global_step,
                sample_rate=self.hparams.preprocess_config.sampling_rate)

            # then logging resynthesis of ground truth mel (through hifigan)
            with torch.no_grad():
                mel_gt = mels_gt[i, :mel_lens_gt[i], :].cpu().numpy().transpose(1, 0)
                wav_gt = self.hifigan.synth(mel_gt)

                self.logger.experiment.add_audio(
                    tag=f"wav/resynth_{i}",
                    snd_tensor=wav_gt,
                    global_step=self.global_step,
                    sample_rate=self.hparams.preprocess_config.sampling_rate)

                self.logger.experiment.add_image(
                    f"mel/target_{i}",
                    plot_spectrogram_to_numpy(mel_gt),
                    global_step=self.global_step,
                    dataformats="HWC"
                )

    def on_test_epoch_end(self):
        pass

    def on_validation_epoch_end(self):
        pass

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        self.scheduler = get_lr_scheduler(optimizer, 50, self.hparams.max_epochs, min_lr=0)

        return [optimizer], [self.scheduler]
