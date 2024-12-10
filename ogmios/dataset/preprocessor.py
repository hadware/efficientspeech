import csv
import json
import random
from pathlib import Path
from random import shuffle

import librosa
import numpy as np
import pyworld as pw
import torch
from scipy.interpolate import interp1d
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tqdm import tqdm

from .aligments import AlignmentFile
from .commons import DatasetFolder, PreprocessingConfig, AcousticStats
from ..audio.stft import TacotronSTFT
from ..utils import logger


class PitchProcessingError(Exception):
    pass


class DatasetPreprocessor:
    def __init__(self, config: PreprocessingConfig, dataset: DatasetFolder):
        self.config = config
        self.dataset = dataset

        assert config.pitch.feature_level in ["phoneme", "frame"]
        assert config.energy.feature_level in ["phoneme", "frame"]

        self.STFT = TacotronSTFT(
            filter_length=config.stft.filter_length,
            hop_length=config.stft.hop_length,
            win_length=config.stft.win_length,
            n_mel_channels=config.mel.n_mel_channels,
            sampling_rate=config.sampling_rate,
            mel_fmin=config.mel.mel_fmin,
            mel_fmax=config.mel.mel_fmax,
        )

    def post_process_alignments(self, aligments: AlignmentFile) \
            -> tuple[list[str], np.ndarray, float, float]:
        # TODO : look back into this again: silences should be just trimmed and not completely filtered out
        dead_phones = {"sil", "sp", "spn"}
        # filtering out silent phones
        filtered_phones = [p for p in aligments.phones.intervals if p.annot not in dead_phones]
        starts, ends, filtered_phones = zip(*((p.start, p.end, p.annot) for p in filtered_phones))
        starts = np.array(starts)
        ends = np.array(ends)
        sampling_factor = self.config.sampling_rate / self.config.stft.hop_length

        # computing duration (in number of mel frames) for each phones
        durations = (np.round(ends * sampling_factor) - np.round(starts * sampling_factor)).astype(int)

        return filtered_phones, durations, starts.min(), ends.max()

    def compute_pitch(self, wav: np.ndarray, durations: np.ndarray) -> np.ndarray:
        pitch, t = pw.dio(
            wav.astype(np.float64),
            self.config.sampling_rate,
            frame_period=self.config.stft.hop_length / self.config.sampling_rate * 1000,
        )
        pitch = pw.stonemask(wav.astype(np.float64), pitch, t, self.config.sampling_rate)

        pitch = pitch[: durations.sum()]
        if np.sum(pitch != 0) <= 1:
            raise PitchProcessingError("Invalid pitch: all values null")

        if self.config.pitch.feature_level == "phoneme":
            # do pitch averaging per phoneme via linear interpolation
            nonzero_ids = np.where(pitch != 0)[0]
            interp_fn = interp1d(
                nonzero_ids,
                pitch[nonzero_ids],
                fill_value=(pitch[nonzero_ids[0]], pitch[nonzero_ids[-1]]),
                bounds_error=False,
            )
            pitch = interp_fn(np.arange(0, len(pitch)))

            # Phoneme-level average
            pos = 0
            for i, d in enumerate(durations):
                if d > 0:
                    pitch[i] = np.mean(pitch[pos: pos + d])
                else:
                    pitch[i] = 0
                pos += d
            pitch = pitch[: len(durations)]

        if len(pitch) == 0:
            raise PitchProcessingError("No pitch values")

        return pitch

    def compute_mel_and_energy(self, wav: np.ndarray, durations: np.ndarray) \
            -> tuple[np.ndarray, np.ndarray]:
        total_dur = durations.sum()

        with torch.no_grad():
            audio = torch.clip(torch.FloatTensor(wav).unsqueeze(0), -1, 1)
            mel, energy = self.STFT.mel_spectrogram(audio)
            mel = torch.squeeze(mel, 0).numpy().astype(np.float32)
            energy = torch.squeeze(energy, 0).numpy().astype(np.float32)

        if self.config.energy.feature_level == "phoneme":
            # do energy averaging per phoneme
            pos = 0
            for i, d in enumerate(durations):
                if d > 0:
                    energy[i] = np.mean(energy[pos: pos + d])
                else:
                    energy[i] = 0
                pos += d
            energy = energy[: len(durations)]

        return mel[:, :total_dur], energy[:total_dur]

    def process_utterance(self, wav_path: Path, alignments_path: Path):
        with open(alignments_path) as json_file:
            alignments = AlignmentFile(**json.load(json_file))

        wav, _ = librosa.load(wav_path, sr=self.config.sampling_rate)

        phones, durations, start, end = self.post_process_alignments(alignments)
        # trimming wav to first and last phone
        wav = wav[int(start * self.config.sampling_rate):int(end * self.config.sampling_rate)]
        total_duration = durations.sum()

        pitch = self.compute_pitch(wav, durations)
        mel, energy = self.compute_mel_and_energy(wav, durations)
        mel = mel.T

        np.save(self.dataset.durations_folder / f"{wav_path.stem}.npy", durations)
        np.save(self.dataset.pitches_folder / f"{wav_path.stem}.npy", pitch)
        np.save(self.dataset.energies_folder / f"{wav_path.stem}.npy", energy)
        np.save(self.dataset.mels_folder / f"{wav_path.stem}.npy", mel)

        return phones, pitch, energy, total_duration

    def process_files(self):
        transcripts = self.dataset.transcripts

        self.dataset.durations_folder.mkdir(exist_ok=True, parents=True)
        self.dataset.pitches_folder.mkdir(exist_ok=True)
        self.dataset.energies_folder.mkdir(exist_ok=True)
        self.dataset.mels_folder.mkdir(exist_ok=True)

        n_frames = 0  # used for total time count
        pitch_scaler = StandardScaler()
        energy_scaler = StandardScaler()
        pitch_minmax_scaler = MinMaxScaler()
        energy_minmax_scaler = MinMaxScaler()

        metadata_csv_path = self.dataset.preprocessed_folder / "metadata.csv"
        self.valid_ids: set[str] = set()
        self.all_phones = set()
        with open(metadata_csv_path, "w") as md_f:
            csv_writer = csv.writer(md_f, delimiter="\t")
            for wav_path, align_path in tqdm(list(self.dataset)):
                try:
                    phones, pitch, energy, total_frames = self.process_utterance(wav_path, align_path)
                except PitchProcessingError:
                    continue
                csv_writer.writerow([
                    wav_path.stem,
                    "|".join(phones),
                    transcripts[wav_path.stem]])

                pitch = self.remove_outlier(pitch)
                energy = self.remove_outlier(energy)

                self.valid_ids.add(wav_path.stem)
                self.all_phones.update(phones)
                pitch_scaler.partial_fit(pitch.reshape((-1, 1)))
                pitch_minmax_scaler.partial_fit(pitch.reshape((-1, 1)))
                energy_scaler.partial_fit(energy.reshape((-1, 1)))
                energy_minmax_scaler.partial_fit(energy.reshape((-1, 1)))
                n_frames += total_frames

        logger.info(f"Total duration: {n_frames * self.config.stft.hop_length / self.config.sampling_rate / 3600}")
        # TODO: investigate into the normalisation thing.
        #   Norm should bring std and mean to 1 and 0.
        #  It seems that there is a mismatch between the data and the stats

        # Store pitch stats then optionally normalize
        pitch_stats = AcousticStats(
            mean=pitch_scaler.mean_[0],
            std=pitch_scaler.scale_[0],
            max=pitch_minmax_scaler.data_max_[0],
            min=pitch_minmax_scaler.data_min_[0])
        if self.config.pitch.normalization:
            self.normalize(self.dataset.pitches_folder, pitch_stats.mean, pitch_stats.std)
            pitch_stats.max = (pitch_stats.max - pitch_stats.mean) / pitch_stats.std
            pitch_stats.min = (pitch_stats.min - pitch_stats.mean) / pitch_stats.std
            pitch_stats.mean = 0
            pitch_stats.std = 1
        self.pitch_stats = pitch_stats

        # Same with energy
        energy_stats = AcousticStats(
            mean=energy_scaler.mean_[0],
            std=energy_scaler.scale_[0],
            max=energy_minmax_scaler.data_max_[0],
            min=energy_minmax_scaler.data_min_[0])
        if self.config.energy.normalization:
            self.normalize(self.dataset.energies_folder, energy_stats.mean, energy_stats.std)
            energy_stats.max = (energy_stats.max - energy_stats.mean) / energy_stats.std
            energy_stats.min = (energy_stats.min - energy_stats.mean) / energy_stats.std
            energy_stats.mean = 0
            energy_stats.std = 1
        self.energy_stats = energy_stats

    def save_stats(self):
        with open(self.dataset.preprocessed_folder / "stats.json", "w") as f:
            json.dump({"pitch": self.pitch_stats.model_dump(),
                       "energy": self.energy_stats.model_dump()},
                      f)

    def save_phones(self):
        with open(self.dataset.preprocessed_folder / "phones.json", "w") as f:
            json.dump(list(sorted(self.all_phones)), f)

    def save_splits(self):
        valid_ids = list(self.valid_ids)
        random.seed(4577)
        shuffle(valid_ids)
        train_set, val_set = valid_ids[self.config.val_size:], valid_ids[:self.config.val_size]
        with open(self.dataset.preprocessed_folder / "train.txt", "w") as f:
            f.write("\n".join(train_set))
        with open(self.dataset.preprocessed_folder / "val.txt", "w") as f:
            f.write("\n".join(val_set))

    def remove_outlier(self, values: np.ndarray):
        p25 = np.percentile(values, 25)
        p75 = np.percentile(values, 75)
        lower = p25 - 1.5 * (p75 - p25)
        upper = p75 + 1.5 * (p75 - p25)
        normal_indices = np.logical_and(values > lower, values < upper)

        return values[normal_indices]

    def normalize(self, in_dir: Path, mean: float, std: float):
        for filename in in_dir.iterdir():
            values = (np.load(filename) - mean) / std
            np.save(filename, values)
