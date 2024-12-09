import csv
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, Iterable

import librosa
import numpy as np
import pyworld as pw
import tgt
import torch
from pydantic import BaseModel
from scipy.interpolate import interp1d
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tqdm import tqdm

from .aligments import AlignmentFile
from ..audio.stft import TacotronSTFT


# TODO : deserialize preprocessing parameters using dataclasses
# TODO: adapt preprocessor class to new implem

@dataclass
class DatasetFolder:
    root_path: Path
    ds_name: Optional[str] = None

    @property
    def name(self):
        return self.ds_name if self.ds_name is not None else self.root_path.name

    @property
    def wavs_folder(self):
        return self.root_path / "wavs"

    @property
    def alignments_folder(self):
        return self.root_path / "alignments"

    @property
    def preprocessed_folder(self):
        return self.root_path / "preprocessed"

    @property
    def pitches_folder(self):
        return self.preprocessed_folder / "pitch"

    @property
    def energies_folder(self):
        return self.preprocessed_folder / "energy"

    @property
    def durations_folder(self):
        return self.preprocessed_folder / "duration"

    @property
    def mels_folder(self):
        return self.preprocessed_folder / "mel"

    @property
    def transcripts(self) -> dict[str, str]:
        with open(self.root_path / "transcripts.csv", "r") as f:
            return {t[0]: t[1] for t in csv.reader(f, delimiter="\t")}

    def __iter__(self) -> Iterable[tuple[Path, Path]]:
        for wav_path in self.wavs_folder.glob("*.wav"):
            yield wav_path, self.alignments_folder / f"{wav_path.name}.json"


class PreprocessingConfig(BaseModel):
    val_size: int = 512

    sampling_rate: int = 22050

    class TextConfig(BaseModel):
        language: str
        max_length: int = 4096

    text: TextConfig

    class STFTConfig(BaseModel):
        filter_length: int = 1024
        hop_length: int = 256
        win_length: int = 1024

    STFT: STFTConfig

    class MelConfig(BaseModel):
        n_mel_channels: int = 80
        mel_fmin: int = 0
        mel_fmax: int = 8000  # Set to 8000 for hifigan

    mel: MelConfig

    class PitchConfig(BaseModel):
        feature_level: Literal["phoneme", "frame"] = "phoneme"
        normalization: bool = True

    pitch: PitchConfig

    class EnergyConfig(BaseModel):
        feature_level: Literal["phoneme", "frame"] = "phoneme"
        normalization: bool = True

    energy: EnergyConfig


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
        dead_phones = {"sil", "sp", "spn"}
        # filtering out silent phones
        phones = [p for p in aligments.phones.entries if p.annot not in dead_phones]
        starts, ends, phones = zip(*((p.stat, p.end, p.annot) for p in phones))
        starts = np.array(starts)
        ends = np.array(ends)
        sampling_factor = self.config.sampling_rate / self.config.stft.hop_length

        # computing duration (in number of mel frames) for each phones
        durations = (np.round(ends * sampling_factor) - np.round(starts * sampling_factor)).int()

        return phones, durations, starts.min(), ends.max()

    def compute_pitch(self, wav: np.ndarray, durations: np.ndarray) -> np.ndarray:
        pitch, t = pw.dio(
            wav.astype(np.float64),
            self.config.sampling_rate,
            frame_period=self.config.stft.hop_length / self.config.sampling_rate * 1000,
        )
        pitch = pw.stonemask(wav.astype(np.float64), pitch, t, self.sampling_rate)

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
        with open(metadata_csv_path, "w") as md_f:
            csv_writer = csv.writer(md_f, delimiter="\t")
            for wav_path, align_path in self.dataset:
                try:
                    phones, pitch, energy, total_frames = self.process_utterance(wav_path, align_path)
                except PitchProcessingError:
                    continue
                csv_writer.writerow([
                    wav_path.stem,
                    "|".join(phones),
                    transcripts[wav_path.stem]])

                # TODO: filter outliers

                pitch_scaler.partial_fit(pitch.reshape((-1, 1)))
                pitch_minmax_scaler.partial_fit(pitch.reshape((-1, 1)))
                energy_scaler.partial_fit(energy.reshape((-1, 1)))
                energy_minmax_scaler.partial_fit(energy.reshape((-1, 1)))
                n_frames += total_frames

        # TODO: investigate into the normalisation thing.
        #   Norm should bring std and mean to 1 and 0.
        #  It seems that there is a mismatch between the data and the stats
        pitch_mean = pitch_scaler.mean_[0]
        pitch_std = pitch_scaler.scale_[0]
        if self.config.pitch.normalization:
            self.normalize(self.dataset.pitches_folder, pitch_mean, pitch_std)

        energy_mean = energy_scaler.mean_[0]
        energy_std = energy_scaler.scale_[0]
        if self.config.energy.normalization:
            self.normalize(self.dataset.energies_folder, energy_mean, energy_std)

    def save_stats(self):
        pass

    def save_splits(self):
        pass

    def remove_outlier(self, values: np.ndarray):
        p25 = np.percentile(values, 25)
        p75 = np.percentile(values, 75)
        lower = p25 - 1.5 * (p75 - p25)
        upper = p75 + 1.5 * (p75 - p25)
        normal_indices = np.logical_and(values > lower, values < upper)

        return values[normal_indices]

    def normalize(self, in_dir: Path, mean: float, std: float):
        for filename in in_dir.iterdir():
            filename = os.path.join(in_dir, filename)
            values = (np.load(filename) - mean) / std
            np.save(filename, values)




class Preprocessor:
    def __init__(self, config):
        self.config = config
        self.in_dir = config["path"]["raw_path"]
        self.tg_dir = config["path"]["tg_path"]
        self.out_dir = config["path"]["preprocessed_path"]
        self.val_size = config["preprocessing"]["val_size"]
        self.sampling_rate = config["preprocessing"]["audio"]["sampling_rate"]
        self.hop_length = config["preprocessing"]["stft"]["hop_length"]

        assert config["preprocessing"]["pitch"]["feature"] in [
            "phoneme_level",
            "frame_level",
        ]
        assert config["preprocessing"]["energy"]["feature"] in [
            "phoneme_level",
            "frame_level",
        ]
        self.pitch_phoneme_averaging = (
                config["preprocessing"]["pitch"]["feature"] == "phoneme_level"
        )
        self.energy_phoneme_averaging = (
                config["preprocessing"]["energy"]["feature"] == "phoneme_level"
        )

        self.pitch_normalization = config["preprocessing"]["pitch"]["normalization"]
        self.energy_normalization = config["preprocessing"]["energy"]["normalization"]

        self.STFT = Audio.stft.TacotronSTFT(
            config["preprocessing"]["stft"]["filter_length"],
            config["preprocessing"]["stft"]["hop_length"],
            config["preprocessing"]["stft"]["win_length"],
            config["preprocessing"]["mel"]["n_mel_channels"],
            config["preprocessing"]["audio"]["sampling_rate"],
            config["preprocessing"]["mel"]["mel_fmin"],
            config["preprocessing"]["mel"]["mel_fmax"],
        )

    def build_from_path(self):
        os.makedirs((os.path.join(self.out_dir, "mel")), exist_ok=True)
        os.makedirs((os.path.join(self.out_dir, "pitch")), exist_ok=True)
        os.makedirs((os.path.join(self.out_dir, "energy")), exist_ok=True)
        os.makedirs((os.path.join(self.out_dir, "duration")), exist_ok=True)

        print("Processing Data ...")
        out = list()
        n_frames = 0  # used for total time count
        pitch_scaler = StandardScaler()
        energy_scaler = StandardScaler()

        # Compute pitch, energy, duration, and mel-spectrogram
        speakers = {}
        for i, wav_name in enumerate(tqdm(os.listdir(self.in_dir))):
            if ".wav" not in wav_name:
                continue

            basename = wav_name.split(".")[0]
            tg_path = os.path.join(
                self.tg_dir, "{}.TextGrid".format(basename)
            )
            if os.path.exists(tg_path):
                ret = self.process_utterance(basename)
                if ret is None:
                    continue

                info, pitch, energy, n = ret
                out.append(info)

            if len(pitch) > 0:
                pitch_scaler.partial_fit(pitch.reshape((-1, 1)))
            if len(energy) > 0:
                energy_scaler.partial_fit(energy.reshape((-1, 1)))

            n_frames += n

        print("Computing statistic quantities ...")
        # Perform normalization if necessary
        if self.pitch_normalization:
            pitch_mean = pitch_scaler.mean_[0]
            pitch_std = pitch_scaler.scale_[0]
        else:
            # A numerical trick to avoid normalization...
            pitch_mean = 0
            pitch_std = 1
        if self.energy_normalization:
            energy_mean = energy_scaler.mean_[0]
            energy_std = energy_scaler.scale_[0]
        else:
            energy_mean = 0
            energy_std = 1

        pitch_min, pitch_max = self.normalize(
            os.path.join(self.out_dir, "pitch"), pitch_mean, pitch_std
        )
        energy_min, energy_max = self.normalize(
            os.path.join(self.out_dir, "energy"), energy_mean, energy_std
        )

        # Save files
        with open(os.path.join(self.out_dir, "speakers.json"), "w") as f:
            f.write(json.dumps(speakers))

        with open(os.path.join(self.out_dir, "stats.json"), "w") as f:
            stats = {
                "pitch": [
                    float(pitch_min),
                    float(pitch_max),
                    float(pitch_mean),
                    float(pitch_std),
                ],
                "energy": [
                    float(energy_min),
                    float(energy_max),
                    float(energy_mean),
                    float(energy_std),
                ],
            }
            f.write(json.dumps(stats))

        print(
            "Total time: {} hours".format(
                n_frames * self.hop_length / self.sampling_rate / 3600
            )
        )

        random.shuffle(out)
        out = [r for r in out if r is not None]

        # Write metadata
        with open(os.path.join(self.out_dir, "train.txt"), "w", encoding="utf-8") as f:
            for m in out[self.val_size:]:
                f.write(m + "\n")
        with open(os.path.join(self.out_dir, "val.txt"), "w", encoding="utf-8") as f:
            for m in out[: self.val_size]:
                f.write(m + "\n")

        return out

    def process_utterance(self, basename):
        wav_path = os.path.join(self.in_dir, "{}.wav".format(basename))
        text_path = os.path.join(self.in_dir, "{}.lab".format(basename))
        tg_path = os.path.join(
            self.tg_dir, "{}.TextGrid".format(basename)
        )

        # Get alignments
        textgrid = tgt.io.read_textgrid(tg_path)
        phone, duration, start, end = self.get_alignment(
            textgrid.get_tier_by_name("phones")
        )
        text = "{" + " ".join(phone) + "}"
        if start >= end:
            return None

        # Read and trim wav files
        wav, _ = librosa.load(wav_path)
        wav = wav[
              int(self.sampling_rate * start): int(self.sampling_rate * end)
              ].astype(np.float32)

        # Read raw text
        with open(text_path, "r") as f:
            raw_text = f.readline().strip("\n")

        # Compute fundamental frequency
        pitch, t = pw.dio(
            wav.astype(np.float64),
            self.sampling_rate,
            frame_period=self.hop_length / self.sampling_rate * 1000,
        )
        pitch = pw.stonemask(wav.astype(np.float64), pitch, t, self.sampling_rate)

        pitch = pitch[: sum(duration)]
        if np.sum(pitch != 0) <= 1:
            return None

        # Compute mel-scale spectrogram and energy
        mel_spectrogram, energy = Audio.tools.get_mel_from_wav(wav, self.STFT)
        mel_spectrogram = mel_spectrogram[:, : sum(duration)]
        energy = energy[: sum(duration)]

        if self.pitch_phoneme_averaging:
            # perform linear interpolation
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
            for i, d in enumerate(duration):
                if d > 0:
                    pitch[i] = np.mean(pitch[pos: pos + d])
                else:
                    pitch[i] = 0
                pos += d
            pitch = pitch[: len(duration)]

        if self.energy_phoneme_averaging:
            # Phoneme-level average
            pos = 0
            for i, d in enumerate(duration):
                if d > 0:
                    energy[i] = np.mean(energy[pos: pos + d])
                else:
                    energy[i] = 0
                pos += d
            energy = energy[: len(duration)]

        # Save files
        dur_filename = f"duration-{basename}.npy"
        np.save(os.path.join(self.out_dir, "duration", dur_filename), duration)

        pitch_filename = f"pitch-{basename}.npy"
        np.save(os.path.join(self.out_dir, "pitch", pitch_filename), pitch)

        energy_filename = f"energy-{basename}.npy"
        np.save(os.path.join(self.out_dir, "energy", energy_filename), energy)

        mel_filename = f"mel-{basename}.npy"
        np.save(
            os.path.join(self.out_dir, "mel", mel_filename),
            mel_spectrogram.T,
        )

        return (
            "|".join([basename, text, raw_text]),
            self.remove_outlier(pitch),
            self.remove_outlier(energy),
            mel_spectrogram.shape[1],
        )

    def get_alignment(self, tier):
        sil_phones = ["sil", "sp", "spn"]

        phones = []
        durations = []
        start_time = 0
        end_time = 0
        end_idx = 0
        for t in tier._objects:
            s, e, p = t.start_time, t.end_time, t.text

            # Trim leading silences
            if phones == []:
                if p in sil_phones:
                    continue
                else:
                    start_time = s

            if p not in sil_phones:
                # For ordinary phones
                phones.append(p)
                end_time = e
                end_idx = len(phones)
            else:
                # For silent phones
                phones.append(p)

            durations.append(
                int(
                    np.round(e * self.sampling_rate / self.hop_length)
                    - np.round(s * self.sampling_rate / self.hop_length)
                )
            )

        # Trim tailing silences
        phones = phones[:end_idx]
        durations = durations[:end_idx]

        return phones, durations, start_time, end_time

    def remove_outlier(self, values):
        values = np.array(values)
        p25 = np.percentile(values, 25)
        p75 = np.percentile(values, 75)
        lower = p25 - 1.5 * (p75 - p25)
        upper = p75 + 1.5 * (p75 - p25)
        normal_indices = np.logical_and(values > lower, values < upper)

        return values[normal_indices]

    def normalize(self, in_dir, mean, std):
        max_value = np.finfo(np.float64).min
        min_value = np.finfo(np.float64).max
        for filename in os.listdir(in_dir):
            filename = os.path.join(in_dir, filename)
            values = (np.load(filename) - mean) / std
            np.save(filename, values)

            max_value = max(max_value, max(values))
            min_value = min(min_value, min(values))

        return min_value, max_value
