import csv
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Iterable, Literal

from pydantic import BaseModel

from ogmios.utils import logger


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


    @property
    def phonemes(self) -> list[str]:
        with open(self.preprocessed_folder / "phones.json") as f:
            return json.load(f)


    @property
    def stats(self) -> dict[str, "AcousticStats"]:
        with open(self.preprocessed_folder / "stats.json") as f:
            return json.load(f)

    def __iter__(self) -> Iterable[tuple[Path, Path]]:
        all_alignments = {
            p.stem : p for p in self.alignments_folder.glob("**/*.json")
        }
        for wav_path in self.wavs_folder.glob("*.wav"):
            try:
                yield wav_path, all_alignments[wav_path.stem]
            except KeyError:
                logger.debug(f"Missing alignment for {wav_path.stem}")
                continue


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

    stft: STFTConfig

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


class AcousticStats(BaseModel):
    mean: float
    std: float
    min: float
    max: float
