'''
EfficientSpeech: An On-Device Text to Speech Model
https://ieeexplore.ieee.org/abstract/document/10094639
Rowel Atienza, 2023
Apache 2.0 License
'''
import csv
import json
from typing import Literal

import numpy as np
import torch
from lightning import LightningDataModule
from torch.utils.data import Dataset, DataLoader

from ogmios.dataset.commons import PreprocessingConfig, DatasetFolder
from ogmios.utils import get_mask_from_lengths
from ogmios.utils import pad_1D, pad_2D


class OgmiosDataModule(LightningDataModule):
    def __init__(self,
                 dataset_folder: DatasetFolder,
                 preprocess_config: PreprocessingConfig,
                 batch_size: int = 64,
                 num_workers: int = 4):
        super().__init__()
        self.preprocess_config = preprocess_config
        self.dataset_folder = dataset_folder
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.sort = True

    def collate_fn(self, batch):
        x, y = zip(*batch)
        len_arr = np.array([d["phoneme"].shape[0] for d in x])
        idxs = np.argsort(-len_arr).tolist()

        phonemes = [x[idx]["phoneme"] for idx in idxs]
        texts = [x[idx]["text"] for idx in idxs]
        mels = [y[idx]["mel"] for idx in idxs]
        pitches = [x[idx]["pitch"] for idx in idxs]
        energies = [x[idx]["energy"] for idx in idxs]
        durations = [x[idx]["duration"] for idx in idxs]

        phoneme_lens = np.array([phoneme.shape[0] for phoneme in phonemes])
        mel_lens = np.array([mel.shape[0] for mel in mels])

        phonemes = pad_1D(phonemes)
        mels = pad_2D(mels)
        pitches = pad_1D(pitches)
        energies = pad_1D(energies)
        durations = pad_1D(durations)

        phonemes = torch.from_numpy(phonemes).int()
        phoneme_lens = torch.from_numpy(phoneme_lens).int()
        max_phoneme_len = torch.max(phoneme_lens).item()
        phoneme_mask = get_mask_from_lengths(phoneme_lens, max_phoneme_len)

        pitches = torch.from_numpy(pitches).float()
        energies = torch.from_numpy(energies).float()
        durations = torch.from_numpy(durations).int()

        mels = torch.from_numpy(mels).float()
        mel_lens = torch.from_numpy(mel_lens).int()
        max_mel_len = torch.max(mel_lens).item()
        mel_mask = get_mask_from_lengths(mel_lens, max_mel_len)

        # TODO: define typedict for this
        x = {"phoneme": phonemes,
             "phoneme_len": phoneme_lens,
             "phoneme_mask": phoneme_mask,
             "text": texts,
             "mel_len": mel_lens,
             "mel_mask": mel_mask,
             "pitch": pitches,
             "energy": energies,
             "duration": durations}

        y = {"mel": mels}

        return x, y

    def prepare_data(self):
        self.train_dataset = OgmiosDataset(self.dataset_folder,
                                           self.preprocess_config,
                                           "train")
        self.test_dataset = OgmiosDataset(self.dataset_folder,
                                          self.preprocess_config,
                                          "val")

    def setup(self, stage=None):
        self.prepare_data()

    def train_dataloader(self):
        self.train_dataloader = DataLoader(self.train_dataset,
                                           shuffle=True,
                                           batch_size=self.batch_size,
                                           collate_fn=self.collate_fn,
                                           num_workers=self.num_workers)
        return self.train_dataloader

    def test_dataloader(self):
        self.test_dataloader = DataLoader(self.test_dataset,
                                          shuffle=False,
                                          batch_size=self.batch_size,
                                          collate_fn=self.collate_fn,
                                          num_workers=self.num_workers)
        return self.test_dataloader

    def val_dataloader(self):
        return self.test_dataloader()


class OgmiosDataset(Dataset):
    def __init__(self,
                 dataset_folder: DatasetFolder,
                 preprocess_config: PreprocessingConfig,
                 split: Literal["train", "val"],
                 sort: bool = False,
                 drop_last: bool = False):
        self.dataset_folder = dataset_folder
        self.split = split
        self.preprocess_config = preprocess_config
        self.sort = sort
        self.drop_last = drop_last

        # loading split ids
        with open(self.dataset_folder.preprocessed_folder / f"{split}.txt", "r") as f:
            self.split_idx = set(f.read().split("\n"))

        self.files_idx, self.phonemes, self.raw_texts = zip(*self.load_metadata())
        # building a {phone -> index} mapping to convert phonemes to a sequence of numbers
        self.phonemes_mapping = {ph : i for i, ph in enumerate(dataset_folder.phonemes)}

    def __len__(self):
        return len(self.files_idx)

    def phonemes_to_sequence(self, phonemes: list[str]) -> np.ndarray[np.int32]:
        return np.array([self.phonemes_mapping[p] for p in phonemes])

    def __getitem__(self, idx):
        basename = self.files_idx[idx]
        raw_text = self.raw_texts[idx]
        phonemes_seq = self.phonemes_to_sequence(self.phonemes[idx])
        mel = np.load(self.dataset_folder.mels_folder / f"{basename}.npy")
        pitch = np.load(self.dataset_folder.pitches_folder / f"{basename}.npy")
        energy = np.load(self.dataset_folder.energies_folder / f"{basename}.npy")
        duration = np.load(self.dataset_folder.durations_folder / f"{basename}.npy")

        x = {"phoneme": phonemes_seq,
             "text": raw_text,
             "pitch": pitch,
             "energy": energy,
             "duration": duration}

        y = {"mel": mel}

        return x, y

    def load_metadata(self):
        with open(self.dataset_folder.preprocessed_folder / "metadata.csv", "r") as f:
            csv_reader = csv.reader(f, delimiter="\t")
            for row in csv_reader:
                if row[0] not in self.split_idx:
                    continue

                phonemes = row[1].split("|")
                if len(phonemes) > self.preprocess_config.text.max_length:
                    continue

                yield row[0], phonemes, row[2]
