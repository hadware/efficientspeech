'''
EfficientSpeech: An On-Device Text to Speech Model
https://ieeexplore.ieee.org/abstract/document/10094639
Rowel Atienza, 2023
Apache 2.0 License
'''

import os

import numpy as np
import torch
from lightning import LightningDataModule
from torch.utils.data import Dataset, DataLoader

from ogmios.dataset.commons import PreprocessingConfig, DatasetFolder
from ogmios.phonemizers.ljspeech import text_to_sequence
from ogmios.utils import get_mask_from_lengths
from ogmios.utils import pad_1D, pad_2D


class OgmiosDataModule(LightningDataModule):
    def __init__(self,
                 preprocess_config: PreprocessingConfig,
                 batch_size: int=64,
                 num_workers: int=4):
        super().__init__()
        self.preprocess_config = preprocess_config
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
             "duration": durations, }

        y = {"mel": mels, }

        return x, y

    def prepare_data(self):
        self.train_dataset = OgmiosDataset("train.txt",
                                           self.preprocess_config)
        self.test_dataset = OgmiosDataset("val.txt",
                                          self.preprocess_config)


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
                 sort: bool=False,
                 drop_last: bool=False):
        self.dataset_folder = dataset_folder
        self.preprocessed_path = preprocess_config["path"]["preprocessed_path"]
        self.cleaners = preprocess_config["preprocessing"]["text"]["text_cleaners"]
        # self.batch_size = batch_size
        self.max_text_length = preprocess_config["preprocessing"]["text"]["max_length"]
        self.basename, self.text, self.raw_text = self.process_meta(filename)
        self.sort = sort
        self.drop_last = drop_last

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        basename = self.basename[idx]
        raw_text = self.raw_text[idx]
        phoneme = np.array(text_to_sequence(self.text[idx], self.cleaners))
        mel_path = os.path.join(
            self.preprocessed_path,
            "mel",
            f"mel-{basename}.npy",
        )
        mel = np.load(mel_path)
        pitch_path = os.path.join(
            self.preprocessed_path,
            "pitch",
            f"pitch-{basename}.npy",
        )
        pitch = np.load(pitch_path)
        energy_path = os.path.join(
            self.preprocessed_path,
            "energy",
            f"energy-{basename}.npy",
        )
        energy = np.load(energy_path)
        duration_path = os.path.join(
            self.preprocessed_path,
            "duration",
            f"duration-{basename}.npy",
        )
        duration = np.load(duration_path)

        x = {"phoneme": phoneme,
             "text": raw_text,
             "pitch": pitch,
             "energy": energy,
             "duration": duration}

        y = {"mel": mel}

        return x, y

    def process_meta(self, filename):
        with open(
                os.path.join(self.preprocessed_path, filename), "r", encoding="utf-8"
        ) as f:
            name = []
            text = []
            raw_text = []
            for line in f.readlines():
                n, t, r = line.strip("\n").split("|")
                if len(r) > self.max_text_length:
                    continue
                name.append(n)
                text.append(t)
                raw_text.append(r)
            return name, text, raw_text
