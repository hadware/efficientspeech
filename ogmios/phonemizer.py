from pathlib import Path
from typing import Callable

import numpy
import numpy as np
from numpy.typing import NDArray
from onnx import load, ModelProto
from phonemizer.backend import EspeakBackend
from phonemizer.separator import Separator

from ogmios.dataset.commons import PreprocessingConfig, DatasetFolder

PhonemizeFn = Callable[[str], list[str]]

class Phonemizer:
    # TODO: DOCUMENT
    def __init__(self, lang: str,
                 phones: list[str],
                 phonemize_fn: PhonemizeFn | None = None,
                 preprocessing_fn: Callable[[str], str] | None = None,):

        if phonemize_fn is None:
            self.backend = EspeakBackend(lang)
            self.separator = Separator(phone='|', word=";")
            self.phonemize = self.phonemizer_fn
        else:
            self.phonemize = phonemize_fn

        self.preprocessing_fn = preprocessing_fn
        self.phones_mapping = {phone: i for i, phone in enumerate(phones)}


    @classmethod
    def from_config(cls, config: PreprocessingConfig, dataset_folder: DatasetFolder):
        return cls(config.text.language, dataset_folder.phonemes)

    @classmethod
    def from_onnx_model(cls,
                        onnx_model: str | Path | ModelProto,
                        phonemizer_fn: PhonemizeFn | None = None):
        if isinstance(onnx_model, str | Path):
            onnx_model_path = Path(onnx_model)
            onnx_model = load(onnx_model_path)

        metadata = {metadata.key: metadata.value for metadata in onnx_model.metadata_props}
        if "lang" not in metadata:
            raise ValueError("lang metadata is missing from the onnx model")
        if "phones" not in metadata:
            raise ValueError("phones metadata is missing from the onnx model")

        lang = metadata["lang"]
        phones_list = metadata["phones"].split(",")

        return cls(lang=lang, phones=phones_list, phonemize_fn=phonemizer_fn)

    def phonemizer_fn(self, text: str):
        # TODO: check phonemization
        return self.backend.phonemize([text], separator=self.separator, strip=True)[0]


    def __call__(self, text: str) -> NDArray[np.int32]:
        if self.preprocessing_fn is not None:
            text = self.preprocessing_fn(text)

        phones = self.phonemize(text)
        if missing_phones := set(phones) - set(self.phones_mapping):
            raise ValueError(f"Phones {missing_phones} are not in the phones list")

        return numpy.array([self.phones_mapping[phone] for phone in phones], dtype=np.int32)
