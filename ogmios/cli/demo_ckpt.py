import time
from pathlib import Path
from typing import Literal, TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
import torch
import yaml
from tap import Tap

from ogmios.dataset.commons import PreprocessingConfig
from ogmios.hifigan import HifiganOnnxModel
from ogmios.trainer import EfficientSpeech, get_hifigan
from ..utils import logger


def get_player(sampling_rate: int):
    import sounddevice as sd

    sd.default.reset()
    sd.default.samplerate = sampling_rate
    sd.default.channels = 1
    sd.default.dtype = 'int16'
    sd.default.device = None
    sd.default.latency = 'low'
    return sd


def phonemize(text: str, config: PreprocessingConfig) -> NDArray:
    pass # TODO

def synth(phonemes: NDArray[np.int32],
          config: PreprocessingConfig,
          model: EfficientSpeech,
          hifigan: HifiganOnnxModel):

    start_time = time.time()
    with torch.no_grad():
        phoneme = torch.from_numpy(phonemes).int().to(model.device)
        mel, lengths = model.phoneme2mel.synthesize_one(phoneme)
        mel = mel.transpose(1, 2).cpu().numpy()
        wav = hifigan.synth(mel).squeeze(1)

    elapsed_time = time.time() - start_time
    sampling_rate = config.sampling_rate
    wav_duration = wav.shape[0] / sampling_rate
    real_time_factor = wav_duration / elapsed_time
    
    logger.info(f"Synthesis time: {elapsed_time:.2f} sec")
    logger.info(f"Voice length: {wav_duration:.2f} sec")
    logger.info(f"Real time factor: {real_time_factor:.2f}")

    return wav, phoneme, wav_duration, real_time_factor

class DemoCheckPointCommandParser(Tap):
    config: Path  # Path to processing config file (yaml)
    checkpoint: Path  # Model checkpoint that is to be used for the demo
    hifigan: Path  # Path to hifigan onnx model
    text: str  # Text to phonemize
    verbose: bool = False
    play: bool = False
    iter: int = 50
    infer_device: Literal['gpu', 'cpu'] = 'gpu'  # Device for which to convert the model

if __name__ == "__main__":
    args = DemoCheckPointCommandParser().parse_args()
    preprocess_config = yaml.load(open(args.config, "r"), Loader=yaml.FullLoader)
    preprocessing_config = PreprocessingConfig(**preprocess_config["preprocessing"])
    hifigan_onnx_model = HifiganOnnxModel(args.hifigan)


    model = EfficientSpeech.load_from_checkpoint(checkpoint=args.checkpoint,
                                                 map_location=torch.device('cpu'))

    model = model.to(args.infer_device)
    model.eval()

    if args.play:
        sound_player = get_player(preprocess_config.sampling_rate)

    phonemes = phonemize(args.text, preprocessing_config)
    rtf = []
    warmup = 10
    for i in range(args.iter):
        if args.infer_device == "cuda":
            torch.cuda.synchronize()
        wav, _, _, rtf_i = synth(phonemes, preprocess_config, model, hifigan_onnx_model)
        if i > warmup:
            rtf.append(rtf_i)
        if args.infer_device == "cuda":
            torch.cuda.synchronize()

        if args.play:
            sound_player.play(wav)
            sound_player.wait()

    if len(rtf) > 0:
        mean_rtf = np.mean(rtf)
        # print with 2 decimal places
        print("Average RTF: {:.2f}".format(mean_rtf))
