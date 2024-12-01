'''
EfficientSpeech: An On-Device Text to Speech Model
https://ieeexplore.ieee.org/abstract/document/10094639
Rowel Atienza, 2023
Apache 2.0 License

Usage:
    python3 convert.py --checkpoint tiny_eng_266k.ckpt --onnx tiny_eng_266k.onnx
'''

import torch
import yaml

import hifigan
from layers import Phoneme2Mel
from model import EfficientSpeech, get_hifigan
from utils.tools import get_args

class OgmiosOnnx(torch.nn.Module):
    def __init__(self,
                 phon2mel: Phoneme2Mel,
                 vocoder: hifigan.Generator):
        super().__init__()
        self.phon2mel = phon2mel
        self.vocoder = vocoder

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(0)
        mel, lengths = self.phon2mel.synthesize_one(x)
        mel = mel.transpose(1, 2)
        wav = self.vocoder(mel).squeeze(1)
        wav = wav.squeeze()
        return wav

# main routine
if __name__ == "__main__":
    args = get_args()
    preprocess_config = yaml.load(
        open(args.preprocess_config, "r"), Loader=yaml.FullLoader)

    model = EfficientSpeech.load_from_checkpoint(args.checkpoint, map_location=torch.device('cpu'))
    model = model.to(args.infer_device)
    hifigan = get_hifigan(checkpoint="hifigan/LJ_V2/generator_v2",
                          infer_device=args.infer_device, verbose=args.verbose)

    ogmios_model = OgmiosOnnx(phon2mel=model.phoneme2mel,vocoder=hifigan)

    phoneme = torch.randint(low=70, high=146, size=(args.onnx_insize,)).int().to(args.infer_device)
    print("Input shape: ", phoneme.shape)
    sample_input = [phoneme]
    print("Converting to ONNX ...", args.onnx)

    torch.onnx.export(ogmios_model,
                      f="ogmios_onnx.onnx",
                      args=tuple(sample_input),
                      opset_version=args.onnx_opset,
                      do_constant_folding=True,
                      input_names=["x"],
                      output_names=["wav"],
                      dynamic_axes={
                          "x": {0: "phoneme"},
                          "wav": {0: "frames"}
                      })
