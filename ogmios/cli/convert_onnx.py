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

from ogmios.layers import Phoneme2Mel
from ogmios.model import EfficientSpeech, get_hifigan
from ogmios.utils import get_args

class OgmiosOnnx(torch.nn.Module):
    def __init__(self,
                 phon2mel: Phoneme2Mel):
        super().__init__()
        self.phon2mel = phon2mel

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(0)
        mel, lengths = self.phon2mel.synthesize_one(x)
        mel = mel.transpose(1, 2)
        return mel

# main routine
if __name__ == "__main__":
    args = get_args()
    preprocess_config = yaml.load(
        open(args.config, "r"), Loader=yaml.FullLoader)

    model = EfficientSpeech.load_from_checkpoint(args.checkpoint, map_location=torch.device('cpu'))
    model = model.to(args.infer_device)
    hifigan = get_hifigan(checkpoint="hifigan/LJ_V2/generator_v2",
                          infer_device=args.infer_device, verbose=args.verbose)

    ogmios_model = OgmiosOnnx(phon2mel=model.phoneme2mel)

    phoneme = torch.randint(low=70, high=146, size=(69,)).int().to(args.infer_device)
    print("Input shape: ", phoneme.shape)
    sample_input = [phoneme]
    print("Converting to ONNX ...", args.onnx)

    torch.onnx.export(ogmios_model,
                      f="ogmios_onnx.onnx",
                      args=tuple(sample_input),
                      opset_version=args.onnx_opset,
                      do_constant_folding=True,
                      input_names=["x"],
                      output_names=["mel"],
                      dynamic_axes={
                          "x": {0: "phoneme"},
                          "mel": {0: "frames"}
                      })
