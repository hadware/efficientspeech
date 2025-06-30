from pathlib import Path
from typing import Literal, Optional

import torch
import yaml
from tap import Tap

from ogmios.layers import Phoneme2Mel
from ogmios.trainer import EfficientSpeech, get_hifigan


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


class ConvertOnnxCommandParser(Tap):
    config: Path  # Path to processing config file (yaml)
    checkpoint: Path  # Model checkpoint that is to be converted to onnx
    verbose: bool = False

    infer_device: Literal['gpu', 'cpu'] = 'cpu'  # Device for which to convert the model
    output_path: Optional[Path] = None


# main routine
if __name__ == "__main__":
    args = ConvertOnnxCommandParser.parse_args()
    preprocess_config = yaml.load(open(args.config, "r"), Loader=yaml.FullLoader)

    model = EfficientSpeech.load_from_checkpoint(args.checkpoint, map_location=torch.device(args.infer_device))
    model = model.to(args.infer_device)

    ogmios_model = OgmiosOnnx(phon2mel=model.phoneme2mel)

    phoneme = torch.randint(low=70, high=146, size=(69,)).int().to(args.infer_device)
    print("Input shape: ", phoneme.shape)
    sample_input = [phoneme]
    print("Converting to ONNX ...", args.onnx)

    if args.output_path is None:
        output_path = Path("ogmios_onnx.onnx")

    torch.onnx.export(ogmios_model,
                      f=output_path,
                      args=tuple(sample_input),
                      opset_version=args.onnx_opset,
                      input_names=["x"],
                      output_names=["mel"],
                      dynamic_axes={
                          "x": {0: "phoneme"},
                          "mel": {0: "frames"}
                      })
