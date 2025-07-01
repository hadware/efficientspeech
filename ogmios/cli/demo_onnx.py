import onnx
import onnxruntime
import torch
import yaml

from ogmios.dataset.commons import PreprocessingConfig
from ogmios.hifigan import HifiganOnnxModel
from ogmios.phonemizer import OgmiosPhonemizer
from ogmios.utils import get_player

ONNX_CPU_PROVIDERS = [
    "CPUExecutionProvider",
]

class DemoOnnxCommandParser(Tap):
    onnx: Path  # Onnx model checkpoint that is to be used for the demo
    hifigan: Path  # Path to hifigan onnx model
    text: str  # Text to phonemize
    verbose: bool = False
    play: bool = False


if __name__ == '__main__':
    args = DemoOnnxCommandParser.parse_args()
    config = yaml.load( open(args.config, "r"), Loader=yaml.FullLoader)
    preprocessing_config = PreprocessingConfig(**config["preprocessing"])

    # preparing ONNX model
    onnx_model = onnx.load(args.checkpoint)
    onnx.checker.check_model(args.checkpoint, full_check=True)
    onnx_session = onnxruntime.InferenceSession(args.checkpoint, providers=ONNX_CPU_PROVIDERS)

    # loading hifigan onnx model
    hifigan_onnx_model = HifiganOnnxModel(args.hifigan)

    # preparing text input
    phonemizer = OgmiosPhonemizer.from_onnx_model(onnx_model)
    phonemes = phonemizer(args.text)
    inputs = {onnx_session.get_inputs()[0].name: phonemes}
    outputs = onnx_session.run(None, inputs)

    # vocoding
    hifigan = HifiganOnnxModel(args.hifigan)
    mel = torch.tensor(outputs[0]).cpu().numpy()
    wav = hifigan.synth(mel).squeeze(1)
    wav = wav.squeeze()

    if args.play:
        player = get_player(config.sampling_rate)

        player.play(wav)
        player.wait()
