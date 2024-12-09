import numpy as np
import onnx
import onnxruntime
import torch
import yaml

from ogmios.model import get_hifigan
from ogmios.synthesize import get_lexicon_and_g2p, text2phoneme
from ogmios.tools import get_args

ONNX_CPU_PROVIDERS = [
    "CPUExecutionProvider",
]

if __name__ == '__main__':
    args = get_args()
    preprocess_config = yaml.load(
        open(args.config, "r"), Loader=yaml.FullLoader)

    lexicon, g2p = get_lexicon_and_g2p(preprocess_config)
    sampling_rate = preprocess_config["preprocessing"]["audio"]["sampling_rate"]

    # preparing ONNX model
    onnx_model = onnx.load(args.checkpoint)
    onnx.checker.check_model(args.checkpoint, full_check=True)
    onnx_session = onnxruntime.InferenceSession(args.checkpoint, providers=ONNX_CPU_PROVIDERS)

    # preparing text input
    text = args.text.strip()
    text = text.replace('-', ' ')
    phoneme = np.array(text2phoneme(lexicon, g2p, text, preprocess_config, verbose=args.verbose),
                       dtype=np.int32)
    inputs = {onnx_session.get_inputs()[0].name: phoneme}
    outputs = onnx_session.run(None, inputs)

    # vocoding
    hifigan = get_hifigan(checkpoint="hifigan/LJ_V2/generator_v2",
                          infer_device=args.infer_device, verbose=args.verbose)
    mel = torch.tensor(outputs[0])
    wav = hifigan(mel).squeeze(1)
    wav = wav.squeeze().cpu().numpy()

    if args.play:
        import sounddevice as sd

        sd.default.reset()
        sd.default.samplerate = sampling_rate
        sd.default.channels = 1
        sd.default.dtype = 'int16'
        sd.default.device = None
        sd.default.latency = 'low'

        sd.play(wav)
        sd.wait()
