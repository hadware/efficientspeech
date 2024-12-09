'''
EfficientSpeech: An On-Device Text to Speech Model
https://ieeexplore.ieee.org/abstract/document/10094639
Rowel Atienza
Apache 2.0 License
2023

Usage:
    Torch:
    python3 demo.py --checkpoint tiny_eng_266k.ckpt --infer-device cuda  --text "In additive color mixing, which is used for displays such as computer screens and televisions, the primary colors are red, green, and blue."  --wav-filename color.wav

    ONNX:
    python3 demo.py --checkpoint tiny_eng_266k.onnx --infer-device cuda  --text "In additive color mixing, which is used for displays such as computer screens and televisions, the primary colors are red, green, and blue."  --wav-filename color.wav
    
Additional dependencies for GUI:
    pip3 install pysimplegui
    pip3 install sounddevice 
'''

import time

import numpy as np
import torch
import validators
import yaml

from ogmios.model import EfficientSpeech, get_hifigan
from ogmios.synthesize import get_lexicon_and_g2p, text2phoneme
from ogmios.tools import get_args


def tts(lexicon, g2p, preprocess_config, model: EfficientSpeech, hifigan, args, verbose=False):
    text = args.text.strip()
    text = text.replace('-', ' ')
    phoneme = np.array([text2phoneme(lexicon, g2p, text, preprocess_config, verbose=verbose)],
                       dtype=np.int32)
    start_time = time.time()
    with torch.no_grad():
        phoneme = torch.from_numpy(phoneme).int().to(args.infer_device)
        mel, lengths = model.phoneme2mel.synthesize_one(phoneme)
        mel = mel.transpose(1, 2)
        wav = hifigan(mel).squeeze(1)
        wav = wav.squeeze().cpu().numpy()

    elapsed_time = time.time() - start_time
    message = f"Synthesis time: {elapsed_time:.2f} sec"
    sampling_rate = preprocess_config["preprocessing"]["audio"]["sampling_rate"]
    wav_len = wav.shape[0] / sampling_rate
    message += f"\nVoice length: {wav_len:.2f} sec"
    real_time_factor = wav_len / elapsed_time
    message += f"\nReal time factor: {real_time_factor:.2f}"
    message += f"\nNote:\tFor benchmarking, load the model 1st, do a warmup run for 100x, then run the benchmark for 1000 iterations."
    message += f"\n\tGet the mean of 1000 runs. Use --iter N to run N iterations. eg N=100"

    print(message)
    return wav, message, phoneme, wav_len, real_time_factor


if __name__ == "__main__":
    args = get_args()
    preprocess_config = yaml.load(
        open(args.config, "r"), Loader=yaml.FullLoader)

    lexicon, g2p = get_lexicon_and_g2p(preprocess_config)
    sampling_rate = preprocess_config["preprocessing"]["audio"]["sampling_rate"]
    is_onnx = False

    if validators.url(args.checkpoint):
        checkpoint = args.checkpoint.rsplit('/', 1)[-1]
        torch.hub.download_url_to_file(args.checkpoint, checkpoint)
    else:
        checkpoint = args.checkpoint

    model = EfficientSpeech.load_from_checkpoint(checkpoint,
                                                 infer_device=args.infer_device,
                                                 map_location=torch.device('cpu'))

    model = model.to(args.infer_device)
    model.eval()

    hifigan = get_hifigan(checkpoint="hifigan/LJ_V2/generator_v2",
                          infer_device=args.infer_device, verbose=args.verbose)

    if args.play:
        import sounddevice as sd

        sd.default.reset()
        sd.default.samplerate = sampling_rate
        sd.default.channels = 1
        sd.default.dtype = 'int16'
        sd.default.device = None
        sd.default.latency = 'low'

    if args.text is not None:
        rtf = []
        warmup = 10
        for i in range(args.iter):
            if args.infer_device == "cuda":
                torch.cuda.synchronize()
            wav, _, _, _, rtf_i = tts(lexicon, g2p, preprocess_config, model, hifigan, args)
            if i > warmup:
                rtf.append(rtf_i)
            if args.infer_device == "cuda":
                torch.cuda.synchronize()

            if args.play:
                sd.play(wav)
                sd.wait()

        if len(rtf) > 0:
            mean_rtf = np.mean(rtf)
            # print with 2 decimal places
            print("Average RTF: {:.2f}".format(mean_rtf))
    else:
        print("Nothing to synthesize. Please provide a text file with --text")
