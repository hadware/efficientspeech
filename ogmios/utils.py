'''
EfficientSpeech: An On-Device Text to Speech Model
https://ieeexplore.ieee.org/abstract/document/10094639
Rowel Atienza
Apache 2.0 License
2023
'''

import argparse
import json
import logging
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from scipy.io import wavfile

logger = logging.getLogger("ogmios")


def save_figure_to_numpy(fig, spectrogram=False):
    # save it to a numpy array.
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    if spectrogram:
        return data
    data = np.transpose(data, (2, 0, 1))
    return data


def plot_spectrogram_to_numpy(spectrogram):
    fig, ax = plt.subplots(figsize=(12, 3))
    im = ax.imshow(spectrogram, aspect="auto", origin="lower",
                   interpolation='none')
    plt.colorbar(im, ax=ax)
    plt.xlabel("Frames")
    plt.ylabel("Channels")
    plt.tight_layout()

    fig.canvas.draw()
    data = save_figure_to_numpy(fig, True)
    plt.close()
    return data


def get_mask_from_lengths(lengths, max_len=None):
    batch_size = lengths.shape[0]
    if max_len is None:
        max_len = torch.max(lengths).item()

    ids = torch.arange(0, max_len).unsqueeze(0).expand(batch_size, -1)  # .to(device)
    mask = ids >= lengths.unsqueeze(1).expand(-1, max_len)

    return mask


def expand(values, durations):
    out = list()
    for value, d in zip(values, durations):
        out += [value] * max(0, int(d))
    return np.array(out)


def synth_one_sample(mel_pred,
                     mel_len_pred,
                     vocoder,
                     preprocess_config,
                     wav_path="output",
                     verbose=False):
    if wav_path is not None:
        os.makedirs(wav_path, exist_ok=True)
    sampling_rate = preprocess_config["preprocessing"]["audio"]["sampling_rate"]
    lengths = mel_len_pred * \
              preprocess_config["preprocessing"]["stft"]["hop_length"]
    mel_pred = mel_pred.transpose(1, 2)
    wav_prediction = vocoder_infer(mel_pred,
                                   vocoder,
                                   preprocess_config,
                                   lengths=lengths,
                                   verbose=verbose)

    if wav_path is not None:
        wavfile.write(os.path.join(wav_path, "prediction.wav"),
                      sampling_rate, wav_prediction[0])

    return wav_prediction[0]


def vocoder_infer(mels, vocoder, preprocess_config, lengths=None, verbose=False):
    if verbose:
        start_time = time.time()

    with torch.no_grad():
        wavs = vocoder(mels).squeeze(1)

    if verbose:
        elapsed_time = time.time() - start_time
        print("(HiFiGAN) Synthesizing WAV time: {:.4f}s".format(elapsed_time))

    if verbose:
        start_time = time.time()

    wavs = (
            wavs.cpu().numpy()
            * preprocess_config["preprocessing"]["audio"]["max_wav_value"]
    ).astype("int16")
    wavs = [wav for wav in wavs]

    for i in range(len(mels)):
        if lengths is not None:
            wavs[i] = wavs[i][: lengths[i]]

    if verbose:
        elapsed_time = time.time() - start_time
        print("(Postprocess) WAV time: {:.4f}s".format(elapsed_time))

    return wavs


def pad_1D(inputs, PAD=0):
    def pad_data(x, length, PAD):
        x_padded = np.pad(
            x, (0, length - x.shape[0]), mode="constant", constant_values=PAD
        )
        return x_padded

    max_len = max((len(x) for x in inputs))
    padded = np.stack([pad_data(x, max_len, PAD) for x in inputs])

    return padded


def pad_2D(inputs, maxlen=None):
    def pad(x, max_len):
        PAD = 0
        if np.shape(x)[0] > max_len:
            raise ValueError("not max_len")

        s = np.shape(x)[1]
        x_padded = np.pad(
            x, (0, max_len - np.shape(x)[0]), mode="constant", constant_values=PAD
        )
        return x_padded[:, :s]

    if maxlen:
        output = np.stack([pad(x, maxlen) for x in inputs])
    else:
        max_len = max(np.shape(x)[0] for x in inputs)
        output = np.stack([pad(x, max_len) for x in inputs])

    return output


def pad(input_ele, mel_max_length=None):
    if mel_max_length:
        max_len = mel_max_length
    else:
        max_len = max([input_ele[i].size(0) for i in range(len(input_ele))])

    out_list = list()
    for i, batch in enumerate(input_ele):
        if len(batch.shape) == 1:
            one_batch_padded = F.pad(
                batch, (0, max_len - batch.size(0)), "constant", 0.0
            )
        elif len(batch.shape) == 2:
            one_batch_padded = F.pad(
                batch, (0, 0, 0, max_len - batch.size(0)), "constant", 0.0
            )
        out_list.append(one_batch_padded)
    out_padded = torch.stack(out_list)
    return out_padded


def sequence_mask(length, max_length=None):
    if max_length is None:
        max_length = length.max()
    x = torch.arange(max_length, dtype=length.dtype, device=length.device)
    return x.unsqueeze(0) < length.unsqueeze(1)


def get_args():
    parser = argparse.ArgumentParser()

    choices = ['cpu', 'gpu']
    parser.add_argument("--accelerator", type=str, default=choices[1], choices=choices)

    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--iter", type=int, default=1)
    parser.add_argument("--threads", type=int, default=24)

    # choices = ["bf16-mixed", "16-mixed", 16, 32, 64]
    parser.add_argument("--precision", default=16)

    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_epochs", type=int, default=5000)
    parser.add_argument("--warmup_epochs", type=int, default=50)

    parser.add_argument("--preprocess-config",
                        default="config/LJSpeech/preprocess.yaml",
                        type=str,
                        help="Path to preprocess.yaml", )
    parser.add_argument('--weight-decay',
                        type=float,
                        default=1e-5,
                        metavar='N',
                        help='Optimizer weight decay')
    parser.add_argument('--lr',
                        type=float,
                        default=1e-3,
                        metavar='N',
                        help='Learning rate for AdamW.')

    parser.add_argument('--val-every-epoch',
                        type=int,
                        default=5,
                        metavar='N',
                        help='Check val every N epochs')

    parser.add_argument('--batch-size',
                        type=int,
                        default=128,
                        metavar='N',
                        help='Batch size')

    parser.add_argument('--depth',
                        type=int,
                        default=2,
                        help='Encoder depth. Default for tiny, small & base.')
    parser.add_argument('--block-depth',
                        type=int,
                        default=2,
                        help='Decoder block depth. Default for tiny & small. Base:  3')
    parser.add_argument('--n-blocks',
                        type=int,
                        default=2,
                        help='Decoder blocks. Default for tiny. Small & base: 3.')
    parser.add_argument('--reduction',
                        type=int,
                        default=4,
                        help='Embed dim reduction factor. Default for tiny. Small: 2. Base: 1.')
    parser.add_argument('--head',
                        type=int,
                        default=1,
                        help='Number of transformer encoder head. Default for tiny & small. Base: 2.')
    parser.add_argument('--embed-dim',
                        type=int,
                        default=128,
                        help='Embedding or feature dim. To be reduced by --reduction.')
    parser.add_argument('--kernel-size',
                        type=int,
                        default=3,
                        help='Conv1d kernel size (Encoder). Default for tiny & small. Base is 5.')
    parser.add_argument('--decoder-kernel-size',
                        type=int,
                        default=5,
                        help='Conv1d kernel size (Decoder). Default for tiny, small & base: 5.')
    parser.add_argument('--expansion',
                        type=int,
                        default=1,
                        help='MixFFN expansion. Default for tiny & small. Base: 2.')

    parser.add_argument("--hifigan-checkpoint",
                        default="hifigan/LJ_V2/generator_v2",
                        type=str,
                        help="HiFiGAN checkpoint", )


    choices = ['cpu', 'cuda']
    parser.add_argument("--infer-device",
                        default='cpu',
                        choices=choices,
                        type=str,
                        help="Inference device", )

    parser.add_argument("--checkpoint",
                        default=None,
                        type=str,
                        help="Path to model checkpoint file", )
    parser.add_argument("--wav-path",
                        default="outputs",
                        type=str,
                        help="Folder to wav file to be generated during inference", )
    parser.add_argument("--wav-filename",
                        default="efficient_speech",
                        type=str,
                        help="wav filename to be generated", )

    parser.add_argument("--text",
                        type=str,
                        default=None,
                        help="Raw text to synthesize, for single-sentence mode only", )

    parser.add_argument('--verbose',
                        action='store_true',
                        help='Print out debug information')

    parser.add_argument('--onnx',
                        type=str,
                        default=None,
                        help='Convert to onnx model')
    parser.add_argument('--onnx-insize',
                        type=int,
                        default=128,
                        help='Max input size for the onnx model')
    parser.add_argument('--onnx-opset',
                        type=int,
                        default=14,
                        help='Opset version of onnx model (9<opset<15)')

    parser.add_argument('--jit',
                        type=str,
                        default=None,
                        help='Convert to jit model')
    # use jit modules 
    parser.add_argument('--to-torchscript',
                        action='store_true',
                        help='Convert model to torchscript')

    # if benchmark is True 
    parser.add_argument('--benchmark',
                        action='store_true',
                        help='Run benchmark')

    parser.add_argument('--compile',
                        action='store_true',
                        help='Train using the compiled model')
    parser.add_argument('--play',
                        action='store_true',
                        help='Playback the generated audio. Do not save it to disk.')

    args = parser.parse_args()

    args.num_workers *= args.devices

    return args
