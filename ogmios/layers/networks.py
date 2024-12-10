'''
EfficientSpeech: An On-Device Text to Speech Model
https://ieeexplore.ieee.org/abstract/document/10094639
Rowel Atienza
Apache 2.0 License
2023
'''
from typing import Optional

import torch
from torch import nn

from .blocks import MixFFN, SelfAttention


class Encoder(nn.Module):
    """ Phoneme Encoder """

    def __init__(self,
                 alphabet_dim: int,
                 depth: int = 2,
                 embed_dim: int = 128,
                 kernel_size: int = 3,
                 expansion: int = 1,
                 reduction: int = 4,
                 head: int = 1):
        super().__init__()

        small_embed_dim = embed_dim // reduction
        dim_ins = [small_embed_dim * (2 ** i) for i in range(depth - 1)]
        dim_ins.insert(0, embed_dim)
        self.dim_outs = [small_embed_dim * 2 ** i for i in range(depth)]
        heads = [head * (i + 1) for i in range(depth)]
        kernels = [kernel_size - (2 if i > 0 else 0) for i in range(depth)]
        paddings = [k // 2 for k in kernels]
        strides = [2 for _ in range(depth - 1)]
        strides.insert(0, 1)

        self.embed = nn.Embedding(alphabet_dim + 1, embed_dim, padding_idx=0)

        self.attn_blocks = nn.ModuleList([])
        for dim_in, dim_out, head, kernel, stride, padding in zip(dim_ins, self.dim_outs, \
                                                                  heads, kernels, strides, paddings):
            self.attn_blocks.append(
                nn.ModuleList([
                    # depthwise separable-like convolution
                    nn.Conv1d(dim_in, dim_in, kernel_size=kernel, stride=stride, \
                              padding=padding, bias=False),
                    nn.Conv1d(dim_in, dim_out, kernel_size=1, bias=False),
                    SelfAttention(dim_out, num_heads=head),
                    MixFFN(dim_out, expansion),
                    nn.LayerNorm(dim_out),
                    nn.LayerNorm(dim_out),
                ]))

    def get_feature_dims(self):
        return self.dim_outs

    def forward(self, phoneme: torch.Tensor,
                mask: Optional[torch.Tensor] = None) \
            -> tuple[torch.Tensor, torch.Tensor]:
        features = []
        x = self.embed(phoneme)
        # merge, attn and mixffn operates on n or seqlen dim
        # b = batch, n = sequence len, c = channel (1st layer is embedding)
        # (b, n, c)
        n = x.shape[-2]
        decoder_mask = None
        pool = 1

        for merge3x3, merge1x1, attn, mixffn, norm1, norm2 in self.attn_blocks:
            # after each encoder block, merge features
            x = x.permute(0, 2, 1)
            x = merge3x3(x)
            x = merge1x1(x)
            x = x.permute(0, 2, 1)
            # self-attention with skip connect
            if mask is not None:
                pool = int(torch.round(torch.tensor([n / x.shape[-2]], requires_grad=False)).item())

            y, attn_mask = attn(x, mask=mask, pool=pool)
            x = norm1(y + x)
            if attn_mask is not None:
                x = x.masked_fill(attn_mask, 0)
                if decoder_mask is None:
                    decoder_mask = attn_mask

            # Mix-FFN with skip connect
            x = norm2(mixffn(x) + x)

            if attn_mask is not None:
                x = x.masked_fill(attn_mask, 0)
            # mlp decoder operates on c or channel dim
            features.append(x)

        return features, decoder_mask


class AcousticDecoder(nn.Module):
    """ Pitch, Duration, Energy Predictor """

    def __init__(self, dim: int,
                 pitch_stats=None,
                 energy_stats=None,
                 n_mel_channels: int = 80,
                 duration=False):
        super().__init__()

        self.n_mel_channels = n_mel_channels

        self.conv1 = nn.Sequential(nn.Conv1d(dim, dim, kernel_size=3, padding=1), nn.ReLU())
        self.norm1 = nn.LayerNorm(dim)
        self.conv2 = nn.Sequential(nn.Conv1d(dim, dim, kernel_size=3, padding=1), nn.ReLU())
        self.norm2 = nn.LayerNorm(dim)
        self.linear = nn.Linear(dim, 1)
        self.duration = duration

        # TODO: switch to gaussian linspace, using pitch stats
        if pitch_stats is not None:
            pitch_min, pitch_max = pitch_stats
            self.pitch_bins = nn.Parameter(torch.linspace(pitch_min, pitch_max, dim - 1),
                                           requires_grad=False, )
            self.pitch_embedding = nn.Embedding(dim, dim)
        else:
            self.pitch_bins = None
            self.pitch_embedding = None

        if energy_stats is not None:
            energy_min, energy_max = energy_stats
            self.energy_bins = nn.Parameter(torch.linspace(energy_min, energy_max, dim - 1),
                                            requires_grad=False, )
            self.energy_embedding = nn.Embedding(dim, dim)
        else:
            self.energy_bins = None
            self.energy_embedding = None

    def get_pitch_embedding(self, pred, target, mask, control=1.):
        if target is not None:
            embedding = self.pitch_embedding(torch.bucketize(target, self.pitch_bins))
        else:
            # pred = pred * control
            embedding = self.pitch_embedding(torch.bucketize(pred, self.pitch_bins))
        return embedding

    def get_energy_embedding(self, pred, target, mask, control=1.):
        if target is not None:
            embedding = self.energy_embedding(torch.bucketize(target, self.energy_bins))
        else:
            # pred = pred * control
            embedding = self.energy_embedding(torch.bucketize(pred, self.energy_bins))
        return embedding

    def get_embedding(self, pred, target, mask, control=1.):
        if self.pitch_embedding is not None:
            return self.get_pitch_embedding(pred, target, mask, control)
        elif self.energy_embedding is not None:
            return self.get_energy_embedding(pred, target, mask, control)
        return None

    def forward(self, fused_features):
        y = fused_features.permute(0, 2, 1)
        y = self.conv1(y)
        y = y.permute(0, 2, 1)
        y = nn.ReLU()(self.norm1(y))
        y = y.permute(0, 2, 1)
        y = self.conv2(y)
        y = y.permute(0, 2, 1)
        features = self.norm2(y)
        y = self.linear(y)
        if self.duration:
            y = nn.ReLU()(y)
            return y, features

        return y


class Fuse(nn.Module):
    """ Fuse Attn Features"""

    def __init__(self, dims, kernel_size=3):
        super().__init__()

        assert (len(dims) > 0)

        dim = dims[0]
        self.mlps = nn.ModuleList([])
        for d in dims:
            upsample = d // dim
            self.mlps.append(
                nn.ModuleList([
                    nn.Linear(d, dim),
                    nn.ConvTranspose1d(dim, dim, kernel_size=kernel_size, stride=upsample) \
                        if upsample > 1 else nn.Identity()
                ]))

        self.fuse = nn.Linear(dim * len(dims), dim)

    def forward(self, features, mask=None):

        fused_features = []

        # each feature from encoder block
        for feature, mlps in zip(features, self.mlps):
            mlp, upsample = mlps
            # linear projection to uniform channel size (eg 256)
            x = mlp(feature)
            # upsample operates on the n or seqlen dim
            x = x.permute(0, 2, 1)
            # upsample sequence len downsampled by encoder blocks
            x = upsample(x)

            if mask is not None:
                x = x[:, :, :mask.shape[1]]
            elif len(fused_features) > 0:
                x = x[:, :, :fused_features[0].shape[-1]]

            fused_features.append(x)
            # print(x.size())

        # cat on the feature dim
        fused_features = torch.cat(fused_features, dim=-2)
        fused_features = fused_features.permute(0, 2, 1)

        fused_features = self.fuse(fused_features)
        if mask is not None:
            fused_features = fused_features.masked_fill(mask, 0)

        return fused_features


class GaussianUpsampling(torch.nn.Module):
    """
    Gaussian upsampling with fixed temperature as in:
    https://arxiv.org/abs/2010.04301

    Implem from : https://github.com/oortur/text-to-speech/blob/c1086d1cc71a37f5974bdd7eaaaec4db965156c7/models.py#L106

    """

    def __init__(
            self,
            log_sigma=0,
    ):
        super().__init__()
        log_sigma = log_sigma or torch.randn(1).item()
        self.log_sigma = nn.Parameter(torch.tensor(log_sigma), requires_grad=True)
        self.mask_fill_value = -1e10

    def forward(self, emb: torch.Tensor, durations: torch.Tensor):
        """
        emb ~ (bs, seq_len, emb_dim)
        durations ~ (bs, seq_len)
        """
        bs, seq_len = emb.shape[:2]
        device = emb.device

        # (bs, seq_len)
        centers = torch.cumsum(durations, dim=1) - durations * 0.5

        # T - max num of ticks per batch / Mel max len
        T = torch.sum(durations, dim=1).int().max()

        # (bs, T, 1)
        t = torch.arange(T).repeat(bs, 1).unsqueeze(2).to(device)
        normal = torch.distributions.Normal(loc=centers.unsqueeze(1), scale=torch.exp(self.log_sigma).view(1, 1, 1))
        # (bs, T, seq_len)
        prob = normal.log_prob(t + 0.5)
        mask = (durations == 0).unsqueeze(1)
        prob = prob.masked_fill(mask, self.mask_fill_value)
        w = nn.Softmax(dim=2)(prob)

        # (bs, T, emb_dim)
        x = torch.bmm(w, emb)

        # (bs, T)
        out_mask = t < durations.sum(dim=1).view(bs, 1, 1)
        out_mask = out_mask.repeat(1, 1, emb.shape[-1])

        return x, out_mask


class MelDecoder(nn.Module):
    """ Mel Spectrogram Decoder """

    def __init__(self, dim, kernel_size=5, n_mel_channels=80,
                 n_blocks=2, block_depth=2, ):
        super().__init__()

        self.n_mel_channels = n_mel_channels
        dim_x2 = min(4 * dim, 256)
        dim_x4 = 4 * dim
        padding = kernel_size // 2

        self.proj = nn.Sequential(
            nn.Linear(dim_x4, dim_x2), nn.Tanh(), nn.LayerNorm(dim_x2), )

        self.blocks = nn.ModuleList([])
        for _ in range(n_blocks):
            conv = nn.ModuleList([])
            for _ in range(block_depth):
                conv.append(nn.ModuleList([nn.Sequential( \
                    nn.Conv1d(dim_x2, dim_x2, groups=dim_x2, kernel_size=kernel_size, padding=padding), \
                    nn.Conv1d(dim_x2, dim_x2, kernel_size=1), \
                    nn.Tanh(), ),
                    nn.LayerNorm(dim_x2)]))

            self.blocks.append(nn.ModuleList([conv, nn.LayerNorm(dim_x2)]))

        self.mel_linear = nn.Linear(dim_x2, self.n_mel_channels)

    def forward(self, features):
        skip = self.proj(features)
        for convs, skip_norm in self.blocks:
            x = skip
            for conv, norm in convs:
                x = conv(x.permute(0, 2, 1))
                x = norm(x.permute(0, 2, 1))

            skip = skip_norm(x + skip)

        # resize channel to mel length (eg 80)
        mel = self.mel_linear(skip)

        return mel


class PhonemeEncoder(nn.Module):
    """ Encodes phonemes to acoustic features """

    def __init__(self,
                 alphabet_dim: int,
                 pitch_stats=None,
                 energy_stats=None,
                 depth=2,
                 reduction=4,
                 head=1,
                 embed_dim=128,
                 kernel_size=3,
                 expansion=1):
        super().__init__()

        self.encoder = Encoder(alphabet_dim=alphabet_dim,
                               depth=depth,
                               reduction=reduction,
                               head=head,
                               embed_dim=embed_dim,
                               kernel_size=kernel_size,
                               expansion=expansion)

        dim = embed_dim // reduction
        self.fuse = Fuse(self.encoder.get_feature_dims(), kernel_size=kernel_size)
        self.feature_upsampler = GaussianUpsampling()
        self.pitch_decoder = AcousticDecoder(dim, pitch_stats=pitch_stats)
        self.energy_decoder = AcousticDecoder(dim, energy_stats=energy_stats)
        self.duration_decoder = AcousticDecoder(dim, duration=True)

    def forward(self, x, train=False):
        phoneme = x["phoneme"]
        phoneme_mask = x["phoneme_mask"] if phoneme.shape[0] > 1 else None

        pitch_target = x["pitch"] if train else None
        energy_target = x["energy"] if train else None
        duration_target = x["duration"] if train else None

        features, mask = self.encoder(phoneme, mask=phoneme_mask)
        fused_features = self.fuse(features, mask=mask)

        pitch_pred = self.pitch_decoder(fused_features)
        pitch_features = self.pitch_decoder.get_embedding(pitch_pred, pitch_target, mask)
        pitch_features = pitch_features.squeeze()
        if mask is not None:
            pitch_features = pitch_features.masked_fill(mask, 0)
        elif pitch_features.dim() != 3:
            pitch_features = pitch_features.unsqueeze(0)

        energy_pred = self.energy_decoder(fused_features)
        energy_features = self.energy_decoder.get_embedding(energy_pred, energy_target, mask)
        energy_features = energy_features.squeeze()

        if mask is not None:
            energy_features = energy_features.masked_fill(mask, 0)
        elif energy_features.dim() != 3:
            energy_features = energy_features.unsqueeze(0)

        duration_pred, duration_features = self.duration_decoder(fused_features)
        if mask is not None:
            duration_features = duration_features.masked_fill(mask, 0)

        fused_features = torch.cat([fused_features,
                                    pitch_features,
                                    energy_features,
                                    duration_features], dim=-1)

        if duration_target is None:
            durations = torch.round(duration_pred).squeeze()
        else:
            durations = duration_target

        if phoneme_mask is not None:
            durations = durations.masked_fill(phoneme_mask, 0).clamp(min=0)
        else:
            durations = durations.unsqueeze(0)

        features, mel_mask = self.feature_upsampler(
            emb=fused_features,
            durations=durations,
        )
        mel_len_pred = durations.sum(dim=1)

        y = {"pitch": pitch_pred,
             "energy": energy_pred,
             "duration": duration_pred,
             "mel_len": mel_len_pred,
             "features": features,
             "mel_mask": mel_mask, }

        return y

    @torch.inference_mode()
    def infer_one(self, phoneme: torch.Tensor):
        features, _ = self.encoder(phoneme, mask=None)
        fused_features = self.fuse(features, mask=None)

        pitch_pred = self.pitch_decoder(fused_features)
        pitch_features = self.pitch_decoder.get_embedding(pitch_pred, None, None)
        pitch_features = pitch_features.squeeze()
        pitch_features = pitch_features.unsqueeze(0)

        energy_pred = self.energy_decoder(fused_features)
        energy_features = self.energy_decoder.get_embedding(energy_pred, None, None)
        energy_features = energy_features.squeeze()
        energy_features = energy_features.unsqueeze(0)

        duration_pred, duration_features = self.duration_decoder(fused_features)

        fused_features = torch.cat([fused_features,
                                    pitch_features,
                                    energy_features,
                                    duration_features], dim=-1)

        durations = torch.round(duration_pred).squeeze().unsqueeze(0)

        features, _ = self.feature_upsampler(emb=fused_features,
                                             durations=durations)

        y = {"pitch": pitch_pred,
             "energy": energy_pred,
             "duration": duration_pred,
             "features": features,
             "masks": None}

        return y


class Phoneme2Mel(nn.Module):
    """ From Phoneme Sequence to Mel Spectrogram """

    def __init__(self,
                 encoder: PhonemeEncoder,
                 decoder: MelDecoder):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x, train=False):
        pred = self.encoder(x, train=train)
        mel = self.decoder(pred["features"])

        mel_mask = pred["mel_mask"]
        # masking mels based on mask computed by upsampler
        if mel_mask is not None and mel.size(0) > 1:
            mel_mask = mel_mask[:, :, :mel.shape[-1]]
            mel = mel.masked_fill(~mel_mask, 0)

        pred["mel"] = mel

        if train:
            return pred

        return mel, pred["mel_len"], pred["duration"]

    @torch.inference_mode()
    def synthesize_one(self, x: torch.Tensor):
        # inserting "fake" batch dim for inference
        pred = self.encoder.infer_one(x)
        mel = self.decoder(pred["features"])
        return mel, pred["duration"]
