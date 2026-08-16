'''DINOv2 ViT-B/14 with an OpenDroneKit-owned UPerNet-style decoder.

This is training architecture, not a shipped task model. The retained Meta checkpoint
initialises only the encoder. A run must train the decoder, evaluate the exact schema,
and export a full model before ``core.semantic_engine`` will accept it.
'''

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch
from torch import nn
import torch.nn.functional as F


def _groups(channels: int) -> int:
    for value in (32, 16, 8, 4, 2):
        if channels % value == 0:
            return value
    return 1


def _conv_block(
    in_channels: int,
    out_channels: int,
    kernel_size: int = 3,
    *,
    normalise: bool = True,
) -> nn.Sequential:
    padding = kernel_size // 2
    layers: list[nn.Module] = [
        nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=False)
    ]
    if normalise:
        layers.append(nn.GroupNorm(_groups(out_channels), out_channels))
    layers.append(nn.GELU())
    return nn.Sequential(*layers)


class PyramidPooling(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, scales: Sequence[int]) -> None:
        super().__init__()
        branch_channels = max(32, out_channels // len(scales))
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(scale),
                # A scale-1 branch is shaped (N,C,1,1); GroupNorm rejects a
                # single-value batch, so pooled branches intentionally omit it.
                _conv_block(in_channels, branch_channels, kernel_size=1, normalise=False),
            )
            for scale in scales
        ])
        self.fuse = _conv_block(in_channels + branch_channels * len(scales), out_channels)

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        size = feature.shape[-2:]
        pooled = [
            F.interpolate(branch(feature), size=size, mode='bilinear', align_corners=False)
            for branch in self.branches
        ]
        return self.fuse(torch.cat([feature, *pooled], dim=1))


class UPerNetDecoder(nn.Module):
    '''Pyramid pooling plus top-down feature fusion for four encoder stages.'''

    def __init__(
        self,
        in_channels: Sequence[int],
        num_classes: int,
        channels: int = 256,
        ppm_scales: Sequence[int] = (1, 2, 3, 6),
    ) -> None:
        super().__init__()
        if len(in_channels) != 4:
            raise ValueError('UPerNetDecoder expects exactly four feature maps.')
        if num_classes < 2:
            raise ValueError('Shared semantic models require at least two classes.')
        self.laterals = nn.ModuleList([
            _conv_block(int(value), channels, kernel_size=1) for value in in_channels
        ])
        self.ppm = PyramidPooling(int(in_channels[-1]), channels, ppm_scales)
        self.fpn_blocks = nn.ModuleList([_conv_block(channels, channels) for _ in in_channels])
        self.fusion = _conv_block(channels * len(in_channels), channels)
        self.classifier = nn.Conv2d(channels, num_classes, kernel_size=1)

    def forward(
        self,
        features: Sequence[torch.Tensor],
        output_size: tuple[int, int] | None = None,
    ) -> torch.Tensor:
        if len(features) != 4:
            raise ValueError('UPerNetDecoder received the wrong number of feature maps.')
        laterals = [layer(feature) for layer, feature in zip(self.laterals, features)]
        laterals[-1] = self.ppm(features[-1])
        for index in range(len(laterals) - 1, 0, -1):
            laterals[index - 1] = laterals[index - 1] + F.interpolate(
                laterals[index],
                size=laterals[index - 1].shape[-2:],
                mode='bilinear',
                align_corners=False,
            )
        target_size = laterals[0].shape[-2:]
        pyramid = [
            F.interpolate(block(value), size=target_size, mode='bilinear', align_corners=False)
            for block, value in zip(self.fpn_blocks, laterals)
        ]
        logits = self.classifier(self.fusion(torch.cat(pyramid, dim=1)))
        if output_size is not None and logits.shape[-2:] != output_size:
            logits = F.interpolate(logits, size=output_size, mode='bilinear', align_corners=False)
        return logits


class DinoV2UPerNet(nn.Module):
    '''Use four official DINOv2 intermediate token maps as a semantic pyramid.'''

    def __init__(
        self,
        encoder: nn.Module,
        num_classes: int,
        *,
        layer_indices: Sequence[int] = (2, 5, 8, 11),
        decoder_channels: int = 256,
        freeze_encoder: bool = False,
    ) -> None:
        super().__init__()
        if len(layer_indices) != 4:
            raise ValueError('Exactly four DINOv2 layer indices are required.')
        self.encoder = encoder
        self.layer_indices = tuple(int(value) for value in layer_indices)
        embed_dim = int(getattr(encoder, 'embed_dim', 768))
        self.decoder = UPerNetDecoder(
            [embed_dim] * 4,
            num_classes,
            channels=decoder_channels,
        )
        if freeze_encoder:
            for parameter in self.encoder.parameters():
                parameter.requires_grad = False

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.encoder.get_intermediate_layers(
            images,
            n=self.layer_indices,
            reshape=True,
            return_class_token=False,
            norm=True,
        )
        return self.decoder(features, output_size=images.shape[-2:])


def build_dinov2_vitb14_upernet(
    checkpoint_path: str | Path,
    num_classes: int,
    *,
    dinov2_source: str = 'facebookresearch/dinov2',
    source: str = 'github',
    freeze_encoder: bool = False,
) -> DinoV2UPerNet:
    '''Construct the official encoder and load the retained Apache-2.0 weights.

    For offline training, clone the official DINOv2 repository at a reviewed commit,
    pass that directory as ``dinov2_source`` and set ``source='local'``.
    '''
    checkpoint = Path(checkpoint_path)
    # The retained local checkpoint is the offline path and stays the default: an
    # air-gapped site should not need a network to train. But the weights are not in the
    # repository -- they are a 350 MB binary -- so a fresh clone on a hosted GPU has the
    # code and not the file, and refusing there would make the whole rented session a
    # FileNotFoundError. When the encoder is being fetched from GitHub anyway, letting
    # torch.hub bring its own pretrained weights is the same Apache-2.0 artefact from the
    # same publisher, so it is fetched rather than demanded.
    fetch_weights = not checkpoint.is_file()
    if fetch_weights and source != 'github':
        raise FileNotFoundError(
            f'DINOv2 checkpoint not found: {checkpoint}. With source={source!r} there is '
            'no way to fetch it -- point --dinov2-source at a local clone with the '
            'weights beside it, or use --source github to download them.'
        )
    # source and dinov2_source have to agree, and the config supplies a LOCAL PATH
    # because offline training is the default. Handing that path to torch.hub with
    # source='github' makes it try to read 'training/sources/dinov2' as owner/repo and
    # die on `too many values to unpack` -- an error that says nothing about the actual
    # mistake. Reconciled here rather than at every call site.
    reference = str(dinov2_source)
    if source == 'github' and reference.count('/') != 1:
        print(
            f'source=github but dinov2_source is {reference!r}, which is a path rather '
            "than owner/repo; using 'facebookresearch/dinov2'.", flush=True
        )
        reference = 'facebookresearch/dinov2'
    encoder = torch.hub.load(
        reference,
        'dinov2_vitb14',
        source=source,
        pretrained=fetch_weights,
        trust_repo=True,
    )
    if not fetch_weights:
        state = torch.load(checkpoint, map_location='cpu', weights_only=True)
        encoder.load_state_dict(state, strict=True)
    else:
        print(
            'DINOv2 encoder weights fetched from torch.hub (Apache-2.0). The local '
            f'checkpoint at {checkpoint} was absent.', flush=True
        )
    return DinoV2UPerNet(
        encoder,
        num_classes,
        freeze_encoder=freeze_encoder,
    )
