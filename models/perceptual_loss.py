"""
VGG-based Perceptual Loss for image synthesis quality.

Computes L1 distance between VGG feature maps of predicted and target images.
This encourages the generator to produce images with similar high-level
features (edges, textures, structures) rather than just pixel-level accuracy.

For single-channel or multi-channel medical images, input is replicated to
3 channels and normalized to ImageNet statistics before feeding to VGG.

Reference:
  Johnson et al., "Perceptual Losses for Real-Time Style Transfer and
  Super-Resolution", ECCV 2016.
"""

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class VGGPerceptualLoss(nn.Module):
    """
    VGG-19 perceptual loss using features from relu1_2, relu2_2, relu3_4, relu4_4.

    The VGG network is frozen (no gradients). Input images in [-1, 1] are
    rescaled and normalized to ImageNet statistics.

    Layer weights follow Johnson et al. defaults: equal weight per layer,
    normalized by number of elements.

    Args:
        layer_weights: optional dict mapping layer index to weight.
            Default uses {3: 1.0, 8: 1.0, 17: 1.0, 26: 1.0} corresponding
            to relu1_2, relu2_2, relu3_4, relu4_4.
        resize: if True, resize input to 224x224 for VGG (not recommended
            for high-res medical images; default False).
    """

    # VGG-16 feature extraction layers (after ReLU)
    # relu1_2=3, relu2_2=8, relu3_3=15, relu4_3=22
    DEFAULT_LAYERS = {3: 1.0, 8: 1.0, 15: 1.0, 22: 1.0}

    # ImageNet normalization constants
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]

    def __init__(
        self,
        layer_weights: dict = None,
        resize: bool = False,
    ):
        super().__init__()
        self.layer_weights = layer_weights or self.DEFAULT_LAYERS
        self.resize = resize

        # Load pretrained VGG-16 features (VGG16 is lighter and commonly cached)
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        max_layer = max(self.layer_weights.keys()) + 1
        self.features = nn.Sequential(*list(vgg.features.children())[:max_layer])

        # Freeze all VGG parameters
        for param in self.features.parameters():
            param.requires_grad = False
        self.features.eval()

        # Register normalization buffers
        self.register_buffer(
            "vgg_mean",
            torch.tensor(self.MEAN).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "vgg_std",
            torch.tensor(self.STD).view(1, 3, 1, 1),
        )

    def _preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Convert input from [-1, 1] to VGG-normalized [0, 1] RGB format.

        For multi-channel 2.5D data (C >= 6), returns a list of 3-channel
        groups covering the full depth range. For C <= 3, returns a single
        3-channel tensor.

        Returns:
            List of (N, 3, H, W) tensors, each VGG-normalized.
        """
        # [-1, 1] → [0, 1]
        x = (x + 1.0) * 0.5

        C = x.shape[1]
        groups = []

        if C == 1:
            groups.append(x.repeat(1, 3, 1, 1))
        elif C == 2:
            # Pad to 3: first, second, mean
            groups.append(torch.cat([x, x.mean(dim=1, keepdim=True)], dim=1))
        elif C == 3:
            groups.append(x)
        elif C >= 6:
            # Multi-slice: extract non-overlapping 3-channel groups
            # e.g. C=12 → [0:3], [3:6], [6:9], [9:12] = 4 groups
            n_groups = C // 3
            for i in range(n_groups):
                groups.append(x[:, i*3:(i+1)*3, :, :])
        else:
            # C in [4, 5]: take first 3 and last 3 (may overlap)
            groups.append(x[:, :3, :, :])
            groups.append(x[:, -3:, :, :])

        # Apply VGG normalization and optional resize to each group
        result = []
        for g in groups:
            if self.resize:
                g = F.interpolate(g, size=(224, 224), mode="bilinear", align_corners=False)
            g = (g - self.vgg_mean) / self.vgg_std
            result.append(g)
        return result

    def _extract_features(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Extract feature maps at specified VGG layers."""
        features = []
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i in self.layer_weights:
                features.append(x)
        return features

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute perceptual loss.

        For multi-channel 2.5D data (C >= 6), computes perceptual loss on
        multiple non-overlapping 3-slice groups and averages them. This
        provides full depth coverage instead of only using the middle 3 slices.

        Args:
            pred:   (N, C, H, W) predicted image in [-1, 1]
            target: (N, C, H, W) target image in [-1, 1]

        Returns:
            Scalar perceptual loss (weighted sum of L1 distances in VGG space).
        """
        pred_groups = self._preprocess(pred)
        target_groups = self._preprocess(target)

        assert len(pred_groups) == len(target_groups)
        layer_indices = sorted(self.layer_weights.keys())

        total_loss = 0.0
        for pred_vgg, target_vgg in zip(pred_groups, target_groups):
            pred_feats = self._extract_features(pred_vgg)
            with torch.no_grad():
                target_feats = self._extract_features(target_vgg)

            for feat_idx, layer_idx in enumerate(layer_indices):
                weight = self.layer_weights[layer_idx]
                total_loss = total_loss + weight * F.l1_loss(
                    pred_feats[feat_idx], target_feats[feat_idx]
                )

        # Average over groups
        return total_loss / len(pred_groups)

    def train(self, mode=True):
        """Override to keep VGG always in eval mode."""
        super().train(mode)
        self.features.eval()
        return self
