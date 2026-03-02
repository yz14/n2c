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

        Handles arbitrary channel count by replicating to 3 channels.
        """
        # [-1, 1] → [0, 1]
        x = (x + 1.0) * 0.5

        # Handle channel dimension: replicate single/multi-channel to 3ch
        C = x.shape[1]
        if C == 1:
            x = x.repeat(1, 3, 1, 1)
        elif C != 3:
            # For multi-slice input (e.g. C=3 is fine, C=9 take middle 3)
            mid = C // 2
            if C >= 3:
                x = x[:, mid - 1:mid + 2, :, :]
            else:
                x = x[:, :1, :, :].repeat(1, 3, 1, 1)

        if self.resize:
            x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)

        # ImageNet normalization
        x = (x - self.vgg_mean) / self.vgg_std
        return x

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

        Args:
            pred:   (N, C, H, W) predicted image in [-1, 1]
            target: (N, C, H, W) target image in [-1, 1]

        Returns:
            Scalar perceptual loss (weighted sum of L1 distances in VGG space).
        """
        pred_vgg = self._preprocess(pred)
        target_vgg = self._preprocess(target)

        pred_feats = self._extract_features(pred_vgg)
        with torch.no_grad():
            target_feats = self._extract_features(target_vgg)

        loss = 0.0
        layer_indices = sorted(self.layer_weights.keys())
        for feat_idx, layer_idx in enumerate(layer_indices):
            weight = self.layer_weights[layer_idx]
            # Normalize by number of elements for scale-invariance
            n_elements = pred_feats[feat_idx].numel() / pred_feats[feat_idx].shape[0]
            loss = loss + weight * F.l1_loss(
                pred_feats[feat_idx], target_feats[feat_idx]
            )
        return loss

    def train(self, mode=True):
        """Override to keep VGG always in eval mode."""
        super().train(mode)
        self.features.eval()
        return self
