"""
Configuration for Latent Diffusion Model (LDM) for NCCT→CTA translation.

Architecture overview:
  Stage 1: AutoencoderKL compresses C-channel 2.5D slices to a low-dim latent.
  Stage 2: Conditional diffusion UNet denoises in latent space,
           conditioned on the NCCT latent (concatenation-based).

Default VAE: 256×256 input → 32×32 latent (f=8), z_channels=4.
Default UNet: operates on 32×32 latent, channel_mult=(1,2,4).

Data loading reuses the existing NCCTDataset and GPUAugmentor from Scheme 1.
"""

import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Tuple

import yaml

# Reuse Scheme 1 DataConfig to avoid duplication
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DataConfig  # noqa: E402


@dataclass
class VAEConfig:
    """AutoencoderKL configuration."""
    in_channels: int = 3           # input channels (= num_slices from 2.5D)
    out_channels: int = 3          # output channels (same as in_channels)
    z_channels: int = 4            # latent space channels
    embed_dim: int = 4             # embedding dim (typically = z_channels)
    ch: int = 128                  # base channel count
    ch_mult: Tuple[int, ...] = (1, 2, 4, 4)   # channel multipliers per level
    num_res_blocks: int = 2        # residual blocks per level
    attn_resolutions: Tuple[int, ...] = (32,)  # resolutions where attention is applied
    dropout: float = 0.0
    resolution: int = 256          # input spatial resolution
    double_z: bool = True          # output 2*z_channels for mean+logvar


@dataclass
class VAETrainConfig:
    """VAE training configuration (Stage 1)."""
    batch_size: int = 8
    num_epochs: int = 100
    lr: float = 4.5e-6             # learning rate (follows LDM paper)
    weight_decay: float = 0.0
    kl_weight: float = 1e-6        # KL divergence loss weight
    l1_weight: float = 1.0         # L1 reconstruction loss weight
    perceptual_weight: float = 0.0 # perceptual loss weight (0=disabled)
    lr_scheduler: str = "cosine"   # "cosine", "step", "none"
    warmup_steps: int = 500
    ema_rate: float = 0.9999
    grad_clip_norm: float = 1.0
    save_interval: int = 10
    log_interval: int = 50
    val_interval: int = 1
    output_dir: str = "./outputs_vae"
    resume_checkpoint: str = ""
    seed: int = 42


@dataclass
class DiffusionUNetConfig:
    """Conditional diffusion UNet configuration."""
    model_channels: int = 128      # base channel count
    channel_mult: Tuple[int, ...] = (1, 2, 4)   # channel multipliers per level
    num_res_blocks: int = 2        # residual blocks per level
    attention_resolutions: Tuple[int, ...] = (4, 2)  # downsample rates where attention is used
    dropout: float = 0.0
    num_heads: int = 4             # attention heads
    use_scale_shift_norm: bool = True  # FiLM conditioning on timestep


@dataclass
class SchedulerConfig:
    """Noise scheduler configuration."""
    num_train_timesteps: int = 1000
    beta_schedule: str = "linear"          # "linear" or "cosine"
    beta_start: float = 0.00085
    beta_end: float = 0.012
    prediction_type: str = "epsilon"       # "epsilon" (predict noise) or "v_prediction"
    # DDIM sampling
    num_inference_steps: int = 50          # DDIM steps for sampling


@dataclass
class DiffusionTrainConfig:
    """Diffusion UNet training configuration (Stage 2)."""
    batch_size: int = 8
    num_epochs: int = 200
    lr: float = 1e-4               # diffusion UNet learning rate
    weight_decay: float = 0.0
    lr_scheduler: str = "cosine"   # "cosine", "step", "none"
    warmup_steps: int = 500
    ema_rate: float = 0.9999
    grad_clip_norm: float = 1.0
    save_interval: int = 10
    log_interval: int = 50
    val_interval: int = 1
    output_dir: str = "./outputs_diffusion"
    resume_checkpoint: str = ""
    pretrained_vae: str = ""       # path to pretrained VAE checkpoint (required)
    seed: int = 42


@dataclass
class LDMConfig:
    """Top-level LDM configuration."""
    data: DataConfig = field(default_factory=DataConfig)
    vae: VAEConfig = field(default_factory=VAEConfig)
    vae_train: VAETrainConfig = field(default_factory=VAETrainConfig)
    unet: DiffusionUNetConfig = field(default_factory=DiffusionUNetConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    diffusion_train: DiffusionTrainConfig = field(default_factory=DiffusionTrainConfig)

    def save(self, path: str):
        """Save configuration to YAML."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        data = asdict(self)
        self._tuples_to_lists(data)
        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    @staticmethod
    def _tuples_to_lists(d):
        for k, v in d.items():
            if isinstance(v, tuple):
                d[k] = list(v)
            elif isinstance(v, dict):
                LDMConfig._tuples_to_lists(v)

    @classmethod
    def load(cls, path: str) -> "LDMConfig":
        """Load configuration from YAML."""
        with open(path) as f:
            raw = yaml.safe_load(f)

        def _lists_to_tuples(d, tuple_keys):
            for k in tuple_keys:
                if k in d and isinstance(d[k], list):
                    d[k] = tuple(d[k])

        data_raw = raw.get("data", {})
        vae_raw = raw.get("vae", {})
        _lists_to_tuples(vae_raw, ["ch_mult", "attn_resolutions"])
        unet_raw = raw.get("unet", {})
        _lists_to_tuples(unet_raw, ["channel_mult", "attention_resolutions"])

        return cls(
            data=DataConfig(**data_raw),
            vae=VAEConfig(**vae_raw),
            vae_train=VAETrainConfig(**raw.get("vae_train", {})),
            unet=DiffusionUNetConfig(**unet_raw),
            scheduler=SchedulerConfig(**raw.get("scheduler", {})),
            diffusion_train=DiffusionTrainConfig(**raw.get("diffusion_train", {})),
        )

    def sync_channels(self):
        """Sync VAE in/out channels with the data num_slices."""
        self.vae.in_channels = self.data.num_slices
        self.vae.out_channels = self.data.num_slices
        self.vae.resolution = self.data.image_size


if __name__ == "__main__":
    cfg = LDMConfig()
    out_path = "configs/ldm_default.yaml"
    cfg.save(out_path)
    print(f"Default LDM config saved to {out_path}")
