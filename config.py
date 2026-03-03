"""
Centralized configuration management using dataclasses + YAML.
"""

import yaml
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Tuple, Optional, List


@dataclass
class DataConfig:
    data_dir: str = "D:/codes/data/ncct_tiny"
    split_dir: str = "./splits"
    num_slices: int = 3          # C: number of output slices per sample
    image_size: int = 256        # H_out = W_out
    hu_min: float = -1024.0
    hu_max: float = 3071.0
    num_workers: int = 4
    # augmentation
    aug_prob: float = 0.5        # probability of applying augmentation
    max_angle: float = 15.0      # max rotation angle in degrees
    scale_range: float = 0.1     # scale factor range: [1-s, 1+s]
    translate_frac: float = 0.05 # translate fraction of image size
    noise_std: float = 0.02      # Gaussian noise std (in normalized space)
    brightness_range: float = 0.1
    contrast_range: float = 0.1
    lung_sample_bias: float = 0.0  # lung-aware sampling bias (0=uniform, 2.0=strong lung preference)
    ncct_degrade_prob: float = 0.0  # NCCT quality degradation probability (0=disabled, 0.3=recommended)


@dataclass
class ModelConfig:
    in_channels: int = 3         # same as num_slices
    out_channels: int = 3        # same as num_slices
    model_channels: int = 64     # base channel count
    channel_mult: Tuple[int, ...] = (1, 2, 4, 8)
    num_res_blocks: int = 2
    attention_resolutions: Tuple[int, ...] = (4, 8)  # downsample rates
    dropout: float = 0.0
    use_checkpoint: bool = False
    num_heads: int = 4
    num_head_channels: int = -1
    use_scale_shift_norm: bool = False
    conv_resample: bool = True
    resblock_updown: bool = False
    use_fp16: bool = False
    residual_output: bool = True  # output = input + model_prediction


@dataclass
class RegistrationConfig:
    enabled: bool = False            # ON/OFF switch for staged training
    nb_features: Tuple[int, ...] = (16, 32, 32, 32)  # encoder feature counts
    integration_steps: int = 7       # scaling & squaring steps (0 = direct displacement)
    smoothness_weight: float = 1.0   # weight for deformation smoothness loss
    smoothness_penalty: str = "l2"   # 'l1' or 'l2' for GradLoss
    lr: float = 1e-4                 # separate learning rate for registration net
    # --- R pre-training: train R with degraded+misaligned CTA before main training ---
    r_pretrain_epochs: int = 0       # pre-train R for N epochs with augmented CTA pairs (0=disabled)
    r_pretrain_max_angle: float = 5.0   # max rotation angle (degrees) for spatial misalignment
    r_pretrain_max_translate: float = 0.03  # max translation fraction for spatial misalignment
    r_pretrain_max_scale: float = 0.05  # max scale deviation for spatial misalignment
    r_pretrain_elastic_alpha: float = 6.0  # elastic deformation intensity (0=disabled)
    r_pretrain_elastic_points: int = 8  # control grid size for elastic deformation
    r_pretrain_degrade: bool = True  # also apply pixel degradation (blur/noise/downsample)


@dataclass
class DiscriminatorConfig:
    enabled: bool = True            # ON/OFF switch for staged training
    ndf: int = 64                    # base filters
    n_layers: int = 3                # conv layers per sub-discriminator (patchgan)
    num_D: int = 3                   # number of discriminator scales
    use_spectral_norm: bool = True   # apply spectral normalization
    gan_weight: float = 1.0          # weight for GAN loss (generator side)
    feat_match_weight: float = 10.0  # weight for feature matching loss
    lr: float = 2e-4                 # separate (typically higher) LR for discriminator
    lr_scheduler: str = "cosine"     # LR scheduler for D: "cosine", "step", "none"
    warmup_steps: int = 500          # warmup steps for D scheduler
    label_smoothing: float = 0.1     # one-sided label smoothing for D real targets (0=off)
    grad_clip_norm_D: float = 0.0    # D gradient clipping (0=disabled; SN already ensures stability)
    # --- New: anti-collapse settings ---
    d_cond_mode: str = "concat"      # D input: "concat"|"none"|"diff" (diff=image-ncct, best for NCCT→CTA)
    gan_loss_type: str = "lsgan"     # GAN loss: "lsgan" or "hinge"
    r1_gamma: float = 10.0           # R1 gradient penalty weight (0=disabled)
    r1_interval: int = 16            # lazy R1 every N D-steps (reduces overhead)
    # --- New: discriminator architecture ---
    disc_type: str = "patchgan"      # "patchgan" (original) or "resblock" (enhanced)
    n_blocks_resD: int = 4           # number of ResBlocks per sub-D (resblock type only)
    use_attention: bool = True       # self-attention in ResBlock D
    # --- New: training stability ---
    gan_warmup_epochs: int = 5       # linearly ramp GAN weight from 0 to gan_weight over N epochs
    d_warmup_steps: int = 0          # pre-train D for N steps before joint training (0=disabled)
    diffaugment_policy: str = ""     # DiffAugment policy: "color,translation,cutout" or "" (disabled)
    # --- Quality degradation negative samples for D ---
    d_quality_aug: str = ""          # degradation types: "blur,noise,downsample" or "" (disabled)
    d_quality_aug_prob: float = 0.5  # probability of adding degraded negative per D step
    d_quality_aug_weight: float = 0.5  # weight for quality assessment D loss term


@dataclass
class RefineConfig:
    """Configuration for the refinement network (G2)."""
    enabled: bool = False            # ON/OFF switch
    hidden_dim: int = 64             # hidden channel dimension in ResBlocks
    num_blocks: int = 6              # number of residual blocks
    lr: float = 1e-4                 # learning rate for G2
    lr_scheduler: str = "cosine"     # LR scheduler: "cosine", "step", "none"
    warmup_steps: int = 200          # warmup steps for G2 scheduler
    freeze_G: bool = True            # freeze G when G2 is enabled
    # --- G2 multi-input training modes (Phase 3) ---
    # Comma-separated list of input modes, randomly sampled each step:
    #   "synthesized"  — G2(G(ncct)): refine G's direct output
    #   "degraded"     — G2(degrade(cta)): refine quality-degraded real CTA
    #   "intermediate" — G2(ncct * |G(ncct)|): refine intermediate representation
    # Default "intermediate" matches the original single-mode behavior.
    g2_input_modes: str = "intermediate"


@dataclass
class TrainConfig:
    batch_size: int = 2
    lr: float = 1e-4
    weight_decay: float = 0.0
    num_epochs: int = 200
    save_interval: int = 10      # save checkpoint every N epochs
    log_interval: int = 50       # log every N steps
    val_interval: int = 1        # validate every N epochs
    l1_weight: float = 1.0
    ssim_weight: float = 1.0
    use_3d_ssim: bool = True     # use 3D SSIM loss (treats C as depth)
    lung_weight: float = 10.0    # loss weight multiplier for lung regions (used when schedule is empty)
    lung_weight_schedule: str = ""  # progressive lung weight: "epoch1:w1,epoch2:w2,..." e.g. "10:5,20:10,40:20,999:40"
    ema_rate: float = 0.999
    lr_scheduler: str = "cosine" # "cosine" or "step" or "none"
    warmup_steps: int = 500
    output_dir: str = "./outputs"
    resume_checkpoint: str = ""
    seed: int = 42
    # Pretrained model paths (load weights only, train from epoch 0)
    pretrained_G: str = ""       # path to pretrained generator weights
    pretrained_R: str = ""       # path to pretrained registration net weights
    pretrained_D: str = ""       # path to pretrained discriminator weights
    pretrained_G2: str = ""      # path to pretrained refinement net weights
    # Training tricks
    grad_clip_norm: float = 5.0  # max gradient norm for clipping (0 = disabled)
    grad_accumulation_steps: int = 1  # gradient accumulation steps (1 = no accumulation)
    skip_warmup: bool = False    # skip LR warmup (useful when loading pretrained weights)
    perceptual_weight: float = 0.0  # VGG perceptual loss weight (0=disabled, recommended: 0.1-1.0)
    freq_weight: float = 0.0        # FFT frequency loss weight (0=disabled, recommended: 0.1-0.5)
    # --- CTA degradation dual-task: G also learns degraded_CTA → real CTA ---
    g_cta_degrade_prob: float = 0.0  # prob of replacing NCCT input with degraded CTA (0=disabled)
    # --- Self-refinement: G(ncct * G(ncct).abs()) → CTA ---
    g_self_refine_prob: float = 0.0   # prob of self-refine step per training step (0=disabled)
    g_self_refine_weight: float = 0.5 # weight for self-refinement loss


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    registration: RegistrationConfig = field(default_factory=RegistrationConfig)
    discriminator: DiscriminatorConfig = field(default_factory=DiscriminatorConfig)
    refine: RefineConfig = field(default_factory=RefineConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def save(self, path: str):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        d = asdict(self)
        # Convert tuples to lists for YAML compatibility
        self._tuples_to_lists(d)
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(d, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

    @staticmethod
    def _tuples_to_lists(d):
        for k, v in d.items():
            if isinstance(v, tuple):
                d[k] = list(v)
            elif isinstance(v, dict):
                Config._tuples_to_lists(v)

    @classmethod
    def load(cls, path: str) -> "Config":
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        data_cfg = DataConfig(**raw.get("data", {}))
        model_raw = raw.get("model", {})
        # Convert lists back to tuples for tuple-typed fields
        for k in ("channel_mult", "attention_resolutions"):
            if k in model_raw and isinstance(model_raw[k], list):
                model_raw[k] = tuple(model_raw[k])
        model_cfg = ModelConfig(**model_raw)
        reg_raw = raw.get("registration", {})
        for k in ("nb_features",):
            if k in reg_raw and isinstance(reg_raw[k], list):
                reg_raw[k] = tuple(reg_raw[k])
        reg_cfg = RegistrationConfig(**reg_raw)
        disc_cfg = DiscriminatorConfig(**raw.get("discriminator", {}))
        refine_cfg = RefineConfig(**raw.get("refine", {}))
        train_cfg = TrainConfig(**raw.get("train", {}))
        return cls(
            data=data_cfg, model=model_cfg, registration=reg_cfg,
            discriminator=disc_cfg, refine=refine_cfg, train=train_cfg,
        )

    def sync_channels(self):
        """Ensure in_channels/out_channels match num_slices."""
        self.model.in_channels = self.data.num_slices
        self.model.out_channels = self.data.num_slices


if __name__ == "__main__":
    cfg = Config()
    cfg.save("configs/default.yaml")
    print("Default config saved to configs/default.yaml")
