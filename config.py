"""
Centralized configuration management using dataclasses + YAML.
"""

import yaml
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Tuple, Optional


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
class TrainConfig:
    batch_size: int = 8
    lr: float = 1e-4
    weight_decay: float = 0.0
    num_epochs: int = 200
    save_interval: int = 10      # save checkpoint every N epochs
    log_interval: int = 50       # log every N steps
    val_interval: int = 1        # validate every N epochs
    l1_weight: float = 1.0
    ssim_weight: float = 1.0
    use_3d_ssim: bool = True     # use 3D SSIM loss (treats C as depth)
    ema_rate: float = 0.999
    lr_scheduler: str = "cosine" # "cosine" or "step" or "none"
    warmup_steps: int = 500
    output_dir: str = "./outputs"
    resume_checkpoint: str = ""
    seed: int = 42


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def save(self, path: str):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        d = asdict(self)
        # Convert tuples to lists for YAML compatibility
        self._tuples_to_lists(d)
        with open(path, "w") as f:
            yaml.dump(d, f, default_flow_style=False, sort_keys=False)

    @staticmethod
    def _tuples_to_lists(d):
        for k, v in d.items():
            if isinstance(v, tuple):
                d[k] = list(v)
            elif isinstance(v, dict):
                Config._tuples_to_lists(v)

    @classmethod
    def load(cls, path: str) -> "Config":
        with open(path, "r") as f:
            raw = yaml.safe_load(f)
        data_cfg = DataConfig(**raw.get("data", {}))
        model_raw = raw.get("model", {})
        # Convert lists back to tuples for tuple-typed fields
        for k in ("channel_mult", "attention_resolutions"):
            if k in model_raw and isinstance(model_raw[k], list):
                model_raw[k] = tuple(model_raw[k])
        model_cfg = ModelConfig(**model_raw)
        train_cfg = TrainConfig(**raw.get("train", {}))
        return cls(data=data_cfg, model=model_cfg, train=train_cfg)

    def sync_channels(self):
        """Ensure in_channels/out_channels match num_slices."""
        self.model.in_channels = self.data.num_slices
        self.model.out_channels = self.data.num_slices


if __name__ == "__main__":
    cfg = Config()
    cfg.save("configs/default.yaml")
    print("Default config saved to configs/default.yaml")
