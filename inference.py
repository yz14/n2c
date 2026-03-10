"""
2.5D sliding window inference for NCCT→CTA translation.

Strategy:
  - Input 3D volume: (D, H, W)
  - Slide a window of size 3C along D, feeding (3C, H, W) to the model
  - Model predicts (C, H, W), but we only keep the middle C slices
    (from index C to 2C) of the prediction
  - Stride = C (non-overlapping middle slices)
  - Boundary handling: pad D at start/end so every output slice is covered
  - Final output: (D, H, W) stitched from middle predictions

Usage (G only):
    python inference.py --config outputs/config.yaml --checkpoint outputs/checkpoint_best.pt

Usage (G + G2, same checkpoint):
    python inference.py --config outputs/config.yaml --checkpoint outputs/checkpoint_best.pt --checkpoint_g2 outputs/checkpoint_best.pt

Usage (G + G2, separate checkpoints):
    python inference.py --config outputs/config.yaml --checkpoint outputs/checkpoint_G.pt --checkpoint_g2 outputs/checkpoint_G2.pt
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from config import Config
from models.unet import UNet
from models.refine_net import RefineNet
from data.dataset import normalize_hu, denormalize_hu
import SimpleITK as sitk

logger = logging.getLogger(__name__)


def load_G(cfg: Config, checkpoint_path: str, device: torch.device, use_ema: bool = True) -> torch.nn.Module:
    """Load trained UNet (G) from checkpoint.

    注意：当训练时启用了 G2（refine），checkpoint 中的 ema_params 跟踪的是 G2
    而非 G，此时应传入 use_ema=False 以加载 G 的原始权重 model_state_dict。
    """
    mcfg = cfg.model
    model = UNet(
        image_size=cfg.data.image_size,
        in_channels=mcfg.in_channels,
        model_channels=mcfg.model_channels,
        out_channels=mcfg.out_channels,
        num_res_blocks=mcfg.num_res_blocks,
        attention_resolutions=tuple(mcfg.attention_resolutions),
        dropout=mcfg.dropout,
        channel_mult=tuple(mcfg.channel_mult),
        conv_resample=mcfg.conv_resample,
        use_checkpoint=mcfg.use_checkpoint,
        use_fp16=mcfg.use_fp16,
        num_heads=mcfg.num_heads,
        num_head_channels=mcfg.num_head_channels,
        resblock_updown=mcfg.resblock_updown,
        residual_output=mcfg.residual_output,
    )

    state = torch.load(checkpoint_path, map_location=device)

    if use_ema and "ema_params" in state:
        # 只有在仅训练 G（无 G2）时，ema_params 才对应 G 的参数
        ema_params = state["ema_params"]
        for p, ema_p in zip(model.parameters(), ema_params):
            p.data.copy_(ema_p.data)
        logger.info("[G] Loaded EMA parameters")
    else:
        model.load_state_dict(state["model_state_dict"])
        logger.info("[G] Loaded model_state_dict")

    model = model.to(device)
    model.eval()
    return model


# 保持向后兼容的别名
load_model = load_G


def load_G2(
    cfg: Config,
    checkpoint_path: str,
    device: torch.device,
    use_ema: bool = True,
) -> torch.nn.Module:
    """Load trained RefineNet (G2) from checkpoint.

    当训练时启用了 G2，checkpoint 中包含：
      - refine_net_state_dict：G2 的原始权重
      - ema_params：G2 的 EMA 权重（训练时 EMA 跟踪的是 G2）

    Args:
        cfg:             Config 对象，读取 refine.hidden_dim / refine.num_blocks
        checkpoint_path: 包含 refine_net_state_dict 的 checkpoint 路径
        device:          目标设备
        use_ema:         True → 尝试加载 ema_params 作为 G2 权重（推荐）
                         False → 加载 refine_net_state_dict（训练末态权重）

    Returns:
        eval 模式下的 RefineNet
    """
    rcfg = cfg.refine
    refine_net = RefineNet(
        in_channels=cfg.data.num_slices,
        hidden_dim=rcfg.hidden_dim,
        num_blocks=rcfg.num_blocks,
    )

    state = torch.load(checkpoint_path, map_location=device)

    if "refine_net_state_dict" not in state:
        raise KeyError(
            f"checkpoint '{checkpoint_path}' 中没有 'refine_net_state_dict'。"
            "请确认此 checkpoint 是在 refine.enabled=true 时保存的。"
        )

    if use_ema and "ema_params" in state:
        # 训练时 use_refine=True 时，ema_params 跟踪的是 G2 的参数
        ema_params = state["ema_params"]
        if len(ema_params) == len(list(refine_net.parameters())):
            for p, ema_p in zip(refine_net.parameters(), ema_params):
                p.data.copy_(ema_p.data)
            logger.info("[G2] Loaded EMA parameters")
        else:
            # ema_params 数量与 G2 不匹配，说明 EMA 跟踪的是 G，回退到 state_dict
            logger.warning(
                "[G2] ema_params 参数数量与 G2 不匹配（可能是 G 的 EMA），"
                "回退到 refine_net_state_dict"
            )
            refine_net.load_state_dict(state["refine_net_state_dict"])
            logger.info("[G2] Loaded refine_net_state_dict")
    else:
        refine_net.load_state_dict(state["refine_net_state_dict"])
        logger.info("[G2] Loaded refine_net_state_dict")

    refine_net = refine_net.to(device)
    refine_net.eval()
    return refine_net


def resize_slice(tensor: torch.Tensor, size: int, mode: str = "bilinear") -> torch.Tensor:
    """Resize (D, H, W) tensor to (D, size, size)."""
    x = tensor.unsqueeze(0)  # (1, D, H, W)
    if mode == "nearest":
        x = F.interpolate(x, size=(size, size), mode="nearest")
    else:
        x = F.interpolate(x, size=(size, size), mode=mode, align_corners=False)
    return x.squeeze(0)


def resize_back(tensor: torch.Tensor, H: int, W: int, mode: str = "bilinear") -> torch.Tensor:
    """Resize (D, H_model, W_model) tensor back to (D, H, W)."""
    x = tensor.unsqueeze(0)  # (1, D, H_model, W_model)
    if mode == "nearest":
        x = F.interpolate(x, size=(H, W), mode="nearest")
    else:
        x = F.interpolate(x, size=(H, W), mode=mode, align_corners=False)
    return x.squeeze(0)


@torch.no_grad()
def predict_volume(
    model: torch.nn.Module,
    ncct_volume: np.ndarray,
    cfg: Config,
    device: torch.device,
    refine_net: Optional[torch.nn.Module] = None,
) -> np.ndarray:
    """
    Predict full 3D CTA volume from NCCT using 2.5D sliding window.

    Pipeline (与训练时一致):
      G only:   pred = G(ncct)
      G + G2:   g_pred = G(ncct)
                intermediate = ncct * |g_pred|   # "intermediate" 输入模式
                pred = G2(intermediate)

    Sliding window strategy:
      1. Pad D so that D_padded is divisible by C.
      2. Add extra C slices at front and back (replicate boundary).
         This creates an "extended" volume of size D_ext = D_padded + 2C.
      3. Slide windows of size 3C with stride C along the extended volume.
         For window i starting at i*C:
           - Context:    extended[i*C : i*C + 3C]
           - Model input: extended[i*C + C : i*C + 2C]  (middle C slices)
           - Prediction covers positions [i*C + C, i*C + 2C) in extended space,
             which maps to [i*C, (i+1)*C) in padded space (offset by extra_front=C).
      4. Stitch all C-slice predictions to cover [0, D_padded).
      5. Crop back to D_orig and resize to original H, W.

    Args:
        model:       trained UNet G (eval mode)
        ncct_volume: (D, H, W) float32 array in HU values
        cfg:         Config object
        device:      torch device
        refine_net:  optional RefineNet G2 (eval mode); if provided, applies G→G2 pipeline

    Returns:
        pred_volume: (D, H, W) float32 array in HU values
    """
    C = cfg.data.num_slices
    context = 3 * C
    image_size = cfg.data.image_size
    hu_min, hu_max = cfg.data.hu_min, cfg.data.hu_max

    D_orig, H_orig, W_orig = ncct_volume.shape

    # Normalize HU → [-1, 1]
    ncct_norm = normalize_hu(ncct_volume.copy(), hu_min, hu_max)

    # Step 1: Pad D so D_padded is a multiple of C
    D_padded = int(np.ceil(D_orig / C)) * C
    pad_total = D_padded - D_orig
    pad_front = pad_total // 2
    pad_back = pad_total - pad_front

    if pad_total > 0:
        ncct_padded = np.pad(
            ncct_norm,
            ((pad_front, pad_back), (0, 0), (0, 0)),
            mode="edge",
        )
    else:
        ncct_padded = ncct_norm

    # Convert to tensor and resize H, W
    ncct_tensor = torch.from_numpy(ncct_padded).float()  # (D_padded, H, W)
    if H_orig != image_size or W_orig != image_size:
        ncct_tensor = resize_slice(ncct_tensor, image_size)

    # Step 2: Add extra C padding at front and back for boundary windows
    D_pad = ncct_tensor.shape[0]
    ncct_extended = F.pad(
        ncct_tensor.unsqueeze(0).unsqueeze(0),  # (1, 1, D_pad, H, W)
        (0, 0, 0, 0, C, C),
        mode="replicate",
    ).squeeze(0).squeeze(0)  # (D_pad + 2C, H, W)

    # Step 3: Sliding window inference
    D_ext = ncct_extended.shape[0]
    n_windows = (D_ext - context) // C + 1
    pred_slices = []
    
    print(ncct_extended.min(), ncct_extended.max())

    use_g2 = refine_net is not None
    for i in tqdm(range(n_windows), desc="Inference (G+G2)" if use_g2 else "Inference (G)", leave=False):
        start = i * C
        window = ncct_extended[start:start + context]           # (3C, H_m, W_m)
        model_input = window[C:2*C].unsqueeze(0).to(device)    # (1, C, H_m, W_m)
        g_pred = model(model_input)                             # (1, C, H_m, W_m)
        if use_g2:
            # 与训练时 _sample_g2_input "intermediate" 模式一致:
            # intermediate = ncct * |G(ncct)|
            intermediate = model_input * g_pred.abs()           # (1, C, H_m, W_m)
            pred = refine_net(intermediate)                     # (1, C, H_m, W_m)
        else:
            pred = g_pred
        pred_slices.append(pred.squeeze(0).cpu())

    # Step 4: Stitch — each prediction covers C slices in padded space
    pred_full = torch.cat(pred_slices, dim=0)  # (n_windows * C, H_m, W_m)
    pred_padded = pred_full[:D_pad]

    # Step 5: Remove D padding and resize back to original spatial dims
    if pad_total > 0:
        pred_orig = pred_padded[pad_front:pad_front + D_orig]
    else:
        pred_orig = pred_padded[:D_orig]

    if H_orig != image_size or W_orig != image_size:
        pred_orig = resize_back(pred_orig, H_orig, W_orig)
    
    mid = ((ncct_norm * 0.5 + 0.5) * (
        ((pred_orig * 0.5 + 0.5))
        ).numpy()) * 2 - 1

    # Denormalize back to HU
    pred_hu = denormalize_hu(pred_orig.numpy(), hu_min, hu_max)
    mid = denormalize_hu(mid, hu_min, hu_max)
    return pred_hu, mid

# def contrast(x, n=2, threshold=0.6):
#     mask = (x <= threshold) * 1.0
#     x0 = x * mask
#     x0 = threshold * (x0 / threshold) ** n
#     x1 = 1 - (1 - threshold) * ((1 - x) / (1 - threshold)) ** n
#     return x0 + x1 * (1 - mask)


def contrast_stretch(img, low=0.01, high=0.99):
    p_low = torch.quantile(img, low)
    p_high = torch.quantile(img, high)

    img = (img - p_low) / (p_high - p_low + 1e-8)
    img = torch.clamp(img, 0, 1)
    return img

def sigmoid_contrast(img, k=10):
    return torch.sigmoid(k * img)

def unsharp_mask(img, amount=1.0):

    kernel = torch.ones((1,1,3,3,3), device=img.device) / 27
    blur = F.conv3d(img[None,None], kernel, padding=1)

    sharp = img + amount * (img - blur)
    return torch.clamp(sharp,0,1)[0,0]

def run_inference(
    cfg: Config,
    checkpoint_path: str,
    output_dir: str,
    use_ema: bool = True,
    checkpoint_g2_path: Optional[str] = None,
    use_ema_g2: bool = True,
):
    """Run inference on the test set.

    Args:
        cfg:               Config 对象
        checkpoint_path:   G 的 checkpoint 路径
        output_dir:        输出目录
        use_ema:           是否加载 G 的 EMA 权重（G 单独训练时推荐 True）
        checkpoint_g2_path: G2 的 checkpoint 路径；为 None 则只用 G 推理
        use_ema_g2:        是否加载 G2 的 EMA 权重（推荐 True）
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # 加载 G
    # 注意：若 checkpoint 是 G+G2 联合训练产物，ema_params 属于 G2，
    # 应用 use_ema=False 加载 G 的 model_state_dict。
    model = load_G(cfg, checkpoint_path, device, use_ema=use_ema)

    # 加载 G2（可选）
    refine_net = None
    if checkpoint_g2_path:
        refine_net = load_G2(cfg, checkpoint_g2_path, device, use_ema=use_ema_g2)
        logger.info(f"[G2] Loaded from: {checkpoint_g2_path}")

    # Load test file list
    split_dir = Path(cfg.data.split_dir)
    test_file = split_dir / "test.txt"
    if not test_file.exists():
        logger.error(f"Test split file not found: {test_file}")
        return

    with open(test_file, "r") as f:
        test_files = [line.strip() for line in f if line.strip()]
    logger.info(f"Found {len(test_files)} test volumes")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for filename in tqdm(test_files, desc="Test volumes"):
        filepath = Path(cfg.data.data_dir) / filename
        logger.info(f"Processing: {filename}")

        # Load volume
        npz = np.load(filepath, mmap_mode="r")
        ncct_volume = npz["ncct"][:].astype(np.float32)  # (D, H, W)

        # Predict
        pred_cta, mid = predict_volume(model, ncct_volume, cfg, device, refine_net=refine_net)
        # mid, _ = predict_volume(model, mid, cfg, device)

        # Save result
        stem = Path(filename).stem
        save_path = output_path / f"{stem}_pred.nii.gz"
        # np.savez_compressed(save_path, pred_cta=pred_cta)
        a = sitk.GetImageFromArray(pred_cta)
        sitk.WriteImage(a, save_path)
        save_path = output_path / f"{stem}_pred2.nii.gz"
        a = sitk.GetImageFromArray(mid)
        sitk.WriteImage(a, save_path)
        logger.info(f"  Saved: {save_path} shape={pred_cta.shape}")

    logger.info("Inference complete.")


def main():
    parser = argparse.ArgumentParser(description="2.5D sliding window inference")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to G checkpoint")
    parser.add_argument(
        "--checkpoint_g2", type=str, default=None,
        help="Path to G2 (RefineNet) checkpoint。"
             "可与 --checkpoint 相同（同一文件同时包含 G 和 G2 权重）。"
             "不指定则只用 G 推理。"
    )
    parser.add_argument("--output_dir", type=str, default="./inference_results", help="Output directory")
    parser.add_argument(
        "--no_ema", action="store_true",
        help="对 G 使用原始权重而非 EMA。"
             "若 checkpoint 是 G+G2 联合训练产物，G 的 EMA 不存在（EMA 属于 G2），"
             "此时程序会自动回退到 model_state_dict，也可显式加 --no_ema。"
    )
    parser.add_argument(
        "--no_ema_g2", action="store_true",
        help="对 G2 使用原始权重（refine_net_state_dict）而非 EMA。"
    )
    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    cfg = Config.load(args.config)
    cfg.sync_channels()

    run_inference(
        cfg=cfg,
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        use_ema=not args.no_ema,
        checkpoint_g2_path=args.checkpoint_g2,
        use_ema_g2=not args.no_ema_g2,
    )


if __name__ == "__main__":
    main()
