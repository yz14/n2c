# 核心原则

## 质量第一
- 宁可多花时间，也要保证代码质量
- 充分思考、分析后再动手实现
- 不要为了快速完成而牺牲代码质量

## 分步完成
- 如果当前对话无法完成所有功能，主动拆分为多轮对话
- 每轮只专注完成一个清晰的目标
- 不贪多，确保每一步都高质量完成

## 充分调研
- 如有需要，充分、彻底地搜索和调研
- 分析和掌握现有的高质量功能实现和算法
- 借鉴业界最佳实践，不要闭门造车

## 调试支持
- 如有需要，可以加入 debug/logging 函数辅助开发
- 通过日志输出帮助定位和解决问题
- 调试代码可在功能稳定后标注或移除

## 代码质量  
- 注意代码尽可能模块化设计，职责尽可能的分离，不要把所有代码写在一个文件里，不方便后续理解和维护  
- 注意代码的复用性，不要写重复的代码  

## 沟通规范
- **开始前**：说明你理解的任务目标和将遵守的规则
- **进行中**：如需拆分，明确告知本轮将完成什么
- **完成后**：总结本轮成果，说明后续计划（如有）  


测试环境为**py310**  


# 分析结论 (2026-03-01)

## Task 1: D 为什么无效？

### 根本原因：D 完全崩塌（D_real ≈ D_fake ≈ 0.749）

从 G+D 训练日志 (train.log) 中，D 对真假图像输出完全相同的值：
- Epoch 1: D_real=0.675, D_fake=0.675
- Epoch 2-13: D_real≈0.749, D_fake≈0.749（恒定）
- d_loss ≈ 0.945, gan_g ≈ 0.198（恒定）

**D 退化为常数函数**，对真假图像无区分能力，GAN 对 G 的梯度信号为零。

**崩塌原因**：fake CTA 与 real CTA 在归一化空间差异极小（l1_vs_cta ≈ 0.13-0.15），
谱归一化 + 高梯度范数（gnorm_D=10-37）导致 D 无法收敛到有意义的决策边界。

### G-only 训练已触及 L1+SSIM 上限
- Best val_loss = 0.635 at E8，后续 79 个 epoch 未改善
- residual_mag 稳定在 0.085-0.12
- L1+SSIM 的固有问题：倾向输出模糊均值（无法产生高频细节）

### 结论
D 无法区分真假的根因是 G 的输出已经足够"接近"真实 CTA（在 L1 意义上），但缺少高频细节。
传统 GAN 路线（直接加 D 到 G 的输出上）在这种场景下效果有限。
G2 精修方案是更合理的替代路线。

## Task 2: G2 精修网络（已实现）

### 方案分析
`ncct * |G(ncct)|` 创造类 CTA 对比度增强效果：
- 血管处 (fake_cta ≈ 1.0): product ≈ ncct（保留）
- 非血管处 (fake_cta ≈ ncct): product ≈ ncct²（压暗）
- 效果：增大血管-组织对比度，视觉上接近 CTA

G2 只需做小幅数值校正，轻量 ResBlock 网络足够。

### 实现的文件
- `config.py` — 新增 `RefineConfig` 数据类 + `pretrained_G2`
- `models/refine_net.py` — RefineNet: conv_in → [ResBlock × N] → conv_out + 全局残差
  - 零初始化最后一层 conv → 初始输出 = 输入（稳定起步）
  - 默认 hidden_dim=64, num_blocks=6 → ~0.5M 参数
  - 输出 clamp 到 [-1, 1]
- `trainer.py` — G2 完整集成（冻结 G、EMA for G2、验证/诊断/检查点）
- `train.py` — 构建 G2 + CLI 参数 `--pretrained_G2`
- `test_pipeline.py` — 新增 `test_refine_net` + `test_g2_pipeline`

### 训练流程
```
ncct → G(ncct) [frozen] → g_pred
intermediate = ncct * |g_pred|
intermediate → G2 → refined_cta
Loss: L1+SSIM(refined_cta, real_cta) [+ GAN + FM if D enabled]
D judges: (ncct, refined_cta) vs (ncct, real_cta)
```

### 使用方法

**Step 1: 先训练好 G（已完成）**

**Step 2: 训练 G2（G 冻结）**
```yaml
# config_g2.yaml 关键配置
model:
  residual_output: true
refine:
  enabled: true
  hidden_dim: 64
  num_blocks: 6
  lr: 0.0001
  lr_scheduler: cosine
  warmup_steps: 200
  freeze_G: true
discriminator:
  enabled: false          # 先不加 D，看 G2 baseline
train:
  pretrained_G: /path/to/G/checkpoint_epoch0070.pt
  pretrained_G2: ''
  skip_warmup: true       # G2 有自己的 warmup
  num_epochs: 100
  output_dir: ./outputs_g2
```

**Step 3: 如果 G2 效果好但仍有模糊，启用 D**
```yaml
discriminator:
  enabled: true
  gan_weight: 1.0
  feat_match_weight: 1.0
train:
  pretrained_G: /path/to/G/checkpoint.pt
  pretrained_G2: /path/to/G2/checkpoint_best.pt
```

# TODO
- 在服务器上运行 `python test_pipeline.py` 验证所有测试通过
- 按上述步骤训练 G2，观察 `inter_l1`（中间产物误差）和 `refine_delta`（G2 修正幅度）
- 如果 G2 baseline 满意，再尝试加 D





**未来可能计划，暂时不用实现**  
Perceptual Loss (VGG/LPIPS): Add a perceptual loss using pre-trained VGG features. This is especially effective for generating realistic textures in medical images. Since the data is single-channel grayscale, you'd replicate to 3 channels before feeding into VGG. Weight ~0.1-1.0 relative to L1.  

Progressive Training Strategy: Train in phases:
Phase 1: G only with L1+SSIM (50 epochs)
Phase 2: Enable R (50 epochs)
Phase 3: Enable D (100+ epochs)
This prevents GAN instability early on and lets G converge to a reasonable baseline first.  

Mixed Precision (AMP): Use torch.cuda.amp for the generator forward/backward pass. The UNet is large (61M params) and AMP would roughly halve memory and speed up training 1.5-2x without quality loss.  

Gradient Penalty (R1): Instead of or in addition to SN, consider R1 gradient penalty on the discriminator. This provides a more direct regularization and is used in StyleGAN-family models. Typical weight: 10.0.  

Attention in Registration Net: The current registration UNet is very lightweight (0.07M). For cases with complex misalignments, adding a single self-attention layer at the bottleneck could help capture long-range spatial correspondences.  

Multi-resolution Loss: Compute L1+SSIM at multiple resolutions (original + 2x/4x downsampled). This helps the generator learn both fine detail and global structure simultaneously.  

Curriculum on Lung Weight: Start with lung_weight=1.0 and gradually increase to 10.0 over the first 20-30 epochs. A sudden 10x emphasis on lung regions might cause early instability.  

Test-Time Augmentation (TTA): During inference, run the model with 2-4 augmented versions (flips, small rotations) of the input and average predictions. This typically improves SSIM/PSNR by 0.5-1.0 dB at the cost of proportional inference time.