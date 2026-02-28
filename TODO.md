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


# TODO  
~~判别器加入后效果无提升分析~~ **已完成分析和修复 (2026-02-28 v2)**

## 分析结论

### 🔴 Bug1（已修复）：验证和可视化使用在线权重而非 EMA 权重
- EMA 权重理论上更好，但从未用于验证/可视化/best checkpoint 选择
- 用户看到的"模糊"可能部分因为没用 EMA

### 🔴 问题2（需实验确认）：G 总 loss 中 FM 过强，GAN 信号被淹没
- 重建 loss: 0.67 (47%)，GAN: 0.26 (18%)，FM: 0.50 (35%)
- GAN 是推动清晰度的信号，但仅占 18%
- FM 本质是 D 特征空间的 L1 正则，不产生锐化效果

### 🟡 问题3：D 梯度范数极高（40→27），grad_clip=5.0 裁掉 80%+
- D 信号极不稳定，需要通过 D_real/D_fake 均值来确认 D 是否有效

### 🟡 问题4（已修复）：pretrained_G 后 LR warmup 从零重启
- G 在前 ~2 epochs 几乎不学习，导致 val_loss 暂时恶化
- 新增 `skip_warmup: true` 配置选项

## 已实施修复
1. `trainer.py` — **EMA 验证**：验证和可视化改用 EMA 权重（_swap_ema_weights/_restore_model_weights）
2. `trainer.py` — **D 诊断日志**：新增 D_real、D_fake（D 对真/假图的平均输出）、w_recon、w_gan、w_fm（加权 loss 组成）
3. `config.py` — **skip_warmup 选项**：`skip_warmup: true` 跳过 LR warmup

## 下一步实验（按优先级）

### 实验1：EMA + skip_warmup 重训 G+D（最小改动验证）
在 config.yaml 中加入 `skip_warmup: true`，其他不变，重新训练 G+D。
观察日志中的 `D_real` 和 `D_fake`：
- 如果 D_real ≈ D_fake（D 无法区分真假）→ D 本身无效
- 如果 D_real >> D_fake（D 有效）但图像仍模糊 → loss 权重有问题

### 实验2：降低 FM 权重（如果实验1显示 D 有效但仍模糊）
将 `feat_match_weight` 从 10.0 降到 2.0，让 GAN 信号占比提升到 30%+。

### 实验3：提高 GAN 权重（如果实验2仍不够）
将 `gan_weight` 从 1.0 提到 2.0-5.0。注意可能导致训练不稳定。

### 实验4：G-only 训练更久（对比基线）
G-only 在 E30 仍在收敛，可训练到 100+ epochs 看 loss 是否还能下降。  





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