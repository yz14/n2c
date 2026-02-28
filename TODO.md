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
将代码上传服务器，用真实数据训练，从D:\codes\work-projects\ncct2cpta\train.py






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