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
1. 我修改了代码并提交，随后在服务器上继续了训练和测试，你可以通过git来查看我具体修改了哪些地方。我先只训练G，配置为D:\codes\work-projects\ncct2cpta\outputs\config0.yaml，日志为D:\codes\work-projects\ncct2cpta\outputs\train0.log。得到我比较满意的结果后（肺血管基本增强了，但是生成的图像有点模糊），于是我启用了判别器D，配置为D:\codes\work-projects\ncct2cpta\outputs\config.yaml，日志为D:\codes\work-projects\ncct2cpta\outputs\train.log，我查看了训练保存的png图像，发现启用判别器似乎对效果没有什么提升，生成的图像仍然比较模糊。请你根据日志进行细致的分析，思考。如果可以发现明确的问题，请更正；如果不明确，请加入debug，并告诉我如何进行试验，以便发现根本的问题。注意，请高质量完成。  
2. 我发现用生成的cta，逐个像素绝对值乘以原ncct，并恢复正负符号，主观上看起来挺像真实cta的，看起来也不算模糊，只不过像素值有些偏差，具体是，x_ncct * x_fake_cta，原先血管的地方x_fake_cta接近1.0，而原先组织的地方仍然是原像素值，所以相乘后，原先血管的地方像素值几乎没有变化，而非血管的地方似乎被平方了，所以值变小了，从而突出了血管像素值，使得看起来像真实的cta。所以，根据这个发现，我可不可以增加一个生成器G2，专门用来微调x_ncct * x_fake_cta的数值，使得它不仅看起来像，数值上也和cta一致。如果这个方案可行，那么G2也需要开关，而且它应该在G之后，在配准网络之前，而且判别器也是针对G2的输出，而且当启动G2后，G应该冻结不再训练（请你思考这样设计是否合理）。G2应该用复杂网络还是轻量网络（我目前倾向轻量网络）。请你仔细分析这个方案是否合理，如果合理，请高质量实现。  




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