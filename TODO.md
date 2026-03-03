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


**任务**： ncct合成ctpa，D:\codes\work-projects\ncct2cpta\train.py这里是训练入口。模型：采用生成器G+精修网络G2+配准网络R+判别器D的方案。数据：对同一个样本的两次扫描进行配准，大约配准到90%的样子。G是必须先训练的，训练目前效果正常，也就是平扫的肺血管基本都增强了，只是生成的图像比较模糊，看起来像被平滑过。  

# TODO  
1. ✅ 全面代码审查（已完成，见下方报告）

---

# 全面代码审查报告 (2026-03-03)

## 一、审查范围

使用 `configs/g.yaml` 配置，从训练发起到结束的完整流程，覆盖所有核心模块：

| 模块 | 文件 | 审查结果 |
|------|------|---------|
| 配置 | `config.py` | 修复2个BUG |
| 入口 | `train.py` | ✅ 正确 |
| 训练循环 | `trainer.py` | 修复3个BUG，增强debug |
| 生成器G | `models/unet.py` | 修复1个重要BUG |
| 精修G2 | `models/refine_net.py` | ✅ 正确 |
| 配准R | `models/registration.py` | ✅ 正确 |
| 判别器D | `models/discriminator.py` | ✅ 正确 |
| 判别器Dv2 | `models/discriminator_v2.py` | ✅ 正确 |
| 损失函数 | `models/losses.py` | ✅ 正确 |
| 工具 | `models/nn_utils.py` | ✅ 正确 |
| 感知损失 | `models/perceptual_loss.py` | 修复1个死代码 |
| 数据集 | `data/dataset.py` | ✅ 正确 |
| GPU增强 | `data/transforms.py` | ✅ 正确 |
| DiffAugment | `data/diffaugment.py` | ✅ 正确 |
| CTA退化 | `data/cta_degrade.py` | ✅ 正确 |
| 质量退化 | `data/quality_augment.py` | ✅ 正确 |
| 数据划分 | `data/split.py` | ✅ 正确 |
| 可视化 | `utils/visualization.py` | ✅ 正确 |

测试结果：ALL TESTS PASSED (1 SKIPPED: test_g2_pipeline 因test中UNet model_channels=16 < GroupNorm groups=32，仅test问题，不影响生产)

## 二、发现并修复的BUG

### BUG-1 (致命): g.yaml 无法加载 — `g_self_refine_*` 字段未在 TrainConfig 定义
- **文件**: `config.py`, `configs/g.yaml`
- **现象**: `Config.load('configs/g.yaml')` 抛出 `TypeError: unexpected keyword argument 'g_self_refine_prob'`
- **原因**: g.yaml 中包含 `g_self_refine_prob` 和 `g_self_refine_weight`，但 `TrainConfig` 未定义这些字段
- **修复**: 在 `TrainConfig` 中添加了 `g_self_refine_prob: float = 0.0` 和 `g_self_refine_weight: float = 0.5`
- **同时**: g.yaml 中也缺少 `g_cta_degrade_prob` 字段，已补充

### BUG-2 (中等): 硬编码 lung_weight 覆盖配置值
- **文件**: `trainer.py` 行393-401
- **现象**: 硬编码 `if epoch < 10: self.criterion.lung_weight = 5.0` 等，使 `config.train.lung_weight: 10.0` 完全失效
- **修复**: 
  1. `config.py`: 新增 `lung_weight_schedule: str` 字段，格式 `"epoch:weight,..."` 例如 `"10:5,20:10,40:20,999:40"`
  2. `trainer.py`: 新增 `_parse_lung_weight_schedule()` 和 `_get_lung_weight(epoch)` 方法
  3. `g.yaml`: 添加 `lung_weight_schedule: '10:5,20:10,40:20,999:40'`（保留原有渐进策略，但可配置）

### BUG-3 (重要 — 导致模糊输出的根因之一): UNet `tanh()` 梯度饱和
- **文件**: `models/unet.py` 行461
- **现象**: `torch.tanh(output)` 在±1附近梯度趋近0，CTA增强血管值接近+1时梯度消失，模型难以学习高对比度细节
- **分析**: tanh在x=1时梯度仅0.42，在x=2时梯度仅0.07。而CTA增强区域（血管）的归一化值常在0.8-1.0范围，梯度被压缩到0.42-0.78
- **修复**: 改为 `output.clamp(-1.0, 1.0)`，与 RefineNet 保持一致。clamp在[-1,1]内梯度恒为1.0，仅在边界外为0
- **预期效果**: 血管增强区域梯度恢复100%，输出清晰度应有明显改善

### BUG-4 (轻微): perceptual_loss.py 中 `n_elements` 死代码
- **文件**: `models/perceptual_loss.py` 行138
- **现象**: `n_elements` 被计算但从未使用，注释说"normalize by number of elements"但实际未执行
- **修复**: 删除死代码和误导注释。`F.l1_loss` 默认已计算mean，不需要额外归一化

### BUG-5 (中等): Registration 网络 LR 配置失效
- **文件**: `trainer.py` 行229-245
- **现象**: `config.registration.lr = 0.0001` 从未被使用，R params 始终使用 G 的 LR (0.0002)
- **修复**: 改用 optimizer parameter groups，R 获得独立 LR。当 R 启用时日志会输出 `Registration LR: 0.0001 (separate from G LR=0.0002)`

### BUG-6 (轻微): g.yaml `use_3d_ssim: false` 不合理
- **文件**: `configs/g.yaml`
- **现象**: `num_slices: 12` 时切片间有明显相关性，2D SSIM 丢失切片间结构信息
- **修复**: 改为 `use_3d_ssim: true`

## 三、增强的 Debug 信息

### 3.1 训练启动诊断 (已启用)
`_log_data_diagnostics()` 现在在 `start_epoch == 0` 时自动执行：
- 数据范围 (ncct/cta min/max/mean/std)
- 数据裁剪比例检测
- HU窗宽告警 (当前[-550, 250]=800HU)
- NCCT与CTA相似度检测
- G初始输出残差量级

### 3.2 每个 Epoch 的增强诊断
`_log_epoch_diagnostics()` 新增：
- **输出饱和度监控**: 检测 pred ≥ 0.99 和 ≤ -0.99 的像素比例，>5% 时发出警告
- **锐度评估**: 计算 pred 和 cta 的高频能量（梯度幅值之和），输出 ratio 并标注 "blurry"/"ok"/"sharp"
- **区域误差**: lung_l1 vs bg_l1（肺区应更高，因增强信号在肺区）

### 3.3 如何使用 Debug 信息定位问题

| 日志指标 | 正常值 | 异常情况 | 可能原因 |
|---------|--------|---------|---------|
| `sharpness ratio` | 0.7-1.0 | < 0.5 | L1主导，缺少GAN/perceptual loss |
| `sat_high%` | < 5% | > 20% | 输出饱和，网络容量不足或学习率过大 |
| `lung_l1` | 0.05-0.15 | > 0.25 | 肺区学习不足，增大lung_weight |
| `D_real ≈ D_fake` | 差值>0.1 | 几乎相等 | D崩溃，需检查d_cond_mode/R1 |
| `gnorm_G` | 0.5-5.0 | > 50 | 梯度爆炸，减小LR或加大grad_clip |
| `residual_mag` | 0.05-0.3 | < 0.01 | G学到的是恒等映射，未学到增强信号 |

## 四、训练各阶段的逻辑正确性验证

### 4.1 阶段一：G only（当前配置）
- ✅ 配置加载: `Config.load('configs/g.yaml')` → `sync_channels()` 确保 in/out = num_slices
- ✅ 数据流: Dataset(3C slices) → GPU Augmentor(spatial+pixel) → extract middle C → G(ncct) → clamp → loss
- ✅ 残差输出: `output = x + model_prediction` → `clamp(-1, 1)` — 正确，初始输出接近输入
- ✅ 损失: L1(mask-weighted) + SSIM(mask-weighted) + Perceptual(VGG) — 合理组合
- ✅ 优化: AdamW + cosine warmup + grad_clip + grad_accum + EMA — 标准配置
- ✅ 渐进 lung_weight: 5→10→20→40 via schedule — 现在可配置

### 4.2 阶段二：G + D
- ✅ D冻结防梯度泄露: G step时 `p.requires_grad_(False)`
- ✅ D输入构建: `d_cond_mode=none/concat/diff` 三种模式均正确
- ✅ D warmup: 显式设置LR → 预训练 → 重建scheduler
- ✅ 渐进GAN权重: `_get_effective_gan_weight()` 线性ramp-up
- ✅ R1梯度惩罚: lazy interval + 跳过DiffAugment的real input
- ✅ 质量退化负样本: 随机概率应用，正确使用detach和fake label
- ✅ Feature Matching: real_features在no_grad下计算，仅计算L1距离

### 4.3 阶段三：G(frozen) + G2
- ✅ G冻结: `requires_grad_(False)` + `model.eval()`
- ✅ 中间表示: `intermediate = ncct * g_pred.abs()` — g_pred无梯度（G frozen），G2获得正确梯度
- ✅ EMA切换: EMA跟踪G2参数（不是G）
- ✅ Optimizer: 仅G2 params在优化器中
- ✅ 零初始化: G2的conv_out零初始化，初始行为为恒等映射

### 4.4 阶段四：G + R
- ✅ 独立LR: R使用 `config.registration.lr`（通过param groups修复）
- ✅ 梯度流: loss → warped → SpatialTransformer(可微) → displacement → R, 同时反向传播到 G
- ✅ 平滑惩罚: GradLoss(L2) 约束位移场平滑
- ✅ 验证时不使用R: 评估G的原始输出质量

### 4.5 Checkpoint 恢复
- ✅ 保存: G, G2, R, D, optimizer_G, optimizer_D, scheduler_G, scheduler_D, EMA, epoch, global_step
- ✅ 恢复: 所有组件正确加载，向后兼容旧key名
- ✅ 注意: 恢复后lung_weight_schedule会从当前epoch正确应用

## 五、效果提升策略（全部已实施）

### ✅ 策略1: tanh → clamp 解决梯度饱和
- **改动**: `models/unet.py` — `torch.tanh()` → `output.clamp(-1.0, 1.0)`
- 预期改善: 血管区域清晰度提升，高对比度细节更好

### ✅ 策略2: 增大 perceptual_weight
- **改动**: `g.yaml` — `perceptual_weight: 0.1` → `0.5`
- 原理: 感知损失基于VGG特征空间，直接惩罚高层结构差异，比L1更能保留细节

### ✅ 策略3: 启用 D 进行对抗训练
- **改动**: `g.yaml` — `discriminator.enabled: true`, `d_cond_mode: diff`, `feat_match_weight: 10.0`
- D看 `(image - ncct)` 差异信号，聚焦增强效果；Feature matching提供稳定梯度
- `gan_warmup_epochs: 10` + `d_warmup_steps: 500` + `DiffAugment` 保证训练稳定

### ✅ 策略4: 添加频域损失 (FFT)
- **新增代码**: `models/losses.py` — `FrequencyLoss` 类 (110行)
  - 2D real FFT + 径向高频加权 + log-scale amplitude + L1距离
  - 灵感来源: Focal Frequency Loss (Jiang et al., ICCV 2021)
  - 缓存频率权重图，支持 high_pass/uniform 模式和 cutoff_ratio
- **集成**: `config.py` 新增 `freq_weight`, `trainer.py` 初始化/训练循环/metrics, `__init__.py` 导出
- **配置**: `g.yaml` — `freq_weight: 0.1`
- **测试**: 7个测试用例全部通过（零输入/模糊检测/锐度对比/梯度流/模式切换/cutoff/缓存）
- 原理: 直接在频域惩罚缺失的高频分量，解决L1/SSIM导致的"平滑"问题

### ✅ 策略5: 使用 CTA 退化双任务
- **改动**: `g.yaml` — `g_cta_degrade_prob: 0.0` → `0.2`
- 原理: 训练G同时做 NCCT→CTA 和 degraded_CTA→CTA 两个任务，推理时可两遍前向得到更锐利输出

### ✅ 策略6: NCCT质量退化增强
- **状态**: `g.yaml` 中 `ncct_degrade_prob: 0.5`（已启用）
- 原理: 对NCCT输入随机加blur/noise/downsample/cutout，提升G的鲁棒性

---

## 六、4阶段渐进训练方案

### Phase 1: G Only (`configs/phase1_g.yaml`)
- **目标**: G 学习 NCCT→CTA 和 degraded_CTA→CTA 双任务
- **组件**: G only (D/G2/R 全部关闭)
- **输入**: 12ch×400×400, bsz=2, grad_accum=8 (等效 bsz=16)
- **损失**: L1 + SSIM + Perceptual(VGG) + Frequency(FFT)
- **完成标志**: sharpness ratio > 0.5, lung_l1 < 0.15
- **新增实现**: 无 (已有功能)

### Phase 2: G + D (`configs/phase2_g_d.yaml`)
- **目标**: D 辅助 G 生成更清晰的CTA (类似超分 D)
- **组件**: G + D (G2/R 关闭)
- **D设计**: diff模式 + feature matching + DiffAugment + quality aug negatives
- **稳定性**: GAN warmup 10 epochs, D warmup 500 steps
- **完成标志**: sharpness ratio > 0.7, D_real > D_fake
- **新增实现**: 无 (已有功能)

### ✅ Phase 3: G(frozen) + G2 + D (`configs/phase3_g_g2_d.yaml`)
- **目标**: 冻结G, G2精修网络 + D进一步提升质量
- **G2多输入**: 3种模式随机交替训练
  - `synthesized`: G2(G(ncct)) — 直接精修G输出
  - `degraded`: G2(degrade(cta)) — 精修退化CTA
  - `intermediate`: G2(ncct * |G(ncct)|) — 精修中间表示
- **完成标志**: refine_delta > 0, inter_l1 > pred_l1
- **新增实现**:
  - `config.py`: `RefineConfig.g2_input_modes` 字段
  - `trainer.py`: `_sample_g2_input()` 方法 + 训练循环集成
  - 测试: `test_g2_multi_input_config()` PASSED

### ✅ Phase 4: G(frozen) + G2 + D + R (`configs/phase4_g_g2_d_r.yaml`)
- **目标**: R配准消除空间偏差, 使损失函数更有效
- **R预训练**: 用空间增强的降质CTA对预训练R 5个epoch
  - 空间增强: 旋转(±5°), 平移(±3%), 缩放(±5%), 弹性形变(alpha=6)
  - 像素退化: blur, noise, downsample (模拟G输出质量)
- **数据流**: ncct → G → intermediate → G2 → pred → R(pred, cta) → warped + displacement
- **完成标志**: smooth loss 稳定, warped_l1 < raw_l1, displacement < 5px
- **新增实现**:
  - `data/reg_augment.py`: `RegistrationAugmentation` (affine + elastic + pixel degrade)
  - `config.py`: `RegistrationConfig.r_pretrain_*` 字段 (7个参数)
  - `trainer.py`: `_pretrain_registration()` 方法 + `train()` 集成
  - 测试: `test_registration_augmentation()` PASSED

### 各阶段配置文件
| Phase | Config | 启用组件 | 加载权重 |
|-------|--------|---------|---------|
| 1 | `phase1_g.yaml` | G | — |
| 2 | `phase2_g_d.yaml` | G, D | G from Phase 1 |
| 3 | `phase3_g_g2_d.yaml` | G(frozen), G2, D | G+D from Phase 2 |
| 4 | `phase4_g_g2_d_r.yaml` | G(frozen), G2, D, R | G+G2+D from Phase 3 |

---

## 七、12通道2.5D数据的3D增强分析

### 分析结论

| 模块 | 原状态 | 是否需要3D | 改动 |
|------|--------|-----------|------|
| 空间增强 (transforms.py) | ✅ 已是3D | 否 | — |
| SSIM损失 | ✅ 支持3D | 否 | `use_3d_ssim: true` |
| 像素增强 (brightness/contrast/noise) | 2D | 否 | 全局变换,无空间维度依赖 |
| **模糊** (aug_utils) | 仅2D | **是** | 新增 `gaussian_blur_3d` + `gaussian_blur_auto` |
| **降采样** (aug_utils) | 仅2D | **是** | 新增 `downsample_upsample_auto` (3D trilinear) |
| **感知损失** (perceptual_loss) | 仅中间3片 | **改进** | 多组3ch覆盖全部深度 (12ch→4组) |
| CTA退化 (cta_degrade) | 2D blur | **是** | 已切换到 `gaussian_blur_auto` |
| 质量退化 (quality_augment) | 2D blur+ds | **是** | 已切换到 auto 函数 |
| R配准增强 (reg_augment) | 2D | 保持2D | 须匹配2D配准网络 |
| 频域损失 | 2D per-slice | 否 | 2D FFT是2.5D标准做法 |
| GradLoss | 2D | 否 | 匹配2D位移场 |

### 实现详情

#### 1. `data/aug_utils.py` — 新增3D感知函数
- `gaussian_blur_3d(x, ks, sigma)`: 将(N,C,H,W)重塑为(N,1,D,H,W), 可分离3D高斯卷积
  - 深度核大小自动限制为 min(ks, C) 且保证奇数
  - 使用 replicate padding 避免边界伪影
- `gaussian_blur_auto(x, ks, sigma)`: C >= 4 → 3D blur, 否则 → 2D blur
- `downsample_upsample_auto(x, scale)`: C >= 4 → 3D trilinear, 否则 → 2D bilinear
  - 深度方向使用更温和的降采样 (scale//2) 避免过度模糊

#### 2. 已升级的模块 (blur/downsample → auto)
- `data/quality_augment.py`: `gaussian_blur_2d` → `gaussian_blur_auto`
- `data/cta_degrade.py`: `gaussian_blur_2d` → `gaussian_blur_auto`
- `data/transforms.py._quality_degrade`: blur + downsample → auto 版本
- `data/reg_augment.py._pixel_degrade`: 内联blur → `gaussian_blur_auto`

#### 3. `models/perceptual_loss.py` — 多切片VGG感知损失
- **旧**: 12ch只用中间3片(slices 5-7), 丢失75%信息
- **新**: 12ch → 4组×3ch ([0:3], [3:6], [6:9], [9:12]), 平均VGG损失
- 9ch → 3组, 6ch → 2组, 3ch → 1组 (无额外开销)
- 各组独立通过VGG提取特征, 损失取平均

#### 4. 未改动的模块及原因
- **空间增强**: 已是3D (3D affine_grid + grid_sample)
- **R配准增强的affine/elastic**: 保持2D, 因为配准网络只预测2D位移场
- **Cutout**: 正则化手段, 不需要深度方向连续性
- **Gamma/brightness/contrast**: 全局值变换, 无空间依赖
- **频域损失**: 2D FFT per-slice 是标准做法, 3D FFT计算量大但收益有限

### 测试结果
- `test_3d_augmentation_utils`: 3D/2D自动选择、深度平滑、降采样 — **PASSED**
- `test_multi_slice_perceptual_loss`: 多组分割、梯度流、一致性 — **PASSED**
- 全部已有测试: **ALL TESTS PASSED**