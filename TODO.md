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


## 测试环境为**py310**  


**任务**： ncct合成ctpa，D:\codes\work-projects\ncct2cpta\train.py这里是训练入口。数据：对同一个样本的两次扫描进行配准，大约配准到90%的样子。方案1：采用生成器G+精修网络G2+配准网络R+判别器D的方案。G是必须先训练的，训练目前效果正常，也就是平扫的肺血管基本都增强了，只是生成的图像比较模糊，看起来像被平滑过。方案2：采用D:\codes\work-projects\ncct2cpta\ldm里面的方案，改方案参考方案1和D:\codes\work-projects\ncct2cpta\ref_models\pix2pixHD，D:\codes\work-projects\ncct2cpta\ref_models\guided-diffusion来训练vae，参考D:\codes\work-projects\ncct2cpta\ref_models\guided-diffusion来训练diffusion。D:\codes\work-projects\ncct2cpta\ldm\train_vae.py和train_diffusion.py是训练入口。  

# TODO  
1. 请全面，细致，认真的分析和审查vae这块代码，保证全流程代码正确无误，算法正确实现，训练逻辑正确等等。我训练了一版vae，见D:\codes\work-projects\ncct2cpta\outputs\config0.yaml配置和对应日志。  
2. 请全面，细致，认真的分析和审查diffusion这块代码，保证全流程代码正确无误，算法正确实现，训练逻辑正确等等。我在vae基础上训练diffusion，见D:\codes\work-projects\ncct2cpta\outputs\config1.yaml配置和对应的日志。  
3. vae和diffusion的训练是否有高质量提升效果的策略，工程，技巧等等加入（不要局限于医学图像领域）。  
4. 如果有修改，记得同步更新D:\codes\work-projects\ncct2cpta\configs\ldm_default.yaml，注意需要加入注释来解释每个参数的选项和意义。  