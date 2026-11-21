# Project 1 — NumPy Image Encoder

Python 课程 Project 1：基于 NumPy 实现 OpenCLIP ViT-B/32 图像编码器的前向计算。主要实现位于 `image_encoder.py`，其余辅助脚本来自课程学生版项目。

## 项目结构

- `image_encoder.py`：图像编码器、Patch 划分、多头注意力、Transformer Block 等实现
- `preprocess_image.py`：图像预处理
- `helpers.py`：数值计算辅助函数
- `runtime.py`：课程模型资源加载
- `run.py`：推理入口，保存输出特征
- `check.py`：分阶段数值正确性验证
- `config.json`：模型配置
- `requirements.txt`：依赖列表

## 环境与运行

在项目目录安装依赖：

```bash
python -m pip install -r requirements.txt
```

**注意：** 仓库不包含课程提供的权重、测试图像、参考输出等大型或非原创资源。要运行完整测试，请先从课程原始项目中放回对应的 `weights/` 和 `assets/` 文件，保持原有目录结构。

```bash
python check.py
python run.py
```

## 实验验证

本地 `check.py` 截图记录：assets、preprocess、patches、tokens、block0、last_tokens、output、batch 共 **8 项 PASS**，退出码为 0。最终输出的最大绝对误差约为 `7.302e-07`。本地 `run.py` 截图记录：

```text
Saved (10, 512) to output_tokens.npz
```

## Patch 排列调试实验

调试时曾临时省略 Patch 轴重排步骤，复现 `patches` 及下游检查失败的现象；恢复正确的 `transpose` 后，检查重新全部通过。此处描述的是错误复现与修复验证，不推断最初编写代码时的经历。

## 贡献与工具

本项目基于教师提供的学生版框架。学生完成编码器相关实现并进行本地测试；ChatGPT 辅助解释测试输出、整理项目文档与实验报告。

报告模板：[Experiment-Report-Template-SZU](https://github.com/ChaoFan996/Experiment-Report-Template-SZU)。
