---
sidebar_position: 3
---

# FastVLM

FastVLM 是一个轻量级视觉语言模型（目前为 0.5B 参数版本）。

:::warning 状态
FastVLM 目前处于初始化阶段，仅有模型元数据定义，尚未完全集成。
:::

## 模型规格

| 属性 | 值 |
|---|---|
| Hidden Size | 896 |
| 层数 | 24 |
| KV Head 数 | 2 |
| Head Dim | 64 |
| Max Length | 8192 |
| EOS Token | 151645 |
| Image Token | 151646 |

## 视觉预处理

| 参数 | 值 |
|---|---|
| 最长边 | 448 |
| 填充尺寸 | 448×448 |
| 填充颜色 | RGB(128, 128, 128) |
| 色彩空间 | RGB |
| 布局 | NCHW |
| 缩放因子 | 1/255 |

## 生成默认值

| 参数 | 值 |
|---|---|
| Max New Tokens | 128 |
| Temperature | 0.7 |
| Top-P | 0.9 |

## 关键文件

- `crates/lumen-hub/src/models/fastvlm/metadata.rs` — 模型元数据常量
