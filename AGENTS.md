# Agent Guide

## Canonical Lumen Capability Terminology

The following four user-facing capability names are exact product terms. Use
them consistently in the CLI TUI, documentation, READMEs, release notes, and
deployment tools.

| Internal service | Simplified Chinese | English |
| --- | --- | --- |
| `siglip` | `图像语义分析` | `Image Semantic Analysis` |
| `face` | `人物识别` | `Person Recognition` |
| `ocr` | `OCR文字识别` | `OCR Text Recognition` |
| `bioclip` | `BioCLIP物种识别` | `BioCLIP Species Recognition` |

Do not rename these capabilities as `语义搜索` / `Semantic Search`, `人脸识别`
/ `Face Recognition`, bare `OCR`, or bare `物种识别` / `Species Recognition`.
Descriptions may explain that a capability enables natural-language search,
face processing, text extraction, or species classification, but the capability
label itself must use the exact term above. Protocol task names, model names,
database fields, and API identifiers remain unchanged.
