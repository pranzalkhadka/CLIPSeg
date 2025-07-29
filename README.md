# ClipSeg Implementation

## Overview

This repository contains a custom implementation of **ClipSeg**, a model for zero-shot and few-shot image segmentation that leverages the power of **CLIP** (Contrastive Language-Image Pretraining) to perform segmentation based on textual prompts.

ClipSeg combines CLIP's vision and text understanding capabilities with a transformer-based decoder to generate segmentation masks for objects described by text without requiring extensive labeled training data.

This is my personal effort to implement ClipSeg from scratch, inspired by the original work by **Lüddecke and Ecker (2022)**. While it captures the core ideas of ClipSeg, it may include different choices and implementations.

---

## What is ClipSeg?

**ClipSeg** is a deep learning model designed for **zero-shot image segmentation**, enabling users to segment objects in images using free-form text prompts (e.g., `"dog"`, `"red car"`, or `"tree in the background"`).

It builds on the **CLIP model**, which aligns images and text in a shared embedding space, allowing the model to understand and segment objects based on textual descriptions without task-specific training.


##  My Implementation

This repository contains my version of ClipSeg, implemented from scratch in **PyTorch**. The implementation includes the following components:

- **CLIP Models**: Utilizes `CLIPVisionModelWithProjection` and `CLIPTextModelWithProjection` from the Hugging Face `transformers` library to extract image and text features.
- **Transformer Decoder**: A custom transformer decoder with multi-head self-attention to process CLIP features and generate segmentation masks.
- **FiLM Modulation**: Conditions the decoder’s features on text embeddings using feature-wise linear modulation.
- **Multi-scale Features**: Extracts features from specific layers of the CLIP vision model (layers 3, 7, and 9) for richer representations.
- **Output**: Produces a segmentation mask (`224x224` by default) for a given image and text prompt.

## References

- Lüddecke, T., & Ecker, A. (2022). *Image Segmentation Using Text and Image Prompts*. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 7086–7096. [arXiv:2112.10003](https://arxiv.org/abs/2112.10003)
