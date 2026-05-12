# Task Instruction for Codex

I am a first-year graduate student modifying the scBGDL model for survival prediction.

This repository contains two code versions:

1. scBGDL-main/
This is the original scBGDL code from the authors. Please use it as the main reference for the original GAT + MinCutPool + Transformer architecture, data input/output format, training process, and Cox survival loss.

2. my_kegg_mamba_version/
This is my modified KEGG + Mamba version. Please use it as a reference for KEGG mapping, batch_S, pathway pooling, and MambaBlock implementation.

The goal is to build a unified ablation framework with four switchable model variants:

Model A: GAT + MinCutPool + Transformer
- Original scBGDL-style baseline.

Model B: GAT + KEGG mean pooling + Mamba
- Based on my current KEGG + Mamba version.

Model C: GAT + KEGG attention pooling + Mamba
- Replace KEGG mean pooling with attention-based pathway pooling.

Model D: GAT + KEGG attention pooling + Bidirectional Mamba + gene-level residual fusion
- Final recommended model.

Requirements:
1. Do not modify the data preprocessing unless necessary.
2. Keep the original training pipeline and Cox survival prediction logic as much as possible.
3. Implement a unified model_factory.py so that I can select models using --model_type A/B/C/D.
4. Keep the forward input/output format consistent across all four models.
5. Model C must implement KEGG-guided attention pathway pooling.
6. Model D must implement Bidirectional Mamba and gene-level residual fusion.
7. Add a basic shape test to check that all four models can complete forward propagation.
8. Return prediction risk score, and if possible also return intermediate outputs such as pathway attention, gene representation, and pathway representation for interpretation.
9. After implementation, explain which files were modified and how to run each model.

Important:
First read the repository and give me a modification plan. Do not modify code immediately.
