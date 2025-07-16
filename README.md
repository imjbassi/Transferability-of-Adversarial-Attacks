# Evaluating Transferability of Adversarial Attacks Across Machine Learning Models

This repository contains code and results for the study: **"Evaluating Transferability of Adversarial Attacks Across Machine Learning Models"**, conducted by Jaiveer Bassi as part of academic research at Grand Canyon University.

## Overview

Adversarial examples—inputs crafted with subtle perturbations—can cause machine learning models to misclassify with high confidence. One of the most concerning properties of such attacks is their **transferability**: adversarial inputs designed for one model often remain effective against other models, including those with different architectures.

This research investigates how adversarial examples generated from one convolutional neural network (CNN) architecture perform against other networks trained on the CIFAR-10 dataset.

## Objectives

- Assess the **transferability** of adversarial examples across CNNs.
- Evaluate **attack success rates** for different source-target model combinations.
- Compare **FGSM**, **PGD**, and **Carlini-Wagner (CW)** attacks.
- Provide insights into which model architectures are most vulnerable in black-box settings.

## Methodology

### Dataset

- **CIFAR-10**: 60,000 color images (32×32), 10 classes.
- 50,000 training and 10,000 test samples.

### Models Used

- `ResNet-18`: Residual learning framework with skip connections.
- `VGG16`: Deep sequential model without residuals.
- `MobileNetV2`: Lightweight architecture optimized for mobile.

### Attacks Implemented

- **FGSM**: Fast Gradient Sign Method (single-step attack).
- **PGD**: Projected Gradient Descent (iterative attack).
- **CW**: Carlini-Wagner L2 Attack (optimization-based).

### Metrics

- **Attack Success Rate (ASR)**: Misclassification on source model.
- **Transfer Success Rate (TSR)**: Misclassification on target model.
- **Accuracy Under Attack**: Drop in classification accuracy under adversarial inputs.

## Results

- All three attacks (FGSM, PGD, CW) achieved **>99% ASR**.
- **High transferability** observed across all model pairs.
- Transferability varied slightly by architecture (ResNet and MobileNet had more mutual vulnerability than VGG16).
- CW attacks, while complex, still showed **near-complete transferability**—contradicting past assumptions.

## Key Findings

- Iterative attacks like PGD and CW transfer **as effectively** as FGSM.
- Adversarial examples remain effective across different model designs.
- Sole reliance on architectural variation is **insufficient** for defense.
- Ensemble training and model-agnostic defense strategies are needed for true robustness.
