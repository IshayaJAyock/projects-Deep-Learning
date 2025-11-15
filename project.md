1. LightVision: Lightweight CNNs for Real-World Image Classification for Low-Resource Environments -enapa

Develop, compress, and evaluate a suite of lightweight CNNs for a single real dataset (choose EuroSAT (RGB) or TrashNet). The project covers baseline training, knowledge distillation (KD), pruning, quantization-aware training (QAT), combined pipelines, rigorous evaluation (accuracy, FLOPs, model size, CPU latency, energy), deployment artifacts, and a reproducibility-ready release. The core publishable claim is an empirical, statistically rigorous comparison of KD vs QAT vs pruning (and their compositions) on a realistic small dataset with device-level benchmarks and robustness tests.

1. Project goals (what you must deliver) 

1. Implement and train a baseline teacher (high-capacity) and several student (lightweight) models.
2. Apply KD, QAT, and pruning (structured/unstructured) separately and in combinations.
3. Measure accuracy vs efficiency: model size, FLOPs, CPU latency, and energy.
4. Demonstrate deployment on a target device (Raspberry Pi or Android via PyTorch Mobile / TFLite).
5. Produce reproducible code, datasets/splits, and a short quality report with ablations and statistical tests.

2. FairVoice: Bias and Explainability in Speech Emotion Recognition - Bernice

Develop, evaluate, and interpret speech emotion recognition (SER) models that are accurate, fair across demographic groups, and explainable. 
Use CREMA-D as the primary dataset (optionally validate on RAVDESS). Produce reproducible artifacts and a publishable empirical study that quantifies bias, shows interpretable explanations (SHAP / attention / saliency), tests mitigation strategies, and reports statistically rigorous results.

1. Project goals (deliverables)

1. Train baseline and advanced SER models (LSTM on acoustic features; transformer-based using Wav2Vec2/Whisper embeddings).
2. Quantify performance across demographic subgroups (gender, age, ethnicity/accent when available).
3. Apply XAI methods (SHAP, LIME, attention maps, time-series saliency) to explain model predictions and subgroup differences.
4. Implement and evaluate bias mitigation strategies (reweighting, adversarial debiasing, fairness-aware loss).
5. Produce rigorous statistical analysis, robustness tests, and reproducible code + model artifacts.
6. Prepare a paper-ready report with tables, plots, and recommendations for ethical deployment.
7. Produce reproducible code, datasets/splits, and a short quality report with ablations and statistical tests.


Project 3: MultiSense — Multimodal Deep Learning for Emotion Understanding - Jessica 

Project Goals

Develop a multimodal deep learning system that integrates visual, audio, and textual cues to predict emotional states in human interactions. 
The goal is to advance emotion understanding beyond unimodal limitations by exploiting cross-modal synergy and temporal dynamics.

This project aims to:

* Combine vision, speech, and linguistic modalities for emotion recognition.
* Compare unimodal, bimodal, and trimodal models for emotion classification.
* Evaluate fusion strategies (early, late, hybrid) and attention mechanisms.
* Produce a reproducible multimodal benchmark and contribute to open, explainable AI research.   
* Produce reproducible code, datasets/splits, and a short quality report with ablations and statistical tests.

Setup
* Define emotion categories (e.g., 6 Ekman emotions + neutral).
* Select dataset(s): CREMA-D, RAVDESS, IEMOCAP, or custom multimodal data.
* Preprocess audio, video, and transcripts.





Project 4: VisionXplain: Interpretable Vision Transformers for Medical Imaging - Benedict

Project Goals

Develop an interpretable Vision Transformer (ViT)-based framework for medical image classification and explainability. The project seeks to demonstrate that transformer-based models can achieve high diagnostic accuracy while maintaining transparency and clinical trustworthiness through robust interpretability.

This project aims to:

* Implement and fine-tune Vision Transformers (ViTs) and hybrid CNN–ViT architectures.
* Apply explainability methods (Grad-CAM, Attention Rollout, LRP) for medical imaging.
* Evaluate interpretability, reliability, and computational efficiency.
* Develop a reproducible, benchmarkable pipeline for medical AI research.
* Produce reproducible code, datasets/splits, and a short quality report with ablations and statistical tests.


Problem Definition
* Select medical imaging task (e.g., disease classification or lesion detection).
* Define target dataset and label structure (binary or multi-label).

