🌾 Beyond Visible Spectrum: AI for Agriculture
Multimodal Crop Disease Classification (ICPR 2026)

This repository documents a progressive modeling journey for multimodal crop disease classification using RGB, Multispectral (MS), and Hyperspectral (HS) imagery.
Each model is saved as a separate notebook and named according to its model number, showing a clear improvement in accuracy and design sophistication.

The goal of this work is to explore how increasingly advanced architectures and fusion strategies improve classification performance on agricultural disease data.

📁 Repository Structure
├── second-aiforagri.ipynb
├── Third_model.ipynb
├── Fourth_Moddel.ipynb
├── fifth_model.ipynb
└── README.md

Each notebook represents an independent experiment and can be run individually.

📊 Model Performance Summary
Model	Notebook Name	Key Idea	Validation Accuracy
2	second-aiforagri.ipynb	Baseline multimodal pipeline	0.58947
3	Third_model.ipynb	Spectral feature learning improvements	0.61052
4	Fourth_Moddel.ipynb	CNN ensemble (RGB + MS + HS)	0.65263
5	fifth_model.ipynb	Deep multimodal fusion (final model)	Highest (Final)

📈 Accuracy increases monotonically, reflecting systematic architectural and training improvements.

🔹 Model-wise Explanation
🔹 Second Model (Baseline Multimodal Model)

Notebook: second-aiforagri.ipynb
Accuracy: 0.58947

🔧 Key Characteristics

Uses basic preprocessing for RGB, MS, and HS images.

Relies on standard CNN feature extraction.

Minimal tuning of:

Learning rate

Batch size

Image resizing

Limited fusion between modalities.

⚠️ Limitations

Modalities are not deeply integrated.

Spectral richness of HS data is underutilized.

Acts primarily as a reference baseline.

🔹 Third Model (Improved Spectral Learning)

Notebook: Third_model.ipynb
Accuracy: 0.61052

🔧 Key Improvements

Better handling of hyperspectral bands.

Avoids naive band-mean aggregation.

Focuses on learning discriminative spectral–spatial patterns.

Introduces improved feature extraction logic.

📈 Outcome

Clear performance gain over the baseline.

Demonstrates that spectral information matters, but still lacks strong modality fusion.

🔹 Fourth Model (CNN Ensemble Model)

Notebook: Fourth_Moddel.ipynb
Accuracy: 0.65263

🧠 Architecture Overview

Independent CNN backbones for:

RGB images

Multispectral images

Hyperspectral images

Features from all branches are combined as an ensemble.

🔧 Technical Details

Uses timm models.

Albumentations for data augmentation.

Standard cross-entropy loss.

GPU-accelerated training on Kaggle.

🚀 Strengths

First model to treat each modality as a first-class citizen.

Stronger generalization.

Sets the stage for deep fusion.

⭐ Fifth Model (Final Model – Deep Multimodal Fusion)

Notebook: fifth_model.ipynb
Accuracy: Highest (Best Performing Model)

This is the final and most important model in the repository.

🧠 Architecture: MultimodalFusionNet

Each modality is processed through a dedicated deep backbone, and learned features are fused jointly.

🔹 Modal Branches
Modality	Backbone	Input Channels	Output Features
HS	ResNet-18	30 channels	512
MS	ResNet-18	5 channels	512
RGB	EfficientNet-B0	3 channels	512
🔗 Feature Fusion Strategy

Feature vectors from all three branches are concatenated:

512 (HS) + 512 (MS) + 512 (RGB) = 1536-D vector

Passed through a fully connected MLP head:

1536 → 512 → 3 (Health / Rust / Other)
⚙️ Training Configuration

Loss Function: Cross-Entropy Loss

Optimizer: Adam

Mixed Precision Training: Enabled (torch.cuda.amp)

Cross-Validation: 5-Fold Stratified K-Fold

Evaluation Metric: Accuracy

🔁 Advanced Techniques Used

Stratified K-Fold Cross Validation

Out-of-Fold (OOF) predictions

Test-Time Augmentation (TTA)

Horizontal flip

Vertical flip

Averaged predictions

✅ Why This Model Works Best

Learns modality-specific representations.

Performs deep feature-level fusion, not shallow averaging.

EfficientNet enhances RGB texture understanding.

ResNet-based HS/MS branches preserve spectral richness.

Strong regularization via cross-validation and TTA.

This model represents a production-ready multimodal learning pipeline.
