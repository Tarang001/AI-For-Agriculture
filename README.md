# Beyond Visible Spectrum: AI for Agriculture  
## Multimodal Crop Disease Classification (ICPR 2026)

This repository presents a structured progression of deep learning models developed for multimodal crop disease classification using RGB, Multispectral (MS), and Hyperspectral (HS) imagery.  
Each model is saved as a separate notebook and named according to its model number.  
The models are designed incrementally, with accuracy improving at each stage.

The objective of this work is to study how architectural complexity and modality fusion improve classification performance in agricultural datasets.

---

## Repository Structure

├── second-aiforagri.ipynb  
├── Third_model.ipynb  
├── Fourth_Moddel.ipynb  
├── fifth_model.ipynb  
└── README.md  

Each notebook is self-contained and can be executed independently.

---

## Model Performance Summary

| Model No. | Notebook Name            | Description                               | Validation Accuracy |
|----------|--------------------------|-------------------------------------------|---------------------|
| 2        | second-aiforagri.ipynb   | Baseline multimodal model                 | 0.58947             |
| 3        | Third_model.ipynb        | Improved spectral feature learning        | 0.61052             |
| 4        | Fourth_Moddel.ipynb      | CNN ensemble (RGB + MS + HS)              | 0.65263             |
| 5        | fifth_model.ipynb        | Deep multimodal fusion (final model)      | 0.68421  (best)     |

The accuracy increases monotonically across models.

---

## Model Descriptions

### Model 2: Baseline Multimodal Model  
**Notebook:** second-aiforagri.ipynb  
**Accuracy:** 0.58947  

This model serves as the baseline for comparison.

- Basic preprocessing for RGB, MS, and HS images  
- Standard CNN-based feature extraction  
- Minimal hyperparameter tuning  
- Limited interaction between modalities  

This model establishes a reference point for further improvements.

---

### Model 3: Improved Spectral Learning  
**Notebook:** Third_model.ipynb  
**Accuracy:** 0.61052  

This model improves upon the baseline by better utilizing spectral information.

- Enhanced handling of hyperspectral bands  
- Improved feature extraction for spectral–spatial patterns  
- Reduced information loss compared to simple band aggregation  

The model shows a clear improvement, demonstrating the importance of spectral features.

---

### Model 4: CNN Ensemble Model  
**Notebook:** Fourth_Moddel.ipynb  
**Accuracy:** 0.65263  

This model introduces an ensemble-style approach.

- Separate CNN backbones for RGB, MS, and HS inputs  
- Independent feature extraction per modality  
- Combined predictions to improve generalization  
- Use of data augmentation and GPU acceleration  

This model significantly improves robustness and accuracy.

---

### Model 5: Final Model – Deep Multimodal Fusion  
**Notebook:** fifth_model.ipynb  
**Accuracy:** Highest (Best Performing Model)  

This is the final and most advanced model in the repository.

#### Architecture Overview

Each modality is processed through a dedicated deep backbone:

- RGB images use EfficientNet-B0  
- Multispectral images use ResNet-18  
- Hyperspectral images use ResNet-18  

Each branch produces a 512-dimensional feature vector.

#### Feature Fusion

- Features from RGB, MS, and HS branches are concatenated  
- Combined feature vector size: 1536  
- Passed through fully connected layers for classification  

#### Training Configuration

- Loss function: Cross-Entropy Loss  
- Optimizer: Adam  
- Mixed precision training enabled  
- 5-Fold Stratified Cross-Validation  
- Test-Time Augmentation (horizontal and vertical flips)  

#### Why This Model Performs Best

- Deep feature-level fusion instead of shallow averaging  
- Strong modality-specific representation learning  
- Efficient handling of spectral and spatial information  
- Robust training strategy with cross-validation and TTA  

This model represents a production-ready multimodal learning pipeline.

## Notes

- All experiments were conducted using GPU acceleration on Kaggle  
- The code is organized for clarity and reproducibility  
- This work is part of an ongoing research effort for ICPR 2026  
