# Beyond Visible Spectrum: AI for Agriculture  
## Multimodal Crop Disease Classification (ICPR 2026)

This repository presents a structured progression of deep learning models developed for multimodal crop disease classification using RGB, Multispectral (MS), and Hyperspectral (HS) imagery.  
Each model is saved as a separate notebook and named according to its model number.  
The models are designed incrementally, with accuracy improving at each stage.

The objective of this work is to study how architectural complexity and modality fusion improve classification performance in agricultural datasets.

---

## Repository Structure

├── baseline_model.ipynb  
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
| 1        | first_model.ipynb        | Baseline model (reference implementation) | 0.53684             |     
| 2        | second-aiforagri.ipynb   | First improved multimodal model           | 0.58947             |
| 3        | Third_model.ipynb        | Improved spectral feature learning        | 0.61052             |
| 4        | Fourth_Moddel.ipynb      | CNN ensemble (RGB + MS + HS)              | 0.65263             |
| 5        | fifth_model.ipynb        | Deep multimodal fusion (final model)      | Highest (Best)      |

---

## Model Descriptions

### Model 1: Baseline Model  
**Notebook:** first_model.ipynb  
**Accuracy:** 0.53684

This model serves as the baseline reference for all subsequent experiments.

 - Three-Stage Training Pipeline: The model is trained incrementally: Stage 1 focuses on Multispectral (MS) features, Stage 2 on Hyperspectral (HS) features, and Stage 3 performs late-fusion of the pretrained encoders.
 - Spectral Feature Engineering
 - Targeted Hyperspectral Processing
 - Late-Fusion Architecture
 - Advanced Training Techniques

---

### Model 2: First Improved Multimodal Model  
**Notebook:** second-aiforagri.ipynb  
**Accuracy:** 0.58947  

This model builds upon the baseline and introduces a basic multimodal learning pipeline.

- Basic preprocessing for RGB, MS, and HS images  
- Standard CNN-based feature extraction  
- Minimal hyperparameter tuning  
- Limited interaction between modalities  

This model establishes the first measurable improvement over the baseline.

---

### Model 3: Improved Spectral Learning  
**Notebook:** Third_model.ipynb  
**Accuracy:** 0.61052  

This model improves upon Model 2 by better utilizing spectral information.

- Enhanced handling of hyperspectral bands  
- Improved feature extraction for spectral–spatial patterns  
- Reduced information loss compared to simple band aggregation  

---

### Model 4: CNN Ensemble Model  
**Notebook:** Fourth_Moddel.ipynb  
**Accuracy:** 0.65263  

This model introduces an ensemble-style approach.

- Separate CNN backbones for RGB, MS, and HS inputs  
- Independent feature extraction per modality  
- Combined predictions to improve generalization  
- Use of data augmentation and GPU acceleration  

---

### Model 5: Final Model – Deep Multimodal Fusion  
**Notebook:** fifth_model.ipynb  
**Accuracy:** Highest (Best Performing Model)  

This is the final and most advanced model in the repository.

#### Architecture Overview

- RGB branch: EfficientNet-B0  
- Multispectral branch: ResNet-18  
- Hyperspectral branch: ResNet-18  

Each branch produces a 512-dimensional feature vector.

#### Feature Fusion

- Feature concatenation across all modalities  
- Combined feature vector size: 1536  
- Fully connected layers for final classification  

#### Training Configuration

- Loss function: Cross-Entropy Loss  
- Optimizer: Adam  
- Mixed precision training enabled  
- 5-Fold Stratified Cross-Validation  
- Test-Time Augmentation (horizontal and vertical flips)  

#### Why This Model Performs Best

- Deep feature-level fusion  
- Strong modality-specific representation learning  
- Effective use of spectral and spatial information  
- Robust training and evaluation strategy  

## Notes

- All experiments were conducted using GPU acceleration on Kaggle  
- The code is organized for clarity and reproducibility  
- This work is part of an ongoing research effort for ICPR 2026  
