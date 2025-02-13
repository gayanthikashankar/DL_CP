# Deep Learning Image Classification Project

## Overview
This project implements image classification using deep learning on a dataset containing images of Airplanes, Motorbikes, and Schooners. The implementation explores both Transfer Learning and Fine-Tuning approaches using three pre-trained CNN models.

## Dataset
- Split ratio: 75% training, 25% testing
- Classes: Airplanes, Motorbikes, Schooners
- Validation split: 10% of training data

## Models
Three pre-trained CNN architectures are implemented:

### 1. Xception
- **Transfer Learning Modifications:**
  - Added Batch Normalization
  - Dropout layer (25% drop rate)
  - Training: 10 epochs
- **Fine-Tuning Configuration:**
  - Initial 25% layers: Non-trainable
  - Remaining layers: Trainable
  - Training: 10 epochs

### 2. ResNet101V2
- **Transfer Learning Modifications:**
  - Added Batch Normalization
  - Dropout layer (35% drop rate)
  - Training: 10 epochs
- **Fine-Tuning Configuration:**
  - Initial 35% layers: Non-trainable
  - Remaining layers: Trainable
  - Training: 10 epochs

### 3. InceptionResNetV2
- **Transfer Learning Modifications:**
  - Dropout layer (15% drop rate)
  - Training: 10 epochs
- **Fine-Tuning Configuration:**
  - All layers set as trainable
  - Training: 10 epochs

## Project Structure
```
project/
├── Notebooks_Source/
│   ├── Model1_TL.ipynb
│   ├── Model1_FT.ipynb
│   ├── Model2_TL.ipynb
│   ├── Model2_FT.ipynb
│   ├── Model3_TL.ipynb
│   └── Model3_FT.ipynb
├── Notebooks_PDFs/
│   ├── Model1_TL.pdf
│   ├── Model1_FT.pdf
│   ├── Model2_TL.pdf
│   ├── Model2_FT.pdf
│   ├── Model3_TL.pdf
│   └── Model3_FT.pdf
└── README.md
```

## Implementation Details
- Each model has separate notebooks for Transfer Learning (TL) and Fine-Tuning (FT)
- Fine-tuning notebooks load the best performing TL models from Google Drive
- GPU runtime enabled in Google Colab
- Best models are preserved using callbacks

## Model Evaluation
Each fine-tuned model is evaluated using:
1. Confusion Matrix
2. Precision
3. Recall
4. F1-Score

## Technical Requirements
- Python 
- TensorFlow 
- Google Colab (GPU runtime)
- Required libraries: numpy, pandas, matplotlib, seaborn

## Usage
1. Upload the dataset to Google Drive
2. Mount Google Drive in Colab notebooks
3. Run Transfer Learning notebooks first (Model#_TL.ipynb)
4. Run Fine-Tuning notebooks (Model#_FT.ipynb) after corresponding TL models are trained
