# RGCR: Reliability-Gated Fusion with Confusion-aware Refinement

This repository contains the implementation of RGCR, a framework for multimodal emotion recognition in conversations. 

## 🖼️ Model Architecture

<p align="center">
  <img src="RGCR.jpg" alt="Framework" width="80%">
</p>


## 🔧 Data Preparation

- Download the preprocessed multimodal features from [Google Drive](https://drive.google.com/drive/folders/1TwT9z6N6SJadsVkDhSNVBiF9ZygEyA6l).
- Put the two feature files into the `data/` directory:

```text
data/iemocap_multimodal_features.pkl
data/meld_multimodal_features.pkl
```

The features are pre-extracted and publicly released by the authors of [SDT](https://github.com/butterfliesss/SDT?tab=readme-ov-file). We thank them for making this resource available.

## ⚡ Environment

- Python >= 3.8
- PyTorch >= 2.3
- numpy >= 1.26
- pandas >= 2.2
- scipy >= 1.11
- scikit-learn >= 1.4

## 🚀 Run

```bash
pip install -r requirements.txt

# Train on IEMOCAP
python train_IEMOCAP.py

# Train on MELD
python train_MELD.py
```

## 👍 Contributing

We welcome contributions and suggestions!
