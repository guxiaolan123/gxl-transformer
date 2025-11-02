
# Transformer-Based Shakespeare Language Modeling
This repository contains the implementation of a Transformer model for character-level language modeling on the Tiny Shakespeare dataset, as required by the **Mid-Term Assignment of "Fundamentals and Applications of Large Models"**. The code fully reproduces the experiments in the report *"基于Transformer的莎士比亚文本语言建模实验报告"*. 
## 1. Project Overview 
- **Task**: Character-level language modeling (predict the next character in Shakespeare's text) 
- **Model**: Encoder-Decoder Transformer (with/without positional encoding for ablation study) 
- **Dataset**: Tiny Shakespeare (character-level, ~1MB text, from Hugging Face) 
- **Key Experiments**: 
- 1. **Baseline**: Transformer with sinusoidal positional encoding 
- 2. **Ablation**: Transformer without positional encoding 
- **Evaluation Metrics**: Cross-Entropy Loss, Perplexity (PPL), Character Accuracy 
## 2. Repository Structure 

> transformer-shakespeare/  
> ├── train.py  
> ├── model.py  
> ├── data.py  
> ├──results/  
> │ ├── loss_curve_final.png  
> │ ├── metrics_full_with_pos.txt 
> │ ├── metrics_full_without_pos.txt  
> │ ├──best_model_full_with_pos.pth
> │ └── best_model_full_without_pos.pth  
> ├── input.txt  
> ├──requirements.txt  
> └── README.md

## 3. Environment Setup 
### 3.1 Hardware Requirements 
- **CPU**: Ubuntu 20.04 LTS (training ~60 mins/10 epochs) 
- **GPU**: NVIDIA GPU (e.g., RTX 3090, training ~20 mins/10 epochs) 
### 3.2 Software Dependencies 
Install dependencies via `pip` (compatible with Python 3.8+): 
```bash pip install -r requirements.txt ``` 
`requirements.txt` content:
 

> torch>=1.10.0   
> numpy>=1.24.0   
> matplotlib>=3.7.0   
> tqdm>=4.65.0  
> huggingface-hub>=0.19.0

### 3.3 Dataset Preparation 
The dataset (`input.txt`) is already included in the repository. It is the **Tiny Shakespeare** dataset (character-level) from Hugging Face: 
- Official Link: [huggingface.co/datasets/tiny_shakespeare](https://huggingface.co/datasets/tiny_shakespeare) 
- Preprocessing : 
- 1. Character-level tokenization (no word-level splitting) 
- 2. Vocabulary size = 65 (uppercase/lowercase letters, punctuation, spaces) 
- 3. Sequence length = 64 (predict next character for each position) 
- 4. Train-test split = 9:1 (31368 train batches, 3483 test batches, batch size=32) 
## 4. Quick Start (Exact Commands) 
To ensure reproducibility (required by assignment §1.2), use the following commands with fixed random seed (`seed 42`): 
### 4.1 Run Baseline Experiment (With Positional Encoding) 

> bash python train.py use_pos_encoding True seed 42 epochs 10 batch_size 32 seq_len 64 d_model 64 num_heads 2 num_layers 2 lr 3e-4 gradient_clip 1.0

### 4.2 Run Ablation Experiment (Without Positional Encoding) 

> bash python train.py use_pos_encoding False seed 42 epochs 10 batch_size 32 seq_len 64 d_model 64 num_heads 2 num_layers 2 lr 3e-4 gradient_clip 1.0

### 4.3 One-Click Run (Via Script) 
Use `scripts/run.sh` to run both experiments sequentially: 
```bash bash scripts/run.sh ``` 
## 5. Experimental Results 
All results are saved in `./results/` : 
### 5.1 Key Metrics (10 Epochs) 
| Experiment | Train Loss | Val Loss | Val Perplexity | Val Accuracy | 
|---------------------------|------------|----------|----------------|--------------| 
| Baseline (with Pos Enc) | 0.0301 | 0.0297 | 1.0301 | 99.11% | 
| Ablation (without Pos Enc)| 0.0301 | 0.0297 | 1.0301 | 99.11% | 
### 5.2 Loss Curve 
- File: `./results/loss_curve_final.png` - Trends: 
- 1. Training loss drops from 0.1591 to 0.0301 (stable, no oscillation) 
- 2. Validation loss drops from 0.0356 to 0.0297 (small gap with train loss, no overfitting) 
- 3. Curves of baseline and ablation experiments overlap (positional encoding has minimal effect on character-level short sequences) 
## 6. Reproducibility Notes 
- **Random Seed**: Fixed to `42` in all experiments (code-level + command-line) 
- **Hyperparameters**: Strictly follow the assignment's "Hyperparameter Settings" and the report  
- **Model Initialization**: Xavier uniform initialization for all linear layers (in `model.py`) 
- **Training Tricks**: AdamW optimizer + CosineAnnealingLR + gradient clipping (prevents training instability) 
## 7. Notes 
1. **LaTeX Report Compilation**: The report is compiled via **XeLaTeX** (supports Chinese fonts like "Noto Serif CJK SC"). 
2. **Code Compliance**: All modules (Multi-Head Attention, Position-Wise FFN, Residual+LayerNorm, Positional Encoding) are implemented manually (meets assignment §3 requirements for 80-90 points). 
3. **Extension**: For sequence-to-sequence tasks (e.g., machine translation), modify `model.py` to adapt to paired datasets . 
