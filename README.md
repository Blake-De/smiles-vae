# smiles-vae

This project implements a Variational Autoencoder (VAE) using PyTorch and generates molecular SMILES strings. 

## Overview

This project implements a Variational Autoencoder (VAE) in PyTorch that encodes tokenized molecular SMILES strings into a latent space using a GRU-based encoder, then decodes samples from this space into new molecules using a GRU-based decoder. 

### Key Features:
- **Baseline:** Standard Autoencoder trained first to confirm basic reconstruction
- **Model Architecture:**
  - **Encoder:** GRU encoder that maps tokenized SMILES strings to a 1024-dimensional latent space
  - **Decoder:** GRU decoder that autoregressively reconstructs SMILES strings from latent samples
- **Training Strategy:** Teacher forcing with cross-entropy loss for reconstruction and KL divergence for regularization
- **Input:** Tokenized SMILES strings (PubChem filtered subset)
- **Export:** Decoder exported as a TorchScript module 
- **Output:** NovelMols — valid and unique molecular structures
- **Framework:** PyTorch (2.6.0), GPU-accelerated
- **Latent Space:** 1024-dimensions

## Metrics Reported
- **ValidSMI**: # of valid SMILES generated
- **UniqueValidMols**: # of unique valid molecules
- **NovelMols**: # not found in training set
- **AveRings**: Average ring count (chemical complexity)

### Model Performance

These are results from evaluating the model on 1,000 samples from the latent space (as per assignment rubric):

- **UniqueSMI**: 1000
- **ValidSMI**: 839	
- **AveRings**: 2.8081
- **UniqueValidMols**: 839
- **NovelMols**: 836


## Usage

### 1. Create and activate the environment
```bash
conda env create -f smiles_vae_env.yml
conda activate smiles-vae
```
### 2. Train the model
```bash
python smiles_vae.py --train_data data/smiles_train.npy --out smiles_vae_model.pth
```

### Example args (adjust as needed)
```bash
--batch_size 512 --epochs 10 --kl_weight 0.001 --embedding_dim 20 --hidden_size 1024
```

### Output

- Traced decoder saved as .pth for evaluation
- Weights & Biases logs (if enabled)
- Evaluation printed for valid, unique, and novel molecules



## Files
File: Description  
- train_smiles_vae.py: Main training script  
- preprocessing.ipynb: Notebook used to prepare and filter SMILES
- autoencoder.py: Trained for to confirm reconstruction error
- smiles_vae_env.yml: Conda env

## Author & Acknowledgments

Blake Degioanni  
[GitHub](https://github.com/Blake-De) • [LinkedIn](https://www.linkedin.com/in/blake-degioanni)

This project was completed as a self-directed assignment for a graduate-level machine learning course at the University of Pittsburgh,  
with a focus on real-world applications of generative modeling.

## Notes

The preprocessing script was mostly provided by the course instructor. Filtering and dataset slicing were used for training speed and can be found in the preprocessing.npy.
