# smiles-vae

This project implements a Variational Autoencoder (VAE) using PyTorch and generates molecular SMILES strings. 

It was developed as part of a ML course assignment, but demonstrates real-world generative modeling skills.

## Overview

- **Baseline:** Standard Autoencoder trained first to confirm basic reconstruction
- **Model:** Variational Autoencoder (VAE)
- **Input:** Tokenized SMILES strings (PubChem filtered subset)
- **Output:** NovelMols, valid molecular structures
- **Framework:** PyTorch (2.6.0), GPU-accelerated
- **Latent Dimension:** 1024

The model encodes tokenized molecular SMILES strings into a latent space, then decodes samples from this space into new molecules.

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

## Notes
The preprocessing script was provided by the course instructor; filtering and dataset slicing were adapted for training speed.

Full PubChem-derived dataset not included — place .npy or filtered .smi.gz files in /data if rerunning.

## Files
File: Description  
- train_smiles_vae.py: Main training script  
- preprocessing.ipynb: Notebook used to prepare and filter SMILES
- autoencoder.py: Trained for to confirm reconstruction error
- smiles_vae_env.yml: Conda env

## Acknowledgments
Based on a University of Pittsburgh assignment.

