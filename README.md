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
 
## Data

- **Source**: The dataset used here is SMILES from PubChem (62,129,800 valid SMILES strings)

### Preprocessing

   - Libraries: `gzip`, `numpy`, `pickle`
   - Map SMILES characters to integer indices using a reduced 28-character vocabulary.
   - Special tokens:
     - `^` = Start of sequence (not used in generation)
     - `$` = End of sequence
   - All sequences are padded to a fixed `max_length` (e.g., 150) with zeros.
   - Only molecules with SMILES shorter than a given threshold (`max_length`) are kept.
   - A NumPy array is created to hold the filtered, tokenized SMILES strings
   - The final dataset is saved as `train_data.npy` using `np.save()`.

> Note: Filtering thresholds like `max_length` may be adjusted during experimentation (e.g., 20, 50, 150). Found is the preproccessing.ipynb


  ## Development Environment Requirements

- **Python**: 3.10+ 
- **PyTorch**: 2.6.0 with CUDA 12.4 (for GPU acceleration)  
- **RDKit**: 2024.03.5 (installed via `conda-forge`)  
- **Libraries**:  
  - `torchvision`, `torchaudio`  
  - `scikit-learn`, `pandas`, `NumPy`, `Matplotlib`  
  - `wandb` (Weights & Biases) for experiment tracking  
  - `jupyterlab` for interactive development  
- Conda (recommended: Miniconda or Anaconda)
- All dependencies are specified in the Conda environment file: `smiles_vae_env.yml`

## Installation

### 1. Clone the repository:
 
 ```bash
git clone https://github.com/Blake-De/smiles-vae.git
```

Navigate into the project directory:

```bash
cd smiles-vae
```

### 2. Create and activate the conda environment:

```bash
conda env create -f smiles_vae_env.yml
conda activate smiles-vae
```

## Usage

### Train the model
```bash
python smiles_vae.py --train_data data/smiles_train.npy --out smiles_vae_model.pth
```

### Command-Line Arguments

| Argument           | Type   | Description                                           | Default              |
|--------------------|--------|-------------------------------------------------------|----------------------|
| `--train_data`     | str    | Path to the training data file (e.g., `.npy`)         | **(required)**       |
| `--out`            | str    | Path to save the exported decoder model               | `vae_generate.pth`   |
| `--batch_size`     | int    | Number of SMILES strings per batch                    | `512`                |
| `--epochs`         | int    | Number of training epochs                             | `5`                  |
| `--lr`             | float  | Learning rate                                         | `1e-4`               |
| `--embedding_dim`  | int    | Dimensionality of SMILES token embeddings             | `20`                 |
| `--hidden_size`    | int    | Size of GRU hidden layers                             | `1024`               |
| `--num_layers`     | int    | Number of GRU layers                                  | `1`                  |
| `--kl_weight`      | float  | Weight applied to KL divergence in the loss function  | `0.001`              |
| `--max_length`     | int    | Maximum SMILES string length                          | `150`                |
| `--evals`          | int    | Number of molecules to generate for evaluation        | `1000`               |

## Evaluation Metrics

- **UniqueSMI**: Number of unique molecules
- **ValidSMI**: Number of valid SMILES strings generated  
- **UniqueValidMols**: Number of unique valid molecules  
- **NovelMols**: Number of molecules not found in the training set  
- **AveRings**: Average number of rings (proxy for chemical complexity)

## Output

- **Traced Decoder**: Saved as `.pth` using TorchScript for efficient inference and sampling  
- **Console Output**: During training, the console prints:
  - Training loss, reconstruction loss, and KL divergence at log intervals  
  - Evaluation metrics at model save checkpoints  
- **Model Checkpoints**: Decoder is saved periodically during training  
- **WandB Logging**: If enabled, logs all training and evaluation metrics in real time

### Model Performance

The model generally preformace well near the defualts. 
These are results from epoch 10 by evaluating the model on 1,000 samples from the latent space. This model was ceated with command: 

```bash
python smiles_vae.py \
  --train_data data/smiles_train.npy \
  --out smiles_vae_model.pth \
  --epochs 18 \
  --num_layers 2 \
  --kl_weight 0.01	
```

| Metric             | Value   |
|--------------------|---------|
| **UniqueSMI**       | 1000    |
| **ValidSMI**        | 839     |
| **AveRings**        | 2.8081  |
| **UniqueValidMols** | 839     |
| **NovelMols**       | 836     |

## Project Structure

```bash
cell-moa-classifier/
├── train_smiles_vae.py         # Main training script  
├── smiles_vae_env.yml          # Conda environment file  
├── README.md                   # Project documentation  
├── .gitignore                  # Git ignore rules
├── preprocessing.ipynb         # Notebook used to prepare and filter SMILES
├── autoencoder.py              # Trained for to confirm reconstruction error
└── model.pth                   # Trained model output (not tracked in Git)
```

## Author & Acknowledgments

Blake Degioanni  
[GitHub](https://github.com/Blake-De) • [LinkedIn](https://www.linkedin.com/in/blake-degioanni)

This project was completed as a self-directed assignment for a graduate-level machine learning course at the University of Pittsburgh,  
with a focus on real-world applications of generative modeling.

## Notes

The preprocessing script was mostly provided by the course instructor. Filtering and dataset slicing were used for training speed and can be found in the preprocessing.npy.
