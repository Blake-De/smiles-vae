"""This script trains a SMILES-generating VAE using PyTorch and saves a torch.jit decoder for sampling. Expects an npy for the data."""

#!/usr/bin/env python3
import gzip
import torch
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import init
import argparse
import time
import wandb
from rdkit.Chem import AllChem as Chem
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")


class Lang:
    """Predefined mapping from characters to indices for our
    reduced alphabet of SMILES with methods for converting.
    You must use this mapping."""

    def __init__(self):
        # $ is the end of sequence token
        # ^ is the start of sequence token, which should never be generated
        self.chartoindex = {
            "$": 0,
            "^": 1,
            "C": 2,
            "(": 3,
            "=": 4,
            "O": 5,
            ")": 6,
            "[": 7,
            "-": 8,
            "]": 9,
            "N": 10,
            "+": 11,
            "1": 12,
            "P": 13,
            "2": 14,
            "3": 15,
            "4": 16,
            "S": 17,
            "#": 18,
            "5": 19,
            "6": 20,
            "7": 21,
            "H": 22,
            "I": 23,
            "B": 24,
            "F": 25,
            "8": 26,
            "9": 27,
        }
        self.indextochar = {
            0: "$",
            1: "^",
            2: "C",
            3: "(",
            4: "=",
            5: "O",
            6: ")",
            7: "[",
            8: "-",
            9: "]",
            10: "N",
            11: "+",
            12: "1",
            13: "P",
            14: "2",
            15: "3",
            16: "4",
            17: "S",
            18: "#",
            19: "5",
            20: "6",
            21: "7",
            22: "H",
            23: "I",
            24: "B",
            25: "F",
            26: "8",
            27: "9",
        }
        self.nchars = 28

    def indexesFromSMILES(self, smiles_str):
        """convert smiles string into numpy array of integers"""
        index_list = [self.chartoindex[char] for char in smiles_str]
        index_list.append(self.chartoindex["$"])
        return np.array(index_list, dtype=np.uint8)

    def indexToSmiles(self, indices):
        """convert list of indices into a smiles string"""
        smiles_str = "".join(
            list(map(lambda x: self.indextochar[int(x)] if x != 0.0 else "E", indices))
        )
        return smiles_str.split("E")[
            0
        ]  # Only want values before output $ end of sequence token


class SmilesDataset(torch.utils.data.Dataset):
    """Dataset that reads in a preprocessed .npy smiles array.
    Each row is a fixed-length tokenized SMILES string. Note we encountered memory usage issues
    when using variable sequence length batches and so use a fixed size.
    There are likely more memory efficient ways to store this data.
    """

    def __init__(self, data_path, max_length):
        self.max_length = max_length
        self.language = Lang()
        self.examples = np.load(data_path)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return torch.tensor(self.examples[idx], dtype=torch.long)

    def getIndexToChar(self):
        return self.language.indextochar


class Encoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers):
        super(Encoder, self).__init__()
        self.latent_dim = 1024

        # Embedding layer converts token indices to dense vectors
        # Input shape:  (B, L)
        # Output shape: (B, L, embedding_dim)
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # GRU processes the embedded sequence
        # Input:  (B, L, embedding_dim)
        # Output: (B, L, hidden_size)
        # hidden output: (num_layers, B, hidden_size)
        self.gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,  # Expects input: (B, L, embedding_dim)
        )

        # Projection to latent dim if needed to match the required latent size
        # z:        (B, latent_dim)
        self.mean = nn.Linear(num_layers * hidden_size, self.latent_dim)
        self.logv = nn.Linear(num_layers * hidden_size, self.latent_dim)

        # Encoder Setup
        self.hidden_size = hidden_size
        self.num_layers = num_layers

    def forward(self, input_seq):
        # input_seq: (B, L) — batch of token indices
        # embedded: (B, L, embedding_dim)
        embedded = self.embedding(input_seq)

        # embedded: (B, L, embedding_dim)
        # output:   (B, L, hidden_size): hidden states for every time step
        # hidden:   (num_layers, B, hidden_size): hidden state at final time step for each layer
        output, hidden = self.gru(embedded)

        # Flatten all GRU layers into a single z vector per sample
        # input_seq: (B, L)
        B = input_seq.size(0)

        # Hidden:   (num_layers, B, hidden_size)
        # Tranpsoe: (B, num_layers, H)
        # View:     (B, num_layers * H)
        hidden = hidden.transpose(0, 1).contiguous().view(B, -1)

        # Hidden:   (B, num_layers * H)
        # z;        (B, latent_dim)
        mean = self.mean(hidden)
        logv = self.logv(hidden)

        return mean, logv


class Decoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers, max_length):
        super(Decoder, self).__init__()
        self.latent_dim = 1024

        # z: (B, latent_dim)
        # Output: (B, num_layers * hidden_size)
        self.latent_to_hidden = nn.Linear(self.latent_dim, hidden_size * num_layers)

        # Embedding layer converts token indices to dense vectors
        # Input shape:  (B, L)
        # Output shape: (B, L, embedding_dim)
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # GRU processes the embedded sequence
        # Input:  (B, L, embedding_dim)
        # Output: (B, L, hidden_size)
        # hidden output: (num_layers, B, hidden_size)
        self.gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,  # Expects input: (B, L, embedding_dim)
        )

        # Maps GRU output at each step to vocab distribution
        self.output_layer = nn.Linear(hidden_size, vocab_size)

        # Decoder Setup
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.max_length = max_length

    def forward(self, z, actual_input=None):
        batch_size = z.size(0)

        # Project latent vector to match GRU hidden state
        # z: (B, latent_dim)
        # Output: (B, num_layers * hidden_size)
        hidden = self.latent_to_hidden(z)

        # Initialize GRU hidden state using latent vector
        # hidden:         (B, num_layers * H)
        # reshape:   (B, num_layers, H)
        # Tranpsoe:  (num_layers, B, hidden_size)
        # hidden:    (num_layers, B, hidden_size)
        hidden = hidden.reshape(
            batch_size, self.num_layers, self.hidden_size
        ).transpose(0, 1)
        hidden = hidden.contiguous()

        # Intilize t none
        logits = None
        generated_sequence = None

        # Training teacher forcing
        if actual_input is not None:

            # Start with ^ token (index 1)
            input_token = torch.full(
                (batch_size, 1), 1, dtype=torch.long, device=z.device
            )
            actual_input = torch.cat([input_token, actual_input[:, :-1]], dim=1)

            # actual_input: (B, L) — batch of token indices
            # embedded: (B, L, embedding_dim)
            embedded = self.embedding(actual_input)

            # embedded: (B, L, embedding_dim)
            # output:   (B, L, hidden_size): hidden states for every time step
            # hidden:   (num_layers, B, hidden_size): hidden state at final time step for each layer
            output, hidden = self.gru(embedded, hidden)

            # Map GRU output to vocab space at each time step
            # output:   (B, L, hidden_size): hidden states for every time step
            # logits: (B, L, vocab_size)
            logits = self.output_layer(output)

        # Generation mode (no teacher forcing)
        else:
            # Sample one token at a time from the decoder
            generated = []

            # Start with ^ token (index 1)
            input_token = torch.full(
                (batch_size, 1), 1, dtype=torch.long, device=z.device
            )

            # one token at a time
            for _ in range(self.max_length):
                # Embedding layer converts token indices to dense vectors
                # Input shape:  (B, L)
                # Output shape: (B, L, embedding_dim)
                embedded = self.embedding(input_token)

                # embedded:     (B, L, embedding_dim)
                # Input hidden: (num_layers, B, hidden_size)
                # output:        (B, L, hidden_size): hidden states for every time step
                # hidden:        (num_layers, B, hidden_size): hidden state at final time step for each layer
                output, hidden = self.gru(embedded, hidden)

                # Map GRU output to vocab space at each time step
                # output:   (B, L, hidden_size): hidden states for every time step
                # sqeeuze logits: (B, vocab_size)
                logits = self.output_layer(output.squeeze(1))

                # prob
                # sqeeuze logits: (B, vocab_size)
                # next_token shape: (B, 1) or (B, L)
                # Convert logits to probabilities
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                generated.append(next_token)
                input_token = next_token

            # (B, L)
            generated_sequence = torch.cat(generated, dim=1)

        # logits:      (B, L, vocab_size) logits if actual_input is provided
        # generated:  (B, L) token indices if actual_input is None
        return logits, generated_sequence


class VAE(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers, max_length):
        super(VAE, self).__init__()

        self.latent_dim = 1024

        # Enocder decoder classes
        self.encoder = Encoder(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
        )

        self.decoder = Decoder(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            max_length=max_length,
        )

    def reparameterize(self, mean, logv):
        # reparmaterize
        # Sample from the latnet space
        std = torch.exp(0.5 * logv)
        eps = torch.randn_like(std)
        return mean + eps * std

    def forward(self, input_seq):
        # input_seq:           (B, L)
        # mean and log:        (B, num_layers * hidden_size)
        mean, logv = self.encoder(input_seq)

        # Sample from latent space
        z = self.reparameterize(mean, logv)

        # input_seq:                (B, L)
        # z:                      (B, num_layers * hidden_size)
        # logits                (B, L, V)
        # genrated sequance:   (B, L)
        logits, generated_sequence = self.decoder(z, input_seq)

        return logits, generated_sequence, mean, logv


def smilesToStatistics(list_of_smiles, trainsmi):
    """Return number valid smiles, number of unique molecules, and average number of rings"""
    count_molecules = 0
    cannonical_smiles = set()
    ringcnt = 0
    novelcnt = 0
    for smiles in list_of_smiles:

        # Skip if smiles is empty.
        if smiles == "":
            continue
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                # Add one to valid.
                count_molecules += 1
                can = Chem.MolToSmiles(mol)
                if can not in cannonical_smiles:
                    # print(can)
                    cannonical_smiles.add(can)
                    r = mol.GetRingInfo()
                    ringcnt += r.NumRings()
                    if can not in trainsmi:
                        novelcnt += 1

        except:
            continue

    # Unique smiles
    N = len(cannonical_smiles)
    ringave = 0 if N == 0 else ringcnt / N

    return count_molecules, N, novelcnt, ringave


if __name__ == "__main__":
    LATENT_DIM = 1024

    # arugment parsing
    parser = argparse.ArgumentParser("Train a Variational Autoencoder")
    parser.add_argument(
        "--train_data", "-T", required=True, help="data to train the VAE with"
    )
    parser.add_argument(
        "--out", default="vae_generate.pth", help="File to save generate function to"
    )
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--embedding_dim", type=int, default=20)
    parser.add_argument("--hidden_size", type=int, default=1024)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--kl_weight", type=float, default=0.001)
    parser.add_argument(
        "--max_length", type=int, default=150, help="Maximum SMILES length"
    )
    parser.add_argument("--evals", default=1000, type=int, help="Number of evaluations")

    # Wanab logging or sweeping is config.
    args = parser.parse_args()
    wandb.init(
        project="ML_VAE", name=f"{args.out.rsplit('.', 1)[0]}", config=vars(args)
    )

    args = wandb.config

    dataset = SmilesDataset(args.train_data, args.max_length)

    language_mapping = dataset.getIndexToChar()

    # Load the data
    # Output: (B, L)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
    )

    # Make model.
    vae = VAE(
        hidden_size=args.hidden_size,
        vocab_size=len(language_mapping),
        embedding_dim=args.embedding_dim,
        num_layers=args.num_layers,
        max_length=args.max_length,
    ).to("cuda")

    # Weight
    optimizer = torch.optim.Adam(vae.parameters(), lr=args.lr)

    # TRAIN THE MODEL
    train_start = time.time()
    log_interval = 1100
    save_interval = 15000
    lang = Lang()
    trainsmi = set()

    for epoch in range(args.epochs):
        vae.train()

        # Track the time
        epoch_start = time.time()

        for batch_idx, batch in enumerate(dataloader):
            batch = batch.to("cuda")

            # Call the model
            # batch:    (B,  L)
            # logits:   (B, L, vocab_size)
            # gnerated sequence: (B, L)
            # mean and log: (B, latent_dim)
            logits, generated_sequence, mean, logv = vae(batch)

            # batch:      (B,  L)
            # logits:     (B, L, V)
            # First view: (B * L, V)
            # Last view:  (B * L)
            # calcuating loss
            recon_loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), batch.view(-1)
            )

            # KL LOss and full loss
            kl_loss = -0.5 * torch.mean(1 + logv - mean.pow(2) - logv.exp())
            loss = recon_loss + args.kl_weight * kl_loss

            # loss and optimizing
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Printing the loss for batch
            # Print every `log_interval` batches
            if batch_idx % log_interval == 0:
                print(
                    f"[Epoch {epoch+1} | Batch {batch_idx}] "
                    f"Total Loss: {loss.item():.4f} | "
                    f"Recon: {recon_loss.item():.4f} | KL: {kl_loss.item():.4f}"
                )

                wandb.log(
                    {
                        "total_loss": loss.item(),
                        "recon_loss": recon_loss.item(),
                        "kl_loss": kl_loss.item(),
                        "epoch": epoch + 1,
                        "batch": batch_idx,
                    }
                )

            if batch_idx % save_interval == 0:
                # This will create the file.
                # generated from a normal distribution. Note that vae.decoder must
                # be a Module
                z_1 = torch.normal(0, 1, size=(1, LATENT_DIM), device="cuda")
                with torch.no_grad():
                    vae.decoder.eval()
                    # torch.multinomial is not deterministic so we disable trace checking
                    traced = torch.jit.trace(
                        vae.decoder, z_1.to("cuda"), check_trace=False
                    )

                    torch.jit.save(
                        traced,
                        f"{args.out.rsplit('.', 1)[0]}_epoch_{epoch+1}_batch{batch_idx+1}.pth",
                    )

                    # This will evalauted the model at given point in time for
                    # Unique smile, valid smiles, and average ring score.
                    generated_smiles = []

                    for i in range(args.evals):
                        z_samples = torch.normal(
                            0, 1, size=(1, LATENT_DIM), device="cuda"
                        )

                        # Use the decoder to generate the token sequence.
                        _, output_smile = vae.decoder(z_samples)

                        # Convert tokens to SMILES and append.
                        generated_smiles.append(lang.indexToSmiles(output_smile[0]))

                    # Evaluate the generated SMILES.
                    valid_smiles, unique_mols, novelcnt, ringave = smilesToStatistics(
                        generated_smiles, trainsmi
                    )

                    print("UniqueSmiles", len(set(generated_smiles)))
                    print("ValidSmiles", valid_smiles)
                    print("UniqueAndValidMols", unique_mols)
                    print("AverageRings", ringave)

                    wandb.log(
                        {
                            "valid_smiles": valid_smiles,
                            "UniqueAndValidMols": unique_mols,
                            "average_rings": ringave,
                        }
                    )

                    # set back to traiing.
                    vae.decoder.train()

        # This will create the file.
        # generated from a normal distribution. Note that vae.decoder must
        # be a Module
        z_1 = torch.normal(0, 1, size=(1, LATENT_DIM), device="cuda")
        with torch.no_grad():
            vae.decoder.eval()
            # torch.multinomial is not deterministic so we disable trace checking
            traced = torch.jit.trace(vae.decoder, z_1.to("cuda"), check_trace=False)

            torch.jit.save(traced, f"{args.out.rsplit('.', 1)[0]}_epoch_{epoch+1}.pth")

        # Print the time for epoch
        epoch_end = time.time()
        print(f"Epoch time {epoch_end - epoch_start: .4f}")

    # total time
    train_end = time.time()
    print(f"Total time {train_end - train_start: .4f}")

    # This will create the file.
    # generated from a normal distribution. Note that vae.decoder must
    # be a Module
    z_1 = torch.normal(0, 1, size=(1, LATENT_DIM), device="cuda")
    with torch.no_grad():
        vae.decoder.eval()
        # torch.multinomial is not deterministic so we disable trace checking
        traced = torch.jit.trace(vae.decoder, z_1.to("cuda"), check_trace=False)

        torch.jit.save(traced, args.out)
