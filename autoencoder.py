"""This script trains an autoencoder used as a base for a VAE."""

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

    def __init__(self, data_path, max_length=10):
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
        self.to_latent = nn.Linear(num_layers * hidden_size, self.latent_dim)

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
        z = self.to_latent(hidden)

        return z


class Decoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers):
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

    def forward(self, z, actual_input):
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

        # actual_input: (B, L) — batch of token indices
        # embedded: (B, L, embedding_dim)
        embedded = self.embedding(actual_input)

        # embedded: (B, L, embedding_dim)
        # output:   (B, L, hidden_size): hidden states for every time step
        # hidden:   (num_layers, B, hidden_size): hidden state at final time step for each layer
        output, hidden = self.gru(embedded)

        # Training teacher forcing

        # Map GRU output to vocab space at each time step
        # output:   (B, L, hidden_size): hidden states for every time step
        # logits: (B, L, vocab_size)
        logits = self.output_layer(output)

        return logits


class VAE(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers):
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
        )

    def forward(self, input_seq):
        # input_seq: (B, L)
        # z:        (B, num_layers * hidden_size)
        z = self.encoder(input_seq)

        # input_seq: (B, L)
        # z:        (B, num_layers * hidden_size)
        # logits:   (B, L, vocab_size)
        logits = self.decoder(z, input_seq)

        return logits


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
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--embedding_dim", type=int, default=20)
    parser.add_argument("--hidden_size", type=int, default=1024)
    parser.add_argument("--num_layers", type=int, default=1)

    args = parser.parse_args()

    dataset = SmilesDataset(args.train_data)

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
    ).to("cuda")

    # Weight
    optimizer = torch.optim.Adam(vae.parameters(), lr=args.lr)

    # TRAIN THE MODEL
    train_start = time.time()
    log_interval = 1100
    vae.train()

    for epoch in range(args.epochs):
        # Track the time
        epoch_start = time.time()

        for batch_idx, batch in enumerate(dataloader):
            batch = batch.to("cuda")

            # Call the model
            # batch:    (B,  L)
            # logits:   (B, L, vocab_size)
            logits = vae(batch)

            # batch:      (B,  L)
            # logits:     (B, L, V)
            # First view: (B * L, V)
            # Last view:  (B * L)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), batch.view(-1))

            # clauclating the loss and optimizing
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Printing the loss for batch
            # Print every `log_interval` batches
            if batch_idx % log_interval == 0:
                print(f"[Epoch {epoch+1} | Batch {batch_idx}] Loss: {loss.item():.4f}")

        # Print the time for epoch
        epoch_end = time.time()
        print(f"Epoch time {epoch_end - epoch_start: .4f}")

    # total time
    train_end = time.time()
    print(f"Total time {train_end - train_start: .4f}")

    # Testing reconstruction
    count = 0
    max_to_check = 20
    vae.eval()
    lang = Lang()

    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to("cuda")
            logits = vae(batch)
            preds = torch.argmax(logits, dim=-1)

            for i in range(batch.size(0)):
                true_str = lang.indexToSmiles(batch[i].tolist())
                pred_str = lang.indexToSmiles(preds[i].tolist())

                # print them
                print(f"Input   {count+1}: {true_str}")
                print(f"Decoded {count+1}: {pred_str}\n")
                count += 1

                if count >= max_to_check:
                    break
