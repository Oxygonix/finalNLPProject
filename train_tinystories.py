# train_char_lm.py (short + reasonably coherent)
#
# Char-level GPT training on:
#   - TinyStories        (default, easiest for small models)
#   - WikiText-2         (set CORPUS = "wikitext2")
#   - Shakespeare text   (set CORPUS = "shakespeare")
#
# This version is tuned for:
#   - fast-ish training
#   - more coherent-looking samples

import torch
from torch.utils.data import Dataset
from datasets import load_dataset

from mingpt.model import GPT
from mingpt.trainer import Trainer

# -------------------------------------------------
# 1) CHOOSE CORPUS
# -------------------------------------------------
CORPUS = "tinystories"   # "tinystories", "wikitext2", or "shakespeare"

# Limit how many characters we use from the corpus for speed
MAX_CHARS = 200_000   # set to None to use all


def load_text(corpus: str) -> str:
    """Load raw text for the selected corpus."""
    if corpus == "wikitext2":
        ds = load_dataset("wikitext", "wikitext-2-raw-v1")
        text = "\n\n".join(ds["train"]["text"])
    elif corpus == "tinystories":
        ds = load_dataset("roneneldan/TinyStories", split="train")
        text = "\n\n".join(ds["text"])
    elif corpus == "shakespeare":
        # Provide your own Shakespeare file at this path
        with open("shakespeare.txt", "r", encoding="utf-8") as f:
            text = f.read()
    else:
        raise ValueError(f"Unknown corpus: {corpus}")

    if MAX_CHARS is not None and len(text) > MAX_CHARS:
        text = text[:MAX_CHARS]
    return text


# -------------------------------------------------
# 2) CHAR-LEVEL ENCODER / DECODER
# -------------------------------------------------
def build_vocab(text: str):
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}

    def encode(s: str):
        return [stoi[c] for c in s]

    def decode(ids):
        return "".join(itos[i] for i in ids)

    return vocab_size, encode, decode


class CharDataset(Dataset):
    """
    Turns one long 1D tensor of token ids into (x, y) training examples:
        x = data[t : t+block_size]
        y = data[t+1 : t+1+block_size]
    """
    def __init__(self, data: torch.Tensor, block_size: int):
        self.data = data
        self.block_size = block_size

    def __len__(self):
        return len(self.data) - self.block_size - 1

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.block_size]
        y = self.data[idx + 1 : idx + 1 + self.block_size]
        return x, y


def get_device_str() -> str:
    if torch.cuda.is_available():
        return "cuda"
    # Apple Silicon / MPS
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    # ------------------------------
    # Load text and build vocab
    # ------------------------------
    raw_text = load_text(CORPUS)
    print(f"Loaded {len(raw_text):,} characters from {CORPUS}")

    vocab_size, encode, decode = build_vocab(raw_text)
    print("vocab_size:", vocab_size)

    data = torch.tensor(encode(raw_text), dtype=torch.long)

    # ------------------------------
    # Dataset
    # ------------------------------
    block_size = 128
    train_dataset = CharDataset(data, block_size)

    # ------------------------------
    # Model config (small but not tiny)
    # ------------------------------
    model_config = GPT.get_default_config()
    # We manually set size, so don't use a preset model_type
    model_config.model_type = None
    model_config.vocab_size = vocab_size
    model_config.block_size = block_size

    # Balanced small model: decent coherence, still fast
    model_config.n_layer = 4
    model_config.n_head = 4
    model_config.n_embd = 256

    model = GPT(model_config)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"number of parameters: {n_params / 1e6:.2f}M")

    device_str = get_device_str()
    device = torch.device(device_str)
    print(f"running on device {device_str}")

    model.to(device)

    # ------------------------------
    # Trainer config (short run)
    # ------------------------------
    train_config = Trainer.get_default_config()
    train_config.learning_rate = 3e-4

    # More steps than before, but on smaller data
    train_config.max_iters = 1000
    train_config.batch_size = 32

    # avoid multiprocessing DataLoader workers on macOS / Python 3.13
    train_config.num_workers = 0

    train_config.device = device_str

    # ------------------------------
    # Train
    # ------------------------------
    trainer = Trainer(train_config, model, train_dataset)
    trainer.run()

    # ------------------------------
    # Sample some text after training
    # ------------------------------
    seed = "The meaning of life is"
    context_ids = torch.tensor([encode(seed)], dtype=torch.long, device=device)

    sample_ids = model.generate(
        context_ids,
        max_new_tokens=300,
        temperature=0.8,
        top_k=50,
    )[0].tolist()

    print("\n=== SAMPLE ===\n")
    print(decode(sample_ids))


if __name__ == "__main__":
    main()