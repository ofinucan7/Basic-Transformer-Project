import os
import sys
import math
import pickle
import hashlib
import torch
import torch.nn as nn
from contextlib import nullcontext
from typing import List, Tuple
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.amp import GradScaler, autocast
from bpe_position_encoding import run_bpe, do_bpe
from transformer_implementation import TransformerImplementation

# ----------------------------------------
# get device and set seed
device = torch.device("cuda" if torch.cuda.is_available() else sys.exit())
torch.backends.cudnn.benchmark = True
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass
g = torch.Generator().manual_seed(42)

# ----------------------------------------
# Inputs (Training) and outputs
LEARNING_MATS = ["corpus.txt", "tiny_shakespeare.txt"]
TEXT_PATH = next((p for p in LEARNING_MATS if os.path.exists(p)), LEARNING_MATS[-1])

TOKENIZER_OUT = "tokenizer.pkl"          # merges + vocab + specials
TOKEN_IDS_OUT = "corpus_token_ids.pt"    # cached encoded ids 
WEIGHTS_OUT = "model_weights.pth"        # where to save the weights of the model

# ----------------------------------------
# Hyperparameters 
hyperparams = {
    "embedding_dims": 256,
    "num_heads": 8,
    "ffn_hidden_dims": 1024,
    "dropout_rate": 0.1,
    "num_layers": 6,
    "context_length": 256,
    "batch_size": 64,
    "num_epochs": 10,         
    "learning_rate": 3e-4,
    "weight_decay": 0.1,
    "warmup_steps": 500,
    "num_merges": 8000,      
    "stride": 16,             
    "grad_accum_steps": 4,    
    "label_smoothing": 0.1,   
    "log_every": 100
}

# ----------------------------------------
# Helpers (tokenizer caching)
def file_sha256(path, chunk_mb):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(1024 * 1024 * chunk_mb)
            if not b:
                break
            h.update(b)
    return h.hexdigest()

def merges_sha(merges):
    s = "|".join(f"{a}⟂{b}" for (a, b) in merges)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def add_special_tokens(s_to_i, i_to_s):
    specials = ["<pad>", "<unk>", "<bos>", "<eos>"]
    for tok in specials:
        if tok not in s_to_i:
            idx = len(s_to_i)
            s_to_i[tok] = idx
            i_to_s[idx] = tok
    return (
        s_to_i,
        i_to_s,
        s_to_i["<pad>"],
        s_to_i["<unk>"],
        s_to_i["<bos>"],
        s_to_i["<eos>"],
    )

def save_tokenizer(path, merges, s_to_i, i_to_s, pad_id, unk_id, bos_id, eos_id):
    obj = {
        "merges": merges,
        "s_to_i": s_to_i,
        "i_to_s": i_to_s,
        "pad_id": pad_id,
        "unk_id": unk_id,
        "bos_id": bos_id,
        "eos_id": eos_id,
    }
    with open(path, "wb") as f:
        pickle.dump(obj, f)

def load_tokenizer(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)

def encode_text_words(text: str, merges, s_to_i: dict, bos_id: int, eos_id: int, unk_id: int) -> List[int]:
    ids: List[int] = [bos_id]
    for w in text.strip().split():
        toks = do_bpe(w, merges)
        for t in toks:
            ids.append(s_to_i.get(t, unk_id))
    ids.append(eos_id)
    return ids

# ----------------------------------------
# dataset sliding windows
class LMSequenceDataset(Dataset):
    def __init__(self, ids: List[int], context_len: int, stride: int):
        self.ids = ids
        self.context_len = context_len
        self.stride = max(1, stride)
        max_start = len(ids) - (context_len + 1)
        self.starts = list(range(0, max_start + 1, self.stride)) if max_start > 0 else []

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, idx):
        s = self.starts[idx]
        x = torch.tensor(self.ids[s : s + self.context_len], dtype=torch.long)
        y = torch.tensor(self.ids[s + 1 : s + 1 + self.context_len], dtype=torch.long)
        return x, y

# ----------------------------------------
# compile guard (skips if Triton isn't available)
def maybe_compile(model):
    if device.type != "cuda":
        print("torch.compile: skipped (CPU).")
        return model
    try:
        import triton  # noqa: F401
    except Exception:
        print("torch.compile: skipped (no Triton on this system).")
        return model
    try:
        model = torch.compile(model)  # inductor+triton
        print("torch.compile: enabled.")
    except Exception as e:
        print(f"torch.compile failed ({e}). Using eager mode.")
    return model

# ----------------------------------------
# training loops
def main():
    if not os.path.exists(TEXT_PATH):
        raise FileNotFoundError("no learning materials were found, try again")

    print(f"Using training material: {TEXT_PATH}")
    print("Loading text …")
    with open(TEXT_PATH, "r", encoding="utf-8") as f:
        text = f.read()

    # ----------------------------------------
    # tokenizer
    need_tokenizer = True
    if os.path.exists(TOKENIZER_OUT):
        try:
            tok = load_tokenizer(TOKENIZER_OUT)
            merges = tok["merges"]; s_to_i = tok["s_to_i"]; i_to_s = tok["i_to_s"]
            pad_id = tok["pad_id"]; unk_id = tok["unk_id"]; bos_id = tok["bos_id"]; eos_id = tok["eos_id"]
            need_tokenizer = False
            print(f"got the tokenizer of vocab_size={len(s_to_i)} and merges={len(merges)}")
        except Exception as e:
            print("rebuilding the tokenizer")

    if need_tokenizer:
        print("starting to learn BPE")
        bpe_info = run_bpe(text, hyperparams["num_merges"])
        merges = bpe_info["merges"]
        s_to_i = bpe_info["s_to_i"].copy()
        i_to_s = bpe_info["i_to_s"].copy()
        s_to_i, i_to_s, pad_id, unk_id, bos_id, eos_id = add_special_tokens(s_to_i, i_to_s)
        save_tokenizer(TOKENIZER_OUT, merges, s_to_i, i_to_s, pad_id, unk_id, bos_id, eos_id)
        print(f"tokenizer saved at {TOKENIZER_OUT} of vocab_size={len(s_to_i)}, merges={len(merges)}")

    vocab_size = len(s_to_i)

    # ----------------------------------------
    # encoder
    this_corpus_sha = file_sha256(TEXT_PATH)
    this_merges_sha = merges_sha(merges)

    need_ids = True
    if os.path.exists(TOKEN_IDS_OUT):
        try:
            cache = torch.load(TOKEN_IDS_OUT, map_location="cpu")
            meta = cache.get("meta", {})
            if meta.get("corpus_sha") == this_corpus_sha and meta.get("merges_sha") == this_merges_sha:
                token_ids = cache["ids"].tolist()
                need_ids = False
                print(f"Loaded cached token ids from {TOKEN_IDS_OUT} (N={len(token_ids)})")
            else:
                print("Cache miss for token ids: corpus or merges changed. Re-encoding …")
        except Exception as e:
            print(f"Warning: failed to load {TOKEN_IDS_OUT}: {e}. Re-encoding …")

    if need_ids:
        print("Encoding full corpus …")
        token_ids = encode_text_words(text, merges, s_to_i, bos_id, eos_id, unk_id)
        torch.save(
            {"ids": torch.tensor(token_ids, dtype=torch.long),
             "meta": {"corpus_sha": this_corpus_sha, "merges_sha": this_merges_sha}},
            TOKEN_IDS_OUT
        )
        print(f"Saved encoded ids cache to {TOKEN_IDS_OUT} (N={len(token_ids)})")

    # ----------------------------------------
    # dataset and loader
    ds = LMSequenceDataset(token_ids, hyperparams["context_length"], hyperparams["stride"])
    if len(ds) == 0:
        raise RuntimeError("Dataset produced zero training windows. Reduce context_length or stride.")
    dl = DataLoader(
        ds,
        batch_size=hyperparams["batch_size"],
        shuffle=True,
        drop_last=True,
        generator=g,
        pin_memory=(device.type == "cuda"),
        num_workers=0, 
    )
    print(f"Training samples: {len(ds)} (batch_size={hyperparams['batch_size']})")

    # ----------------------------------------
    # setting up the model
    print("building model")
    model = TransformerImplementation(
        vocab_size=vocab_size,
        embedding_dims=hyperparams["embedding_dims"],
        num_heads=hyperparams["num_heads"],
        ffn_hidden_dims=hyperparams["ffn_hidden_dims"],
        dropout_rate=hyperparams["dropout_rate"],
        num_layers=hyperparams["num_layers"],
        max_seq_len=hyperparams["context_length"],
    ).to(device)

    if hasattr(model, "embedding") and hasattr(model, "lm_head"):
        try:
            model.lm_head.weight = model.embedding.weight
        except Exception:
            pass

    # ----------------------------------------
    # triton fix
    model = maybe_compile(model)

    # # ----------------------------------------
    # optimizer w/ label smoothing
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim >= 2:
            decay.append(p)
        else:
            no_decay.append(p)
    optimizer = AdamW(
        [{"params": decay, "weight_decay": hyperparams["weight_decay"]},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=hyperparams["learning_rate"],
        betas=(0.9, 0.95),
        eps=1e-8
    )

    criterion = nn.CrossEntropyLoss(label_smoothing=hyperparams["label_smoothing"])

    # ----------------------------------------
    # scheduler (cosine w/ warmup)
    grad_accum_steps = max(1, hyperparams["grad_accum_steps"])
    batches_per_epoch = max(1, len(dl))
    updates_per_epoch = math.ceil(batches_per_epoch / grad_accum_steps)
    total_updates = hyperparams["num_epochs"] * updates_per_epoch
    warmup = min(hyperparams["warmup_steps"], total_updates - 1) if total_updates > 1 else 0

    def lr_lambda(step):
        if warmup > 0 and step < warmup:
            return step / max(1, warmup)
        if total_updates <= warmup:
            return 1.0
        progress = (step - warmup) / max(1, total_updates - warmup)
        cosine = 0.5 * (1 + math.cos(math.pi * progress))
        min_ratio = 0.1
        return min_ratio + (1 - min_ratio) * cosine

    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

    scaler = GradScaler(enabled=(device.type == "cuda"))
    amp_ctx = autocast(device_type="cuda") if device.type == "cuda" else nullcontext()

    # ---------- Training loop ----------
    print("starting to train")
    global_step = 0      # batches seen
    update_step = 0      # optimizer steps taken
    log_every = max(1, hyperparams["log_every"])

    for epoch in range(1, hyperparams["num_epochs"] + 1):
        model.train()
        running = 0.0
        optimizer.zero_grad(set_to_none=True)
        for batch_idx, (x, y) in enumerate(dl, start=1):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            with amp_ctx:
                logits = model(x)  # (B, T, V)
                loss = criterion(logits.view(-1, vocab_size), y.view(-1)) / grad_accum_steps

            scaler.scale(loss).backward()

            if (batch_idx % grad_accum_steps) == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                update_step += 1

            global_step += 1
            running += loss.item() * grad_accum_steps 

            if global_step % log_every == 0:
                print(f"Step {global_step:6d} | Update {update_step:6d} | Loss {running / log_every:.4f}")
                running = 0.0

        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch:2d} complete | LR {current_lr:.6g} | Updates so far {update_step}")

    torch.save(model.state_dict(), WEIGHTS_OUT)
    print(f"Saved weights to {WEIGHTS_OUT}")

# ----------------------------------------

if __name__ == "__main__":
    main()