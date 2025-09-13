import os
import math
import json
import argparse
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import training  
from transformer_implementation import TransformerImplementation
from bpe_position_encoding import do_bpe
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, List

# ---------- Paths ----------
TOKENIZER = Path("tokenizer.pkl")
TOKEN_IDS = Path("corpus_token_ids.pt")
WEIGHTS = Path("model_weights.pth")
OUTDIR = Path("eval_out")
OUTDIR.mkdir(exist_ok=True)

# ---------- Args ----------
parser = argparse.ArgumentParser()
parser.add_argument("--weights", type=str, default=str(WEIGHTS), help="Path to .pth weights")
parser.add_argument("--sanity", type=str, default=None, help="Optional path to a tiny sanity .txt set to report PPL on")
parser.add_argument("--note", type=str, default="", help="Optional note to include in eval_log.json (e.g., 'label_smoothing=0.1, accum=4')")
parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
args = parser.parse_args()

# ---------- Load tokenizer ----------
tok = training.load_tokenizer(str(TOKENIZER))
merges = tok["merges"]
s_to_i = tok["s_to_i"]
i_to_s = tok["i_to_s"]
pad_id = tok["pad_id"]
unk_id = tok["unk_id"]
bos_id = tok["bos_id"]
eos_id = tok["eos_id"]
vocab_size = len(s_to_i)

def encode_text_words(text):
    # replicate training.encode_text_words but local to avoid import surprises
    ids = [bos_id]
    for w in text.strip().split():
        toks = do_bpe(w, merges)
        for t in toks:
            ids.append(s_to_i.get(t, unk_id))
    ids.append(eos_id)
    return ids

# ---------- Load corpus token ids ----------
cache = torch.load(str(TOKEN_IDS), map_location="cpu")
token_ids_full = cache["ids"].tolist()
meta = cache.get("meta", {})
corpus_sha = meta.get("corpus_sha", "unknown")
merges_sha = meta.get("merges_sha", "unknown")

# ---------- Build model ----------
hp = training.hyperparams
model = TransformerImplementation(
    vocab_size=vocab_size,
    embedding_dims=hp["embedding_dims"],
    num_heads=hp["num_heads"],
    ffn_hidden_dims=hp["ffn_hidden_dims"],
    dropout_rate=hp["dropout_rate"],
    num_layers=hp["num_layers"],
    max_seq_len=hp["context_length"],
)
# weight tying (if not already inside)
try:
    model.lm_head.weight = model.embedding.weight
except Exception:
    pass
model.load_state_dict(torch.load(args.weights, map_location="cpu"), strict=True)
device = torch.device(args.device)
model.to(device)
model.eval()

# ---------- Dataset windows ----------
Context = hp["context_length"]
Stride = max(1, hp["stride"])

def make_starts(n_tokens, context, stride):
    max_start = n_tokens - (context + 1)
    return list(range(0, max_start + 1, stride)) if max_start > 0 else []

# Split 95/5 by token position, then derive windows within each region
split = int(0.95 * len(token_ids_full))
train_ids = token_ids_full[:split]
val_ids = token_ids_full[split:]

train_starts = make_starts(len(train_ids), Context, Stride)
val_starts   = make_starts(len(val_ids),   Context, Stride)

def batch_iter(ids, starts, batch_size):
    for i in range(0, len(starts), batch_size):
        S = starts[i:i+batch_size]
        if not S:
            continue
        X = torch.tensor([[ids[s + j] for j in range(Context)] for s in S], dtype=torch.long, device=device)
        Y = torch.tensor([[ids[s + 1 + j] for j in range(Context)] for s in S], dtype=torch.long, device=device)
        yield X, Y

@torch.no_grad()
def compute_loss_and_ppl(ids, starts, cap_batches):
    if not starts:
        return float('nan'), float('nan'), 0
    criterion = nn.CrossEntropyLoss(reduction="sum")  # sum over tokens, divide later
    total_loss = 0.0
    total_toks = 0
    batches_seen = 0
    for X, Y in batch_iter(ids, starts, training.hyperparams["batch_size"]):
        logits = model(X)  # (B, T, V)
        loss = criterion(logits.reshape(-1, vocab_size), Y.reshape(-1))
        total_loss += loss.item()
        total_toks += Y.numel()
        batches_seen += 1
        if cap_batches is not None and batches_seen >= cap_batches:
            break
    avg_loss = total_loss / max(1, total_toks)
    ppl = math.exp(avg_loss) if math.isfinite(avg_loss) else float('inf')
    return avg_loss, ppl, total_toks

# Estimate train loss on a subset of windows (fast), and full val
train_loss, train_ppl, train_tokens = compute_loss_and_ppl(train_ids, train_starts, cap_batches=200)
val_loss,   val_ppl,   val_tokens   = compute_loss_and_ppl(val_ids,   val_starts,   cap_batches=None)

# ----------------------------------------
# sanity checks
sanity_ppl = None
sanity_path = args.sanity
if sanity_path and os.path.exists(sanity_path):
    with open(sanity_path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    sanity_ids = encode_text_words(text)
    sanity_starts = make_starts(len(sanity_ids), Context, Stride)
    _, sanity_ppl, _ = compute_loss_and_ppl(sanity_ids, sanity_starts, cap_batches=None)

# ---------- Plot: Train vs Val (loss + ppl as labels) ----------
fig1 = plt.figure(figsize=(5,4))
xs = ["Train", "Val"]
ys = [train_loss, val_loss]
plt.bar(xs, ys)
plt.ylabel("Average Cross-Entropy Loss (nats)")
plt.title("Train vs Val Loss")
for i, v in enumerate(ys):
    label = f"PPL≈{math.exp(v):.1f}" if math.isfinite(v) else "PPL=∞"
    plt.text(i, v, label, ha="center", va="bottom", fontsize=9)
fn_loss_plot = OUTDIR / "train_vs_val_loss.png"
plt.tight_layout()
plt.savefig(fn_loss_plot)
plt.close(fig1)

# ---------- Plot: LR schedule over optimizer steps ----------
# Derive updates/epoch based on dataset + batch + gradient accumulation
num_train_batches = math.ceil(len(train_starts) / max(1, training.hyperparams["batch_size"]))
updates_per_epoch = num_train_batches // max(1, training.hyperparams["grad_accum_steps"])
total_updates = updates_per_epoch * training.hyperparams["num_epochs"]
warmup = min(training.hyperparams["warmup_steps"], total_updates)

def lr_at(step:int):
    if step < warmup:
        return training.hyperparams["learning_rate"] * (step + 1) / max(1, warmup)
    if total_updates <= warmup:
        return training.hyperparams["learning_rate"]
    progress = (step - warmup) / max(1, total_updates - warmup)
    cosine = 0.5 * (1 + math.cos(math.pi * progress))
    min_ratio = 0.1
    return training.hyperparams["learning_rate"] * (min_ratio + (1 - min_ratio) * cosine)

steps = list(range(total_updates))
lrs = [lr_at(s) for s in steps]

fig2 = plt.figure(figsize=(6,4))
plt.plot(steps, lrs)
plt.xlabel("Optimizer Step")
plt.ylabel("Learning Rate")
plt.title("Cosine LR Schedule with Warmup")
fn_lr_plot = OUTDIR / "lr_schedule.png"
plt.tight_layout()
plt.savefig(fn_lr_plot)
plt.close(fig2)

# ---------- Save report ----------
report = {
    "weights": str(args.weights),
    "device": args.device,
    "vocab_size": vocab_size,
    "context_length": Context,
    "stride": Stride,
    "batch_size": training.hyperparams["batch_size"],
    "grad_accum_steps": training.hyperparams["grad_accum_steps"],
    "label_smoothing_train": training.hyperparams["label_smoothing"],
    "num_epochs": training.hyperparams["num_epochs"],
    "corpus_sha": corpus_sha,
    "merges_sha": merges_sha,
    "train_tokens_evald": train_tokens,
    "val_tokens_evald": val_tokens,
    "train_loss": train_loss,
    "train_ppl": train_ppl,
    "val_loss": val_loss,
    "val_ppl": val_ppl,
    "sanity_path": sanity_path,
    "sanity_ppl": sanity_ppl,
    "notes": args.note,
    "plots": {
        "train_vs_val_loss": str(fn_loss_plot),
        "lr_schedule": str(fn_lr_plot),
    }
}
with open(OUTDIR / "eval_report.json", "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2)

with open(OUTDIR / "eval_report.txt", "w", encoding="utf-8") as f:
    f.write(
        f"Evaluation Report\n"
        f"=================\n"
        f"Weights: {args.weights}\n"
        f"Device: {args.device}\n"
        f"Vocab size: {vocab_size}\n"
        f"Context/Stride: {Context}/{Stride}\n"
        f"Train tokens evald: {train_tokens}\n"
        f"Val tokens evald  : {val_tokens}\n"
        f"Train Loss: {train_loss:.4f}  | PPL≈{(math.exp(train_loss) if math.isfinite(train_loss) else float('inf')):.2f}\n"
        f"Val   Loss: {val_loss:.4f}    | PPL≈{(math.exp(val_loss) if math.isfinite(val_loss) else float('inf')):.2f}\n"
        f"Sanity PPL ({sanity_path}): {sanity_ppl}\n"
        f"LR Plot: {fn_lr_plot}\n"
        f"Notes: {args.note}\n"
    )


with open(OUTDIR / "eval_runs.jsonl", "a", encoding="utf-8") as f:
    import json as _json
    f.write(_json.dumps(report) + "\n")
print(f"[OK] Saved report to {OUTDIR}/eval_report.json and plots to {OUTDIR}/")