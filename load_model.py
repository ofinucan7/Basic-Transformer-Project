import torch
import sys
import torch.nn.functional as F
import training  
import collections
from transformer_implementation import TransformerImplementation
from bpe_position_encoding import do_bpe
from pathlib import Path
from typing import List, Tuple, Optional

class ChatLM:
    def __init__(self, device, weights_path: str="model_weights.pth"):
        tok = training.load_tokenizer("tokenizer.pkl")
        self.merges = tok["merges"]
        self.s_to_i = tok["s_to_i"]
        self.i_to_s = tok["i_to_s"]
        self.pad_id = tok["pad_id"]
        self.unk_id = tok["unk_id"]
        self.bos_id = tok["bos_id"]
        self.eos_id = tok["eos_id"]

        self.vocab_size = len(self.s_to_i)
        hp = training.hyperparams
        self.context_length = hp["context_length"]

        self.model = TransformerImplementation(
            vocab_size=self.vocab_size,
            embedding_dims=hp["embedding_dims"],
            num_heads=hp["num_heads"],
            ffn_hidden_dims=hp["ffn_hidden_dims"],
            dropout_rate=hp["dropout_rate"],
            num_layers=hp["num_layers"],
            max_seq_len=self.context_length,
        )
        try:
            self.model.lm_head.weight = self.model.embedding.weight
        except Exception:
            pass
        sd = torch.load(weights_path, map_location="cpu")
        self.model.load_state_dict(sd, strict=True)

        if device is None:
            device = "cuda" if torch.cuda.is_available() else sys.exit()
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()

    # ----------------------------------------
    # tokenizer helpers    
    def encode(self, text, add_bos=True, add_eos=False):
        ids = [self.bos_id] if add_bos else []
        for w in text.strip().split():
            toks = do_bpe(w, self.merges)
            for t in toks:
                ids.append(self.s_to_i.get(t, self.unk_id))
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def decode(self, ids):
        toks = [self.i_to_s.get(i, "<unk>") for i in ids]
        toks = [t for t in toks if t not in ("<BOS>", "<EOS>", "</w>")]
        out = " ".join(toks).replace(" </w>", "").replace("</w>", " ")
        return out.strip()

    # ----------------------------------------
    # embedding for retrieval
    @torch.no_grad()
    def embed_text_mean(self, text):
        ids = self.encode(text, add_bos=False, add_eos=False)
        if not ids:
            return torch.zeros((self.model.embedding.embedding_dim,), dtype=torch.float32)
        ids_t = torch.tensor(ids, dtype=torch.long, device=self.device).unsqueeze(0)
        emb = self.model.embedding(ids_t)  # (1, T, d)
        vec = emb.mean(dim=1).squeeze(0).detach().cpu()
        return vec  # (d,)

    # ----------------------------------------
    # generation
    @torch.no_grad()
    def chat(self,
             prompt,
             max_tokens: int = 160,
             temperature: float = 0.7,
             top_k: Optional[int] = 50,
             top_p: Optional[float] = None,
             freq_penalty: float = 0.0):

        ids = self.encode(prompt, add_bos=True, add_eos=False)
        if len(ids) > self.context_length:
            ids = ids[-self.context_length:]
        x = torch.tensor(ids, dtype=torch.long, device=self.device).unsqueeze(0)  # (1, T)

        generated: List[int] = []
        for _ in range(max_tokens):
            logits = self.model(x)[:, -1, :]  # (1, V)
            logits = logits.clone()

            if freq_penalty > 0 and generated:
                counts = collections.Counter(generated)
                for tid, cnt in counts.items():
                    logits[0, tid] -= freq_penalty * float(cnt)

            if temperature <= 0:
                next_id = int(torch.argmax(logits, dim=-1).item())
            else:
                if top_k is not None and top_k > 0:
                    topk_vals, topk_idx = torch.topk(logits, k=min(top_k, logits.size(-1)))
                    logits_filtered = torch.full_like(logits, float("-inf"))
                    logits_filtered.scatter_(1, topk_idx, topk_vals)
                    logits = logits_filtered
                if top_p is not None and 0.0 < top_p < 1.0:
                    sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                    probs = torch.softmax(sorted_logits / max(1e-6, temperature), dim=-1)
                    cumprobs = torch.cumsum(probs, dim=-1)
                    mask = cumprobs > top_p
                    mask[..., 1:] = mask[..., :-1].clone()
                    mask[..., 0] = 0
                    sorted_logits[mask] = float("-inf")
                    logits = torch.full_like(logits, float("-inf"))
                    logits.scatter_(1, sorted_idx, sorted_logits)

                probs = torch.softmax(logits / max(1e-6, temperature), dim=-1)
                next_id = int(torch.multinomial(probs, num_samples=1).item())

            generated.append(next_id)
            ids.append(next_id)
            x = torch.tensor(ids[-self.context_length:], dtype=torch.long, device=self.device).unsqueeze(0)
            if next_id == self.eos_id:
                break

        text = self.decode(generated)
        return text