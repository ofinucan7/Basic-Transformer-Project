import os
import glob
import textwrap
from pathlib import Path
from typing import List, Tuple
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from load_model import ChatLM


DOC_DIRS = ["docs", "books"]         
CHUNK_WORDS = 220                      
TOP_K = 5                              
MAX_CONTEXT_CHARS = 2500              
TEMPERATURE = 0.7
TOP_K_SAMPLING = 50
MAX_TOKENS = 160
SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer the user's question using the CONTEXT. "
    'If the answer is not in the context, say "I don\'t know." '
    "Be concise and factual."
)

# ------------------------
def read_txt_files(dirs: List[str]) -> List[Tuple[str, str]]:
    items: List[Tuple[str, str]] = []
    for d in dirs:
        if not os.path.isdir(d):
            continue
        for path in glob.glob(str(Path(d) / "**" / "*.txt"), recursive=True):
            try:
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    txt = f.read()
                items.append((path, txt))
            except Exception:
                pass
    return items
# ------------------------

def split_into_chunks(text: str, words_per_chunk: int) -> List[str]:
    words = text.split()
    chunks = []
    for i in range(0, len(words), words_per_chunk):
        chunk = " ".join(words[i:i + words_per_chunk])
        if chunk.strip():
            chunks.append(chunk)
    return chunks
# ------------------------

def build_corpus():
    raw = read_txt_files(DOC_DIRS)
    chunk_texts, chunk_meta = [], []
    for path, txt in raw:
        chunks = split_into_chunks(txt, CHUNK_WORDS)
        for j, ch in enumerate(chunks):
            chunk_texts.append(ch)
            chunk_meta.append(f"{path}::chunk{j}")
    return chunk_texts, chunk_meta
# ------------------------

def build_index(chunks: List[str]) -> TfidfVectorizer:
    vec = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        max_df=0.9,
        min_df=1,
        max_features=50000,
    )
    vec.fit(chunks)
    return vec
# ------------------------

def retrieve(vec: TfidfVectorizer, chunks: List[str], query: str, k: int = TOP_K) -> List[Tuple[str, float]]:
    qv = vec.transform([query])
    cv = vec.transform(chunks)
    sims = cosine_similarity(qv, cv)[0]
    idx = sims.argsort()[::-1][:k]
    return [(chunks[i], float(sims[i])) for i in idx]
# ------------------------

def make_prompt(system, context_chunks, question):
    ctx = ""
    for ch in context_chunks:
        if len(ctx) + len(ch) + 2 > MAX_CONTEXT_CHARS:
            break
        ctx += ch.strip() + "\n\n"

    tpl = textwrap.dedent(f"""\
        SYSTEM:
        {system}

        CONTEXT:
        {ctx.strip() or "(no relevant passages found)"}

        USER:
        {question}

        ASSISTANT:
    """)
    return tpl
# ------------------------

def main():
    print("[index] Loading documents…")
    chunks, meta = build_corpus()
    if not chunks:
        print("No .txt files found in ./docs or ./books — add some first.")
        return
    print(f"[index] {len(chunks)} chunks")

    print("[index] Building TF-IDF index…")
    vec = build_index(chunks)
    print("[model] Loading model…")
    lm = ChatLM() 

    history: List[Tuple[str, str]] = []  
    print("\nInteractive RAG chat. Ask about your documents.\nCtrl+C to exit.\n")

    while True:
        try:
            q = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not q:
            continue

        hits = retrieve(vec, chunks, q, k=TOP_K)
        ctx_chunks = [t for (t, s) in hits]

        transcript = ""
        for u, a in history[-2:]:
            transcript += f"USER:\n{u}\n\nASSISTANT:\n{a}\n\n"

        prompt = make_prompt(SYSTEM_PROMPT, ctx_chunks, q)
        full_prompt = (transcript + prompt).strip()

        ans = lm.chat(
            full_prompt,
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
            top_k=TOP_K_SAMPLING,
            top_p=None,
        )

        print(f"Bot: {ans}\n")
        history.append((q, ans))

if __name__ == "__main__":
    main()