import torch
from collections import defaultdict
import math
# implemented in langugae model head

# ---------------------------------------------------------------
# BPE

def get_vocab(input_text):
    # Get all of the unique chars used in the input text
    # Input: 1.) input_text (str) - input
    # Return: 1.) individual chars w/ token '</w>' at end of every word (dictionary)
    vocab = {}

    # split the chars into their individual words and increment a frequency
    # append '</w>' to the end of every word
    for word in input_text.strip().split():
        chars = list(word) + ['</w>']
        token = tuple(chars)
        vocab[token] = vocab.get(token, 0) + 1

    return vocab

def get_pair_frequencies(vocab):
    # Get the frequency of all adjacent chars
    # ex. given "dog" --> split to {"d", "o", "g", "</w>"} --> get frequency of "do", "og", "g</w>" respectively
    # Input: 1.) vocab (dictonary) - tokenized vocab
    # Return: 1.) Return the frequency of the combination of tokens (dictionary)
    freq_pairs = defaultdict(int)
    
    # itterate over all words and frequencies of words in vocab
    for word, freq in vocab.items():
        for i in range(len(word)-1):
            pair = (word[i], word[i+1])
            freq_pairs[pair] += freq
    
    return freq_pairs

def merge_most_common_pair(vocab, pair):
    # Merge the pair that had the highest frequency together
    # Input: 1.) vocab (dictonary) - current vocab
    #        2.) pair (tuple) - most frequent pair to be merged
    # Return: 1.) Updated vocab with merged tokens (dictionary)

    updated_vocab = {}

    # merge the pair of tokens twice; once with a space, once without
    spaced_token = ' '.join(pair)
    together_token = ''.join(pair)

    # itterate over all of the previous vocab
    # if you find a word with the "pair" tokens together, merge them 
    for word, freq in vocab.items():
        cur_word_str = ' '.join(word)
        retokenized_word = cur_word_str.replace(spaced_token, together_token)
        updated_vocab[tuple(retokenized_word.split())] = freq
    
    return updated_vocab

def learn_bpe(text, num_merges):
    # Learn which tokens are going to be combined with BPE
    # Input: 1.) text (str) - input training text
    #        2.) num_merges (int) - number of merges to do
    # Return: 1.) List of which chars/tokens to merge (list of tuple)
    vocab = get_vocab(text)
    merges = []

    for _ in range(num_merges):
        pair_freqs = get_pair_frequencies(vocab)
        if not pair_freqs:
            break
        best_pair = max(pair_freqs, key=pair_freqs.get)
        merges.append(best_pair)
        vocab = merge_most_common_pair(vocab, best_pair)

    return merges

def do_bpe(word, merges):
    # Apply BPE with merges
    # Input: 1.) word (str) - What is BPE is being applied to
    #        2.) merges (list) - What chars/tokens are being merged together
    # Return: 1.) BPE tokens after applying merges to the input word (list)
    word = list(word) + ['</w>']
    i = 0
    while i < len(word) - 1:
        pair = (word[i], word[i+1])
        if pair in merges:
            word[i:i+2] = [''.join(pair)]
            i = max(i - 1, 0)
        else:
            i += 1
    return word

# ---------------------------------------------------------------
# Positional Encoding
def get_positional_encoding(seq_len, embedding_dims, device, N=10000.0):
    # Do positional encoding with sin/cos
    # Input: 1.) seq_len (int) - length of sequence
    #        2.) embedding_dims (int) - embedding dimensions
    #        3.) device (str) - 'cpu' or 'cuda' depending on device
    #        4.) N (float) - scale of the frequency
    # # Return: 1.) Positional encoding tensor of shape (seq_len, embedding_dims)
    pe = torch.zeros(seq_len, embedding_dims, device=device)
    for pos in range(seq_len):
        for i in range(0, embedding_dims, 2):
            k = i // 2
            denom = N ** (2 * k / embedding_dims)

            # even index --> use sin
            pe[pos, i] = math.sin(pos / denom)

            # if odd index that is in bounds --> use cos
            if i + 1 < embedding_dims:
                pe[pos, i + 1] = math.cos(pos / denom)
                
    return pe  # shape: (seq_len, embedding_dims)

# ---------------------------------------------------------------
# Driver
def run_bpe(text, num_merges):
    # Driver function to be called in main file
    # Input: 1.) text (str) - What is BPE is being applied to
    #        2.) num_merges (int) - number of BPE merges to be done
    # Return: 1.) dict w/ various info/data (see return statement)
    

    # learn the BPE merges
    merges = learn_bpe(text, num_merges)

    # do BPE to every word
    bpe_encoded = []
    for word in text.strip().split():
        bpe_encoded.extend(do_bpe(word, merges))

    # create the vocab
    vocab = sorted(set(bpe_encoded))
    s_to_i = {tok: i for i, tok in enumerate(vocab)}
    i_to_s = {i: tok for i, tok in enumerate(vocab)}

    # convert into token ids/indices
    token_ids = torch.tensor([s_to_i[token] for token in bpe_encoded], dtype=torch.long) # (seq_len)

    return {
        "tokens": bpe_encoded,
        "token_ids": token_ids,
        "s_to_i": s_to_i,
        "i_to_s": i_to_s,
        "merges": merges
    }