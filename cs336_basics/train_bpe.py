import os
from collections import Counter, defaultdict

import regex as re
import pickle
# GPT 2 Regex
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def save_helper(vocab, merges, filepath):
    with open(filepath, "wb") as f:
        pickle.dump((vocab, merges), f)
   
def per_word_logic(word, a, b):
    "Logic for the byte pair merges"
    new_word = []
    i = 0
    while i < len(word):
        if i < len(word) - 1 and word[i] == a and word[i + 1] == b:
            new_word.append(a + b)
            i += 2
        else:
            new_word.append(word[i])
            i += 1
    return tuple(new_word)


def train_bpe(
    input_path: str | os.PathLike, vocab_size: int, special_tokens: list[str]
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    This function trains a byte pair encoder using the input text

    Args:
        input_path: The input_path of text
        vocab_size: Size of vocab
        special_tokens: Special tokens added to the vocab and are protected

    Returns:
        A tuple of vocab and list of merges

    """
    # 1. Pre-tokenize the input text means split it into white space
    # 2. Initalize Vocab
    # 3. Run the merge loop

    with open(input_path) as file:
        content = file.read()
        if special_tokens:
            pattern = "|".join(re.escape(tok) for tok in special_tokens)
            chuncks = re.split(pattern, content)
        else:
            chuncks = [content]

        freq_table = Counter()
        for chunck in chuncks:
            freq_table.update(re.findall(PAT, chunck))

        word_freq = {}
        for key, value in freq_table.items():
            new_key = tuple(bytes([c]) for c in bytes(key, encoding="utf-8"))
            word_freq[new_key] = value
        vocab = {}
        for i in range(256):
            vocab[i] = bytes([i])
        n = len(vocab)
        for j, st in enumerate(special_tokens):
            vocab[n + j] = bytes(st, encoding="utf-8")
        merges = []
        pair_counter = Counter()
        address_book = defaultdict(set)
        for word, freq in word_freq.items():
            for a, b in zip(word, word[1:]):
                pair_counter[(a, b)] += freq
                address_book[(a,b)].add(word)
        while vocab_size > len(vocab):
            # step a: byte-pair counter
            
            if not pair_counter:
                break

            # step b: Pick the winner, pick lexographical best pair in case of tie
            winner, _ = max(pair_counter.items(), key=lambda x: (x[1], x[0]))
            # step c : add to vocab
            vocab[len(vocab)] = b"".join(winner)
            # step d : record mergers
            merges.append(winner)
            # step e: rewrite word_freq

            a, b = winner
            affected = list(address_book[winner])
            for old_word in affected:
                freq = word_freq[old_word]
                new_word = per_word_logic(old_word, a, b)
                if new_word != old_word:
                    for x, y in zip(old_word, old_word[1:]):
                        pair_counter[(x, y)] -= freq
                        address_book[(x,y)].discard(old_word)
                    for p1, p2 in zip(new_word, new_word[1:]):
                        pair_counter[(p1, p2)] += freq
                        address_book[(p1,p2)].add(new_word)
                del word_freq[old_word]
                word_freq[new_word] = word_freq.get(new_word, 0) + freq
            

    return vocab, merges