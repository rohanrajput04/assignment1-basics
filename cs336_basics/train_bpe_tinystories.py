import os
from collections import Counter, defaultdict
from .pretokenization_example import find_chunk_boundaries
import multiprocessing
import heapq

import regex as re

# GPT 2 Regex
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

class InvPair:
    def __init__(self, p): self.p = p
    def __lt__(self, other): return self.p > other.p

def process(input_path, start, end, special_tokens):
    with open(input_path, "rb") as file:
        file.seek(start)
        content = file.read(end - start).decode("utf-8", errors="ignore")
        # Run pre-tokenization on your chunk and store the counts for each pre-token
        if special_tokens:
            pattern = "|".join(re.escape(tok) for tok in special_tokens)
            chuncks = re.split(pattern, content)
        else:
            chuncks = [content]
        local_counter = Counter()
        for chunck in chuncks:
            local_counter.update(re.findall(PAT, chunck))

        

    return local_counter


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

    with open(input_path, "rb") as file:
        num_processes = os.cpu_count()
        boundaries = find_chunk_boundaries(file, num_processes, b"<|endoftext|>")
        
    arg_list = [(input_path, start, end, special_tokens) for start, end in zip(boundaries[:-1], boundaries[1:])]

    with multiprocessing.Pool(num_processes) as pool:
        results = pool.starmap(process, arg_list)
    freq_table = Counter()
    for c in results:
        freq_table += c

            

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
    heap = [(-count, InvPair(pair)) for pair, count in pair_counter.items()]
    heapq.heapify(heap)

    while vocab_size > len(vocab):
        # step a: byte-pair counter
        
        if not pair_counter:
            break

        # step b: Pick the winner, pick lexographical best pair in case of tie
        #winner, _ = max(pair_counter.items(), key=lambda x: (x[1], x[0]))
        while heap:
            neg_count, inv_pair = heapq.heappop(heap)
            pair = inv_pair.p
            if pair_counter[pair] == -neg_count:
                winner = pair
                break

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
                    heapq.heappush(heap, (-pair_counter[(x,y)], InvPair((x,y))))
                for p1, p2 in zip(new_word, new_word[1:]):
                    pair_counter[(p1, p2)] += freq
                    address_book[(p1,p2)].add(new_word)
                    heapq.heappush(heap, (-pair_counter[(p1,p2)], InvPair((p1,p2))))
            del word_freq[old_word]
            word_freq[new_word] = word_freq.get(new_word, 0) + freq
        

    return vocab, merges