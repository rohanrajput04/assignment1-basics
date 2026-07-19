import os
import numpy as np
import multiprocessing as mp
from .tokenizer import Tokenizer
import sys

SPECIAL_TOKEN = "<|endoftext|>"

# One tokenizer per worker process, loaded once.
_tokenizer = None

def _init_worker(pkl_path):
    global _tokenizer
    _tokenizer = Tokenizer.from_files(pkl_path, special_tokens=[SPECIAL_TOKEN])

def _encode_chunk(args):
    txt_path, start, end = args
    with open(txt_path, "rb") as f:
        f.seek(start)
        text = f.read(end - start).decode("utf-8", errors="ignore")
    return np.asarray(_tokenizer.encode(text), dtype=np.uint16)

def find_chunk_boundaries(file_path, desired_num_chunks, split_token=b"<|endoftext|>"):
    """Pick byte offsets that split the file into ~equal chunks aligned to <|endoftext|>."""
    file_size = os.path.getsize(file_path)
    chunk_size = file_size // desired_num_chunks
    boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    boundaries[-1] = file_size

    mini = 4096
    with open(file_path, "rb") as f:
        for bi in range(1, len(boundaries) - 1):
            pos = boundaries[bi]
            f.seek(pos)
            while True:
                buf = f.read(mini)
                if not buf:
                    boundaries[bi] = file_size
                    break
                found = buf.find(split_token)
                if found != -1:
                    boundaries[bi] = pos + found
                    break
                pos += mini
    return sorted(set(boundaries))

def encode_file(pkl_path, txt_path, out_npy_path,
                num_workers=None, chunks_per_worker=4):
    num_workers = num_workers or os.cpu_count() or 1

    # More chunks than workers => better load balancing on uneven documents.
    boundaries = find_chunk_boundaries(txt_path, num_workers * chunks_per_worker)
    tasks = [(txt_path, s, e) for s, e in zip(boundaries[:-1], boundaries[1:])]

    with mp.Pool(num_workers, initializer=_init_worker, initargs=(pkl_path,)) as pool:
        parts = pool.map(_encode_chunk, tasks)

    arr = np.concatenate(parts)
    np.save(out_npy_path, arr)

    print(arr.shape, len(arr))
    tokenizer = Tokenizer.from_files(pkl_path, special_tokens=[SPECIAL_TOKEN])
    assert arr.max() < len(tokenizer.vocab)

if __name__ == "__main__":
    if len(sys.argv) != 4:
        raise IndexError("Provide correct arguments for the files")
    pkl_path, txt_path, out_npy_path = sys.argv[1], sys.argv[2], sys.argv[3]
    encode_file(pkl_path, txt_path, out_npy_path)
    # encode_file("data/tinystories_bpe.pkl", "data/TinyStoriesV2-GPT4-train.txt", "data/tinystories_train.npy")