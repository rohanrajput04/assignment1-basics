import numpy as np 
from .tokenizer import Tokenizer

def encode_file(pkl_path, txt_path, out_npy_path):
    tokenizer = Tokenizer.from_files(pkl_path, special_tokens=["<|endoftext|>"])
    
    with open(txt_path, "r") as f:
        arr = np.fromiter(tokenizer.encode_iterable(f), dtype=np.uint16)
        np.save(out_npy_path, arr)

        print(arr.shape, len(arr))
        assert arr.max() < len(tokenizer.vocab)

if __name__ == "__main__":
    pkl_path = "data/owt_bpe.pkl"
    txt_path = "data/owt_train.txt"
    out_npy_path = "data/owt_train.npy"
    encode_file(pkl_path, txt_path, out_npy_path)
