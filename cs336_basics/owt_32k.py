import random
from .tokenizer import Tokenizer
random.seed(42)

ts_data = "/Users/rohanrajput/rohan_apps/cs336/assignments/assignment1-basics/data/tinystories_bpe.pkl"
tokenizer = Tokenizer.from_files(file_path=ts_data, special_tokens=["<|endoftext|>"])

valid_data_path = "/Users/rohanrajput/rohan_apps/cs336/assignments/assignment1-basics/data/owt_valid.txt"

with open(valid_data_path, "r") as f:
    data = f.read()
    chunks = data.split("<|endoftext|>")
    # first_10 = chunks[:10]
    first_10 = random.sample(chunks, 10)
total_bytes, total_tokens = 0, 0
for doc in first_10:
    if not doc.strip():
        continue
    doc = doc.strip()
    byte_len = len(doc.encode("utf-8"))
    tokens = len(tokenizer.encode(doc))
    total_bytes += byte_len
    total_tokens += tokens

ratio = total_bytes/total_tokens
print(ratio)


    
