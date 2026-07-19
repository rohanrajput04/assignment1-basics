import random
from .tokenizer import Tokenizer
random.seed(42)
import time

ts_data = "/Users/rohanrajput/rohan_apps/cs336/assignments/assignment1-basics/data/tinystories_bpe.pkl"
tokenizer = Tokenizer.from_files(file_path=ts_data, special_tokens=["<|endoftext|>"])

data_path = "/Users/rohanrajput/rohan_apps/cs336/assignments/assignment1-basics/data/tinystories_10mb.txt"

with open(data_path, "r") as f:
    data = f.read()
    chunks = data.split("<|endoftext|>")
    # first_10 = chunks[:10]
    first_10 = random.sample(chunks, 1000)
total_bytes, total_tokens = 0, 0
elapsed_time = 0
for doc in first_10:
    if not doc.strip():
        continue
    doc = doc.strip()
    
    byte_len = len(doc.encode("utf-8"))
    
    start_time = time.perf_counter()
    tokenize = tokenizer.encode(doc)
    end_time = time.perf_counter()
    tokens = len(tokenize)
    total_bytes += byte_len
    total_tokens += tokens
    elapsed_time += (end_time - start_time)
ratio = total_bytes/total_tokens
throughput = total_bytes/ elapsed_time
print(f"throughput : {throughput}")
pile_bytes = 825 * (1024 ** 3)
pile_seconds = pile_bytes / throughput
days_need = pile_seconds/ (86400)
print(f"days_need : {days_need}")

    
