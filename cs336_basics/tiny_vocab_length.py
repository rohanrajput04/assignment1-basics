
from .tokenizer import Tokenizer


ts_data = "/Users/rohanrajput/rohan_apps/cs336/assignments/assignment1-basics/data/tinystories_bpe.pkl"
tokenizer = Tokenizer.from_files(file_path=ts_data, special_tokens=["<|endoftext|>"])
vocab = tokenizer.vocab
byte_value = max(vocab.values(), key=len)
print(len(byte_value))
print(repr(byte_value))
str_value = byte_value.decode("utf-8", errors='replace')
print(str_value)