from cs336_basics.train_bpe_tinystories import train_bpe
import pickle
import time
import resource
if __name__ == "__main__":
    t = time.perf_counter()
    vocab, merges = train_bpe(input_path="data/TinyStoriesV2-GPT4-train.txt", vocab_size=10000, special_tokens=["<|endoftext|>"])
    with open("data/tinystories_bpe.pkl", "wb") as f:
      pickle.dump((vocab, merges), f)
    print(f"elapsed: {time.perf_counter() - t:.1f}s")
    print(f"peak memory: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9:.2f} GB")
    longest = max(vocab.values(), key=len)
    print(f"longest token ({len(longest)} bytes): {longest!r}")