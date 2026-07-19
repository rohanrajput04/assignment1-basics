import pickle
v, m = pickle.load(open("data/owt_bpe.pkl", "rb"))
print(f"vocab: {len(v)}, merges: {len(m)}")
print(f"longest: {max(v.values(), key=len)!r}")