import regex as re
from collections.abc import Iterator, Iterable
from .train_bpe import PAT, per_word_logic
import pickle

class Tokenizer:

    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None =None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens if special_tokens else [] 

        # create lookup
        self.byte_to_ids = {v:k for k,v in self.vocab.items()}
        # merge rank
        self.merge_rank = {ele: idx for idx, ele in enumerate(self.merges)}

        # add self.special_tokens 
        self.special_tokens.sort(key=len, reverse=True)
        for st in self.special_tokens:
            value = bytes(st, encoding='utf-8')
            if value not in self.byte_to_ids:
                key = len(self.vocab)
                self.vocab[key] = value
                self.byte_to_ids[value] = key

        # check in encode 
        self.special_tokens_set = set(self.special_tokens)
        

    @classmethod
    def from_files(cls, file_path: str, special_tokens: list[str]| None=None):
        with open(file_path, "rb") as f:
            loaded = pickle.load(f)
            vocab, merges = loaded

        return cls(vocab, merges, special_tokens)
    

    def encode(self, text: str) -> list[int]:
        
        result_ids = []
        if self.special_tokens:
            pattern = "(" + "|".join(re.escape(tok) for tok in self.special_tokens) + ")"
            chunks = re.split(pattern, text)
        else:
            chunks = [text]
        
        for chunk in chunks:
            if chunk in self.special_tokens_set:
                value = bytes(chunk, encoding='utf-8')
                token_id = self.byte_to_ids[value]
                result_ids.append(token_id)
            else:
                pre_tokens = re.findall(PAT, chunk)
                for pre_token in pre_tokens:
                    word = tuple(bytes([c]) for c in bytes(pre_token, encoding='utf-8'))
                    
                    while True:
                        rank_dict = {}
                        for w1, w2 in zip(word, word[1:]):
                            pair = tuple([w1, w2])
                            pair_rank = self.merge_rank.get(pair, float('inf'))
                            rank_dict[pair] = pair_rank
                        if rank_dict:
                            winner, winner_rank = min(rank_dict.items(), key=lambda x: x[1])
                        else:
                            break
                        if winner_rank == float('inf'):
                            break
                        word = per_word_logic(word, winner[0], winner[1])
                    result_ids.extend([self.byte_to_ids[ele] for ele in word])
                    



        return result_ids    


    def encode_iterable(self,iterable: Iterable[str]) -> Iterator[int]:
        
        for line in iterable:
            yield from self.encode(line)

    def decode(self, ids: list[int]) -> str:
        
        byte_list = [self.vocab[ele] for ele in ids]
        return b"".join(byte_list).decode('utf-8',errors='replace')


