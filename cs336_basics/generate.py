from .model import load_checkpoint, TransformerLM, AdamW, softmax
from .tokenizer import Tokenizer
import torch


device = "mps"
max_new_tokens = 256
context_length = 256
temprature = 1.0
top_p = 0.9
with torch.no_grad():
    model = TransformerLM(vocab_size=10000, context_length=256, d_model=512, num_heads=16,d_ff=1344,num_layers=4,rope_theta=10000)
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=1e-3)
    iteration = load_checkpoint(src="checkpoints/tinystories_latest.pt", model=model, optimizer=optimizer)


    print(sum(p.sum() for p in model.parameters()))
    model.eval()

    tokenize = Tokenizer.from_files(file_path="data/tinystories_bpe.pkl", special_tokens=["<|endoftext|>"])

    eot_id = tokenize.byte_to_ids[b"<|endoftext|>"]

    test_string = "Once upon a time"
    encoded_list = tokenize.encode(test_string)

    encoded_tensor = torch.tensor(encoded_list, device=device, dtype=torch.long)

    encoded_tensor = encoded_tensor.unsqueeze(0)
    for _ in range(max_new_tokens):
        
        x = encoded_tensor[:, -context_length:]
        
        # forward 
        logits = model(x)

        # last position pred 
        last_logit = logits[:, -1, :]

        # scaled temprature 
        scaled = last_logit/temprature

        # prob 
        probs = softmax(scaled, dim=-1)
        # sort for top p
        sorted_prob, sorted_idx = torch.sort(probs, descending=True, dim=-1)

        # cumsum 
        cumsum = torch.cumsum(sorted_prob, dim=-1)

        # removal mask 
        mask = (cumsum - sorted_prob) >= top_p
        # keep only top p
        sorted_prob[mask] = 0

        sorted_prob = sorted_prob / sorted_prob.sum(dim=-1, keepdim=True)

        sampled = torch.multinomial(sorted_prob, num_samples=1)

        next_token = torch.gather(sorted_idx, -1, sampled)


        # append to encoded tensor 
        encoded_tensor = torch.cat([encoded_tensor, next_token], dim=1)

        if next_token.item() == eot_id:
            break
    print(tokenize.decode(encoded_tensor[0].tolist()))


