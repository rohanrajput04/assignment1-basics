import numpy as np
from .model import TransformerLM, AdamW, cross_entropy, get_lr_cosine_schedule, gradient_cliping, get_batch, save_checkpoint, load_checkpoint
from einops import rearrange
import torch
import time
import wandb
model_size = {
    "d_model" : 512,
    "num_layers" : 4,
    "num_heads" : 16,
    "d_ff" : 1344,
    "vocab_size" : 10000,
    "context_length": 256,
    "rope_theta" : 10000, 
    "use_rope" : True,
    "ffn_type" : "swiglu",
    "norm_position" : "pre",
    "use_rmsnorm" : True,
    "train_path" : "./data/tinystories_train.npy",
    "valid_path" : "./data/tinystories_valid.npy"
}

optimizer_param = {
    "lr_max" : 1e-3,
    "lr_min" : 1e-5,
    "warmup_iters" : 500,
    "cosine_cycle_iters" : 5000,
    "batch_size": 32, 
    "total_iters" : 5000,
    "weight_decay" : 0.01,
    "max_l2_norm" : 1.0,

}

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"device : {device}")


def main(model_size, optimizer_param, device):
    # load data 
    train_path = model_size["train_path"]
    valid_path = model_size["valid_path"]

    train_data = np.load(train_path, mmap_mode="r")
    valid_data = np.load(valid_path, mmap_mode='r')

    model = TransformerLM(vocab_size=model_size["vocab_size"], context_length=model_size["context_length"], d_model=model_size["d_model"],
    num_heads=model_size["num_heads"], num_layers=model_size["num_layers"],d_ff=model_size["d_ff"], rope_theta=model_size["rope_theta"], use_rope=model_size['use_rope'], ffn_type=model_size['ffn_type'],
    norm_position=model_size['norm_position'], use_rmsnorm=model_size["use_rmsnorm"])

    model.to(device=device)
    model = torch.compile(model)
    optim = AdamW(model.parameters(), lr=optimizer_param["lr_min"])

    name = f"vocab={model_size['vocab_size']}_fffn={model_size['ffn_type']}_rope={model_size['use_rope']}_norm={model_size['norm_position']}_ln={model_size['use_rmsnorm']}_lr={optimizer_param['lr_max']}"
    wandb.init(project="cs336-tinystories", name=name, config={**model_size, **optimizer_param})
    start = time.time()
    for it in range(optimizer_param['total_iters']):
        train_input_batch, train_target_batch = get_batch(train_data, batch_size=optimizer_param["batch_size"], context_length=model_size["context_length"], device=device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(train_input_batch)
            logits = rearrange(logits, "batch seq vocab -> (batch seq) vocab")
            train_target_batch = rearrange(train_target_batch, "batch seq -> (batch seq)")
            loss = cross_entropy(logits, train_target_batch)
        optim.zero_grad()
        loss.backward()
        gradient_cliping(model.parameters(), max_l2_norm=optimizer_param["max_l2_norm"])
        lr_cosine_schedule = get_lr_cosine_schedule(t=it, alpha_max=optimizer_param["lr_max"], alpha_min=optimizer_param["lr_min"], T_w=optimizer_param["warmup_iters"], T_c=optimizer_param["cosine_cycle_iters"])
        for group in optim.param_groups:
            group["lr"] = lr_cosine_schedule
        optim.step()

        if it % 100 == 0:
            x_val, y_val = get_batch(valid_data, batch_size=optimizer_param["batch_size"], context_length=model_size["context_length"], device=device)
            model.eval()
            with torch.no_grad():
                val_logits = model(x_val)
                val_logits = rearrange(val_logits, "batch seq vocab -> (batch seq) vocab")
                y_val = rearrange(y_val, "batch seq -> (batch seq)")
                valid_loss = cross_entropy(val_logits, y_val)
                wandb.log({"train_loss": loss.item(), "val_loss": valid_loss.item(), "wall_clock_s": time.time() - start}, step=it)
                print(f"Validation loss for {it} : {valid_loss.item()}")
            model.train()  
            print(f"Training loss for {it} : {loss.item()}")
            save_checkpoint(model=model, optimizer=optim,iteration=it,out=f"tinystories_latest.pt")
    wandb.finish()

        

if __name__ == "__main__":
    main(model_size=model_size, optimizer_param=optimizer_param, device=device)



    
