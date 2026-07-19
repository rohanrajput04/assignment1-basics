from cs336_basics.model import SGD
import torch
if __name__ == "__main__":
    weights = torch.nn.Parameter(5 * torch.randn((10,10)))
    opt = SGD([weights], lr=1e3)

    for t in range(10):
        opt.zero_grad()
        loss = (weights**2).mean()
        print(loss.cpu().item())
        loss.backward()
        opt.step()