"""
Modal launcher for CS336 TinyStories training.

This is infrastructure only: it packages your existing `cs336_basics` code,
mounts your pre-tokenized data on a persistent Modal Volume, and runs your
`cs336_basics.train.main(...)` on a cloud GPU with device="cuda".

Your training/model code is NOT modified — this wrapper just calls it.

--------------------------------------------------------------------------
ONE-TIME SETUP
--------------------------------------------------------------------------
1. Authenticate:
       uv run modal setup

2. Create the volume and upload the TinyStories data + tokenizer into it.
   The volume's internal layout must mirror `./data/...` because your
   train.py reads "./data/tinystories_train.npy" etc.

       uv run modal volume create cs336-data
       uv run modal volume put cs336-data data/tinystories_train.npy  data/tinystories_train.npy
       uv run modal volume put cs336-data data/tinystories_valid.npy  data/tinystories_valid.npy
       uv run modal volume put cs336-data data/tinystories_bpe.pkl    data/tinystories_bpe.pkl

--------------------------------------------------------------------------
RUN
--------------------------------------------------------------------------
   # single run with your default config from train.py
   uv run modal run modal_train.py

   # override hyperparameters (JSON), e.g. a learning-rate point
   uv run modal run modal_train.py --opt-overrides '{"lr_max": 3e-3}'

   # a learning-rate sweep (runs several points back to back)
   uv run modal run modal_train.py::sweep_lr

Checkpoints and logs land back on the volume under /workspace and persist.
"""

import json
import os

import modal

# --- image: python 3.12 + your training deps, plus the local package ------
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch~=2.11.0",
        "numpy>=2.4",
        "einops>=0.8",
        "jaxtyping>=0.3",
        "regex>=2024.0.0",
        "wandb>=0.17",
    )
    # ship your package into the image so `import cs336_basics` works
    .add_local_python_source("cs336_basics")
)

# --- persistent storage for data + checkpoints ----------------------------
# Mounted at /workspace so that cwd-relative "./data/..." in your train.py
# resolves to /workspace/data/... on the volume, and checkpoints written to
# "./tinystories_latest.pt" persist at /workspace/tinystories_latest.pt.
volume = modal.Volume.from_name("cs336-data", create_if_missing=True)

app = modal.App("cs336-tinystories")

WORKSPACE = "/workspace"


def _run(model_overrides: dict | None, opt_overrides: dict | None):
    """Call the student's train.main() with device=cuda and merged config."""
    import torch

    # your default config lives in train.py; merge overrides on top
    from cs336_basics.train import main, model_size, optimizer_param

    ms = {**model_size, **(model_overrides or {})}
    op = {**optimizer_param, **(opt_overrides or {})}

    # cwd so "./data/..." and the checkpoint path resolve onto the volume
    os.chdir(WORKSPACE)

    assert torch.cuda.is_available(), "no CUDA device visible inside the container"
    print(f"[modal] cuda device: {torch.cuda.get_device_name(0)}")
    print(f"[modal] model_size={ms}")
    print(f"[modal] optimizer_param={op}")

    main(model_size=ms, optimizer_param=op, device="cuda")

    # flush checkpoint(s) written to the volume so they survive teardown
    volume.commit()


@app.function(
    image=image,
    gpu="B200",          # leaderboard rules require B200
    volumes={WORKSPACE: volume},
    timeout=60 * 60,     # 1 hour cap; raise if you train longer
    # injects WANDB_API_KEY as an env var; wandb picks it up automatically.
    # create it once with:  uv run modal secret create wandb WANDB_API_KEY=<key>
    secrets=[modal.Secret.from_name("wandb")],
)
def train(model_overrides_json: str = "{}", opt_overrides_json: str = "{}"):
    _run(json.loads(model_overrides_json), json.loads(opt_overrides_json))


@app.local_entrypoint()
def main(model_overrides: str = "{}", opt_overrides: str = "{}"):
    """Single run. Pass --opt-overrides '{\"lr_max\": 3e-3}' to change config."""
    train.remote(model_overrides, opt_overrides)


@app.local_entrypoint()
def sweep_lr():
    """learning_rate experiment: run several lr_max points back to back."""
    for lr in [1e-4, 3e-4, 1e-3, 3e-3, 1e-2]:
        print(f"=== launching lr_max={lr} ===")
        train.spawn("{}", json.dumps({"lr_max": lr, "lr_min": lr / 100}))


@app.local_entrypoint()
def sweep_batch():
    """batch_size_experiment: run several batch sizes back to back.

    The deliverable asks to include 64 and 128 and vary from 1 upward. Large
    batches often benefit from a higher LR, so we scale lr_max mildly with batch
    size (a simple linear-ish heuristic); adjust if a run under/over-shoots.
    """
    for bs in [1, 16, 64, 128, 256]:
        lr = 1e-3 * (bs / 32) ** 0.5   # sqrt scaling of LR with batch size
        print(f"=== launching batch_size={bs} (lr_max={lr:.2e}) ===")
        train.spawn(
            "{}",
            json.dumps({"batch_size": bs, "lr_max": lr, "lr_min": lr / 100}),
        )
