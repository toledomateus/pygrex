"""
Some handy functions for pytroch model training ...
"""

import torch
from torch.optim import Optimizer


# Checkpoints
def save_checkpoint(model, model_dir):
    torch.save(model.state_dict(), model_dir)


def resume_checkpoint(model, model_dir, device_id):
    device = f"cuda:{device_id}"
    state_dict = torch.load(model_dir, map_location=device)
    model.load_state_dict(state_dict)


# Hyper params
def use_cuda(enabled, device_id=0):
    if enabled:
        assert torch.cuda.is_available(), "CUDA is not available"
        torch.cuda.set_device(device_id)


def use_optimizer(
    optimizer_name: str,
    network: torch.nn.Module,
    learning_rate: float,
    momentum: float = 0,
    weight_decay: float = 0,
    alpha: float = 0.99,
) -> Optimizer:
    if optimizer_name == "sgd":
        optimizer = torch.optim.SGD(
            network.parameters(),
            lr=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay,
        )

    elif optimizer_name == "adam":
        optimizer = torch.optim.Adam(
            network.parameters(), lr=learning_rate, weight_decay=weight_decay
        )

    elif optimizer_name == "rmsprop":
        optimizer = torch.optim.RMSprop(
            network.parameters(), lr=learning_rate, alpha=alpha, momentum=momentum
        )
    else:
        raise ValueError(f"Optimizer '{optimizer_name}' is not supported")

    return optimizer
