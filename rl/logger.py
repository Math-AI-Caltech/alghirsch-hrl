"""
logger.py
---------
Logging/checkpointing utilities used by the trainers.
"""
from typing import (
    Optional,
    List,
    Dict
)
import os

import torch
import torch.nn as nn

class Logger:
    def log(self, *args):
        raise NotImplementedError()

class Callback:
    def callback(self, epoch, model, *args):
        raise NotImplementedError()

class WBLogger(Logger):
    def __init__(self, **kwargs):
        import wandb
        self._run = wandb.init(
            entity  = kwargs.get("wandb_entity"),
            project = kwargs.get("wandb_project"),
            name    = kwargs.get("run_name"),
            config  = kwargs.get("config"),
            group   = kwargs.get("group", None))

    @property
    def run(self): return self._run

    def log(self, history: Dict, step: Optional[int] = None):
        self._run.log({k:v[-1] for k,v in history.items()}, step = step)

class CheckpointerCallback(Callback):
    def __init__(self, path_to_checkpoints: str, checkpoint_freq: int):
        self._path_to_checkpoints = path_to_checkpoints
        self._checkpoint_freq = checkpoint_freq
        if not os.path.exists(path_to_checkpoints):
            os.makedirs(path_to_checkpoints, exist_ok = True)

    def callback(self, epoch: int, model: nn.Module | List, history: Dict, *args):
        if epoch % self._checkpoint_freq != 0: return
        if model is not None:
            if isinstance(model, nn.Module):
                torch.save(model.state_dict(), os.path.join(self._path_to_checkpoints, f"model_weights_epoch{epoch}.pth"))
            else:
                for i, m in enumerate(model):
                    torch.save(m.state_dict(), os.path.join(self._path_to_checkpoints, f"model{i}_weights_epoch{epoch}.pth"))
        if history is not None:
            torch.save(history, os.path.join(self._path_to_checkpoints, "history.pth"))

        for i, e in enumerate(args):
            if isinstance(e, List):
                torch.save(e, os.path.join(self._path_to_checkpoints, f"arg_{epoch}_{i}.pth"))
