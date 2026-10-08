import copy
import random

import numpy as np
import torch
import yaml


def log_line(log_path, msg):
    """Print and append to a log file, so progress survives a dropped client connection."""
    import datetime
    line = f"{datetime.datetime.now():%Y-%m-%d %H:%M:%S} {msg}"
    print(line, flush=True)
    with open(log_path, "a") as f:
        f.write(line + "\n")


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def amp_settings(cfg, device):
    """(enabled, dtype) for autocast. fp16 needs a GradScaler; bf16 (H100/H200/A100) does not."""
    enabled = cfg.get("amp", True) and device.type == "cuda"
    dtype = torch.bfloat16 if cfg.get("amp_dtype", "float16") == "bfloat16" else torch.float16
    return enabled, dtype


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class EMA:
    """Exponential moving average "teacher" model for stabilizing stage-2
    unsupervised updates (student trains normally, teacher is a slow-moving
    average of the student's weights)."""

    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.teacher = copy.deepcopy(model)
        for p in self.teacher.parameters():
            p.requires_grad = False
        self.teacher.eval()

    @torch.no_grad()
    def update(self, student):
        for t_param, s_param in zip(self.teacher.parameters(), student.parameters()):
            t_param.data.mul_(self.decay).add_(s_param.data, alpha=1 - self.decay)
        for t_buf, s_buf in zip(self.teacher.buffers(), student.buffers()):
            t_buf.data.copy_(s_buf.data)


class AverageMeterDict:
    """Tracks running averages of a dict of scalar losses, for logging."""

    def __init__(self):
        self.sums = {}
        self.counts = {}

    def update(self, loss_dict, n=1):
        for k, v in loss_dict.items():
            val = v.item() if hasattr(v, "item") else v
            self.sums[k] = self.sums.get(k, 0.0) + val * n
            self.counts[k] = self.counts.get(k, 0) + n

    def averages(self):
        return {k: self.sums[k] / self.counts[k] for k in self.sums}

    def __str__(self):
        avgs = self.averages()
        return " | ".join(f"{k}={v:.4f}" for k, v in avgs.items())
