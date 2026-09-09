import hashlib
import importlib.metadata
import json
import os
import platform
import random
import subprocess
from pathlib import Path

import numpy as np
import torch
import torchvision

CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


def seed_all(seed):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def environment():
    try:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    return dict(python=platform.python_version(), torch=str(torch.__version__),
                torchvision=str(torchvision.__version__), numpy=np.__version__,
                cuda=torch.version.cuda, platform=platform.platform(), git_commit=commit,
                packages={name: importlib.metadata.version(name) for name in ['foolbox', 'eagerpy', 'scipy', 'matplotlib']})


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
