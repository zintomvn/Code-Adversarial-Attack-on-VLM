from pathlib import Path
import gc
import json
import math
import random
import re
from collections.abc import Sequence
from contextlib import redirect_stdout
from io import StringIO

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image, ImageOps
from torch import Tensor
from torchvision.datasets import ImageFolder
from tqdm.auto import tqdm


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def release_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def write_json(path, record):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def image_to_tensor(image):
    data = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    return torch.from_numpy(data).permute(2, 0, 1).unsqueeze(0).float()


def tensor_to_image(tensor):
    tensor = tensor.detach().cpu().squeeze(0).clamp(0, 255).round().byte()
    return Image.fromarray(tensor.permute(1, 2, 0).numpy())


def l2_normalize(x):
    return F.normalize(x.float(), dim=-1)

