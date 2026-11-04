"""Shared imports and numerical helpers; no models loaded at import time."""
from pathlib import Path
import base64
import gc
import hashlib
import io
import json
import math
import os
import platform
import random
import re
import shutil
import threading
import time
from importlib import metadata

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.transforms import RandomResizedCrop
from torchvision.transforms import functional as TF
from torchvision.datasets import ImageFolder
from PIL import Image, ImageOps


def l2_normalize(value):
    return F.normalize(value.float(), dim=-1)


def release_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
