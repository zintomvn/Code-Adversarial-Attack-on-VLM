"""Editable configuration groups matching MPCAttack/config_schema.py."""
from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class WandbConfig:
    entity: str = "???"
    project: str = "local_adversarial_attack"


@dataclass
class BlackboxConfig:
    model_name: str = "gpt4o"
    batch_size: int = 1
    timeout: int = 30


@dataclass
class Blackbox_open_source_Config:
    model_name: str = "Qwen2.5-VL-3B-Instruct"
    batch_size: int = 1
    timeout: int = 30


@dataclass
class DataConfig:
    batch_size: int = 1
    num_samples: int = 2
    cle_data_path: str = "resources/images/bigscale"
    tgt_data_path: str = "resources/images/target_images"
    output: str = "./My-Attack-V3"


@dataclass
class OptimConfig:
    alpha: float = 1.0
    epsilon: float = 16.0
    steps: int = 300


@dataclass
class ModelConfig:
    input_res: int = 224
    use_source_crop: bool = True
    use_target_crop: bool = False
    crop_scale: Tuple[float, float] = (0.5, 0.9)
    ensemble: bool = True
    device: str = "cuda:0"
    backbone: List[str] = field(default_factory=lambda: ["B16", "B32", "Laion"])


@dataclass
class TrainConfig:
    batch_size: int = 1
    lr: float = 0.03
    epochs: int = 300
    eps: int = 16


@dataclass
class NegativeConfig:
    """Only added attack component: the notebook's counterfactual caption bank."""
    num_negatives: int = 10
    model_id: str = "Qwen/Qwen3-VL-8B-Instruct"
    caption_attempts: int = 80
    negative_retries: int = 10
    bank_path: str = ""


@dataclass
class MainConfig:
    data: DataConfig = field(default_factory=DataConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)
    blackbox: BlackboxConfig = field(default_factory=BlackboxConfig)
    blackbox_open_source: Blackbox_open_source_Config = field(default_factory=Blackbox_open_source_Config)
    train: TrainConfig = field(default_factory=TrainConfig)
    negative: NegativeConfig = field(default_factory=NegativeConfig)
    attack: str = "mifgsm"
    lam: float = 0.6
    tau: float = 0.2
    omega: float = 2.0
    seed: int = 2026


@dataclass
class Ensemble3ModelsConfig(MainConfig):
    model: ModelConfig = field(default_factory=ModelConfig)


def register_configs():
    """Optional Hydra registration, without requiring Hydra for argparse/--help."""
    from hydra.core.config_store import ConfigStore
    store = ConfigStore.instance()
    store.store(name="config", node=MainConfig)
    store.store(name="ensemble_3models", node=Ensemble3ModelsConfig)
