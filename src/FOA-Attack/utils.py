"""Shared utilities for adversarial attack and text generation models."""

import os
import json
import yaml
import hashlib
import base64
from typing import Dict, Any, List, Union
from omegaconf import OmegaConf
import wandb
from config_schema import MainConfig


def load_api_keys() -> Dict[str, str]:
    """Load API keys from the api_keys file.
    
    Returns:
        Dict[str, str]: Dictionary containing API keys for different models
        
    Raises:
        FileNotFoundError: If no api_keys file is found
    """
    for ext in ['yaml', 'yml', 'json']:
        file_path = f'api_keys.{ext}'
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                if ext in ['yaml', 'yml']:
                    return yaml.safe_load(f)
                else:
                    return json.load(f)
    
    raise FileNotFoundError(
        "API keys file not found. Please create api_keys.yaml, api_keys.yml, or api_keys.json "
        "in the root directory with your API keys."
    )


def get_api_key(model_name: str) -> str:
    """Get API key for specified model.
    
    Args:
        model_name: Name of the model to get API key for
        
    Returns:
        str: API key for the specified model
        
    Raises:
        KeyError: If API key for model is not found
    """
    api_keys = load_api_keys()
    if model_name not in api_keys:
        raise KeyError(
            f"API key for {model_name} not found in api_keys file. "
            f"Available models: {list(api_keys.keys())}"
        )
    return api_keys[model_name]


def hash_training_config(cfg: MainConfig) -> str:
    """Create a deterministic hash of training-relevant config parameters.
    
    Args:
        cfg: Configuration object containing model settings
        
    Returns:
        str: MD5 hash of the config parameters
    """
    # Convert backbone list to plain Python list
    if isinstance(cfg.model.backbone, (list, tuple)):
        backbone = list(cfg.model.backbone)
    else:
        backbone = OmegaConf.to_container(cfg.model.backbone)
        
    # Create config dict with converted values
    train_config = {
        "data": {
            "batch_size": int(cfg.data.batch_size),
            "num_samples": int(cfg.data.num_samples),
            "cle_data_path": str(cfg.data.cle_data_path),
            "tgt_data_path": str(cfg.data.tgt_data_path),
        },
        "optim": {
            "alpha": float(cfg.optim.alpha),
            "epsilon": int(cfg.optim.epsilon),
            "steps": int(cfg.optim.steps),
        },
        "model": {
            "input_res": int(cfg.model.input_res),
            "use_source_crop": bool(cfg.model.use_source_crop),
            "use_target_crop": bool(cfg.model.use_target_crop),
            "crop_scale": tuple(float(x) for x in cfg.model.crop_scale),
            "ensemble": bool(cfg.model.ensemble),
            "backbone": backbone,
        },
        "attack": cfg.attack,
    }
    
    # Convert to JSON string with sorted keys
    json_str = json.dumps(train_config, sort_keys=True)
    return hashlib.md5(json_str.encode()).hexdigest()


def setup_wandb(cfg: MainConfig, tags=None) -> None:
    """Initialize Weights & Biases logging.
    
    Args:
        cfg: Configuration object containing wandb settings
    """
    config_dict = OmegaConf.to_container(cfg, resolve=True)
    wandb.init(
        project=cfg.wandb.project,
        config=config_dict,
        tags=tags,
    )


def encode_image(image_path: str) -> str:
    """Encode image file to base64 string.
    
    Args:
        image_path: Path to image file
        
    Returns:
        str: Base64 encoded image string
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def ensure_dir(path: str) -> None:
    """Ensure directory exists, create if it doesn't.
    
    Args:
        path: Directory path to ensure exists
    """
    os.makedirs(path, exist_ok=True)


def get_output_paths(cfg: MainConfig, config_hash: str) -> Dict[str, str]:
    """Get dictionary of output paths based on config.
    
    Args:
        cfg: Configuration object
        config_hash: Hash of training config
        
    Returns:
        Dict[str, str]: Dictionary containing output paths
    """
    return {
        'output_dir': os.path.join(cfg.data.output, "img", config_hash),
        'desc_output_dir': os.path.join(cfg.data.output, "description", config_hash)
    }


# Versioned pipeline extensions. The original utility functions above are unchanged.
class ExperimentRunner:
    """Common artifact contract for baseline, ablations and downstream evaluation."""

    def __init__(self, cfg, input_path=None, output=None):
        from pathlib import Path
        from datetime import datetime, timezone
        import subprocess
        import sys
        self.cfg = cfg
        self.input_path = Path(input_path or cfg.runtime.input).resolve() if (input_path or cfg.runtime.input) else None
        self.config = OmegaConf.to_container(cfg, resolve=True)
        self.fingerprint = hashlib.sha256(json.dumps(self.config, sort_keys=True).encode()).hexdigest()
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        self.root = Path(output or cfg.runtime.output).resolve() / f'{cfg.experiment.name}_{stamp}'
        self.root.mkdir(parents=True, exist_ok=False)
        (self.root / 'config_resolved.yaml').write_text(OmegaConf.to_yaml(cfg, resolve=True), encoding='utf-8')
        # Snapshot only implementation/config files: never copy API keys or datasets.
        code = self.root / 'code'
        code.mkdir()
        import shutil
        module_root = Path(__file__).resolve().parent
        for filename in module_root.rglob('*.py'):
            if any(part.startswith('.') for part in filename.relative_to(module_root).parts):
                continue
            dest = code / filename.relative_to(module_root)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(filename, dest)
        shutil.copytree(module_root / 'config', code / 'config')
        freeze = subprocess.run([sys.executable, '-m', 'pip', 'freeze'], capture_output=True, text=True)
        (self.root / 'requirements-resolved.txt').write_text(freeze.stdout, encoding='utf-8')
        self.write_json('run.json', {'config_sha256': self.fingerprint, 'experiment': self.config['experiment'],
                                    'python': sys.version, 'input': str(self.input_path) if self.input_path else None})
        print(f'Output: {self.root}', flush=True)

    @staticmethod
    def sha256(path):
        digest = hashlib.sha256()
        with open(path, 'rb') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(block)
        return digest.hexdigest()

    def write_json(self, filename, value):
        (self.root / filename).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')

    def append(self, filename, row):
        with (self.root / filename).open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
            stream.flush()

    def read_input(self, filename):
        """Select exactly one artifact; reject accidental mixing of experiments/models."""
        if self.input_path is None:
            raise ValueError('Provide --input pointing to the extracted previous-stage output directory.')
        candidates = [self.input_path] if self.input_path.is_file() else list(self.input_path.rglob(filename))
        if len(candidates) != 1 or candidates[0].name != filename:
            raise ValueError(f'Expected exactly one {filename} under {self.input_path}; found {len(candidates)}. Select a specific run directory.')
        path = candidates[0]
        rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
        if filename == 'manifest.jsonl':
            status_path = path.parent / 'status.json'
            if not status_path.is_file():
                raise ValueError('Attack run has no completion marker; do not evaluate an interrupted run.')
            status = json.loads(status_path.read_text(encoding='utf-8'))
            if status.get('status') != 'complete' or status.get('samples') != len(rows):
                raise ValueError('Attack manifest is incomplete or sample count differs from its completion marker.')
        ids = [row['sample_id'] for row in rows]
        if not rows or len(set(ids)) != len(ids):
            raise ValueError('Input is empty or has duplicate sample IDs.')
        self.write_json('input_provenance.json', {'file': str(path), 'sha256': self.sha256(path), 'samples': len(rows)})
        return path.parent, rows

    def run(self):
        raise NotImplementedError


class KaggleRuntime:
    """Thin notebook facade; setup, dispatch and export live in Python, not cells."""

    @staticmethod
    def config(name):
        from pathlib import Path
        from hydra import compose, initialize_config_dir
        if not name or any(char in name for char in '/\\.'):
            raise ValueError('Use a config name without a path or .yaml extension.')
        with initialize_config_dir(version_base=None, config_dir=str(Path(__file__).resolve().parent / 'config')):
            cfg = compose(config_name=name)
        import re
        if not re.fullmatch(r'v[1-9][0-9]*', cfg.experiment.version):
            raise ValueError('experiment.version must be v1, v2, ...')
        return cfg

    @staticmethod
    def secret(name):
        value = os.environ.get(name)
        if value:
            return value
        try:
            from kaggle_secrets import UserSecretsClient
            value = UserSecretsClient().get_secret(name)
        except Exception:
            raise RuntimeError(f'Set environment variable or enable Kaggle Secret {name}.') from None
        if not value:
            raise RuntimeError(f'Empty secret: {name}')
        return value

    @staticmethod
    def environment(cfg):
        from pathlib import Path
        return Path(cfg.runtime.environment_root) / cfg.environment.name

    @classmethod
    def setup(cls, cfg):
        import subprocess
        import sys
        import venv
        env = cls.environment(cfg)
        venv.EnvBuilder(with_pip=True, system_site_packages=True).create(env)
        executable = env / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
        subprocess.run([str(executable), '-m', 'pip', 'install', *cfg.environment.packages], check=True)
        print(f'Environment ready: {executable}')

    @classmethod
    def execute(cls, cfg, args):
        if args.output:
            cfg.runtime.output = args.output
        if args.input:
            cfg.runtime.input = args.input
        if cfg.stage == 'attack':
            if args.source:
                cfg.data.cle_data_path = args.source
            if args.target:
                cfg.data.tgt_data_path = args.target
            from generate_adversarial_samples_foa_attack import FOABaselineRunner, FOAClusterAblationRunner
            runners = {'foa_baseline': FOABaselineRunner, 'foa_cluster_ablation': FOAClusterAblationRunner}
            runner = runners[cfg.experiment.method](cfg, args.input, args.output)
            runner.run(source=args.source, target=args.target)
        elif cfg.stage == 'captioning':
            from blackbox_text_generation import CaptionExperimentRunner
            CaptionExperimentRunner(cfg, args.input, args.output).run()
        elif cfg.stage == 'evaluation':
            from gpt_evaluate import FOAPaperEvaluationRunner
            if cfg.evaluation.protocol != 'foa_paper_gpt4o_v2':
                raise ValueError(f'Unknown evaluation protocol: {cfg.evaluation.protocol}')
            FOAPaperEvaluationRunner(cfg, args.input, args.output).run()
        else:
            raise ValueError(f'Unknown stage: {cfg.stage}')

    @classmethod
    def main(cls):
        import argparse
        import subprocess
        import sys
        from pathlib import Path
        parser = argparse.ArgumentParser(description='Versioned FOA pipeline; legacy entry points are unchanged.')
        parser.add_argument('action', choices=['setup', 'run', 'worker', 'export', 'show'])
        parser.add_argument('--config-name', required=True)
        parser.add_argument('--input')
        parser.add_argument('--source')
        parser.add_argument('--target')
        parser.add_argument('--output')
        args = parser.parse_args()
        cfg = cls.config(args.config_name)
        if args.action == 'setup':
            cls.setup(cfg)
        elif args.action == 'show':
            print(OmegaConf.to_yaml(cfg, resolve=True))
        elif args.action == 'run':
            executable = cls.environment(cfg) / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
            if not executable.is_file():
                raise RuntimeError('Run setup with this config first.')
            subprocess.run([str(executable), str(Path(__file__).resolve()), 'worker', *sys.argv[2:]], check=True)
        elif args.action == 'worker':
            cls.execute(cfg, args)
        else:
            import zipfile
            root = Path(args.output or cfg.runtime.output).resolve()
            runs = sorted((p for p in root.glob(f'{cfg.experiment.name}_*') if p.is_dir()), key=lambda p: p.name)
            if not runs:
                raise FileNotFoundError(f'No output for {cfg.experiment.name} under {root}')
            run = runs[-1]
            archive = root / (run.name + '.zip')
            with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED) as stream:
                for path in run.rglob('*'):
                    if path.is_file():
                        stream.write(path, arcname=str(path.relative_to(run)))
            print(f'Download: {archive}')


if __name__ == '__main__':
    KaggleRuntime.main()
