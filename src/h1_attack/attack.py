"""Launch H1 overlay in a fresh run root using the unchanged frozen kit."""
import argparse
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kit", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--variant", choices=("global", "concept"), default="global")
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--selftest-device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RuntimeError("Set CUBLAS_WORKSPACE_CONFIG before starting the worker")
    import staging as launch
    root = launch.prepare(args.kit, args.run_root, args.variant)
    policy = {"overlay_sha256": hashlib.sha256(Path(__file__).with_name("instrumentation.py").read_bytes()).hexdigest(),
              "launcher_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "variant": args.variant, "parent_kit_sha256": "0123fdbb21e238328fe95e7e7a82976fc4b7ec889f03a14dd42f30e276906cad",
              "seed": 2023, "deterministic": True, "resume_old_results": False}
    identity = root/"debug_identity.json"
    if identity.exists() and json.loads(identity.read_text()) != policy:
        raise RuntimeError("Debug identity changed; use a new run root")
    launch.write(identity, policy)
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(root/"author/m_attack_v2"))
    os.environ.update(WANDB_MODE="disabled", TOKENIZERS_PARALLELISM="false", HF_HUB_DISABLE_XET="1",
                      HF_HOME=os.environ.get("H1_SHARED_HF_CACHE", str(root.parent/"hf_cache")),
                      OMP_NUM_THREADS="4", MKL_NUM_THREADS="4")
    import generate_ad_sample_parallel as run
    import instrumentation as overlay
    overlay.install(run, root/"audit", 2023)
    if not args.selftest:
        launch.execute(root)
        return
    import torch
    from torch import nn
    from torchvision import transforms as T
    from omegaconf import OmegaConf
    from attack.base import AttackFramework
    from surrogates.loss import EnsWeightedMultiAlignmentLoss

    class ToyFeatures(nn.Module):
        def forward(self, image, return_dict=False):
            # Simple differentiable features without an unrelated nondeterministic CUDA pool backward.
            return image.flatten(1)[:, ::1024]

    torch.set_num_threads(4)
    torch.manual_seed(2023)
    source, target = torch.rand(2, 3, 224, 224, device=args.selftest_device), torch.rand(6, 3, 224, 224, device=args.selftest_device)
    cfg = OmegaConf.create({"optim": {"steps": 2, "alpha": .005, "epsilon": 16, "optimizer": "adam", "momentum": .9}})
    attacker = AttackFramework.create("pgd_multi_pass", cfg=cfg,
        ensemble_loss=EnsWeightedMultiAlignmentLoss([ToyFeatures() for _ in range(4)], beta=.3),
        source_crop=[T.RandomResizedCrop(224, scale=(.5, 1.)) for _ in range(3)],
        target_crop=T.Compose([T.RandomResizedCrop(224, scale=(.95, 1.)), T.RandomHorizontalFlip(), T.RandomRotation(10)]),
        change_iters=[40, 260])
    output = attacker.attack(img_index=0, image_org=source, image_tgt=target, target_num=3,
                             amp=False, log_wandb=False, log_interval=1, disable_tqdm=True, multi_pass_num=2)
    if not torch.isfinite(output).all() or (output-source).abs().max() == 0:
        raise RuntimeError("Synthetic integration attack produced invalid/unchanged output")
    if (output-source).abs().max() > 16/255+1e-6:
        raise RuntimeError("Synthetic integration attack exceeds epsilon")
    print("H1_OVERLAY_SYNTHETIC_INTEGRATION_PASS", flush=True)


if __name__ == "__main__":
    main()
