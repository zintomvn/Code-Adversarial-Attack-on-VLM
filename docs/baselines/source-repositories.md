# Baseline source repositories

The upstream repositories under `src/M-Attack-V2`, `src/MPCAttack` and
`src/O-Attack` are pinned Git submodules. Initialize them after cloning:

```sh
git submodule update --init --recursive
```

The local O-Attack checkout also contains research comments and whitespace changes
in `oattack/attacks.py`. They are preserved in
`docs/baselines/o-attack-local-annotations.patch`, separate from the upstream pin.
To reproduce those annotations from the repository root after initialization:

```sh
git -C src/O-Attack apply ../../docs/baselines/o-attack-local-annotations.patch
```

The annotations are already present in the original local checkout; apply the
patch only to an unmodified checkout. Portable campaign bundles have their own
source manifests in `notebooks/attacks/baseline/SOURCE_LOCK.json` and
`notebooks/attacks/baseline/om_attack/SOURCE_LOCK.json`.
