# My-Attack · v5.3 Diffusion Regions

## Google Drive storage repair (2026-10-29)

Real runs now require `--drive-folder-id`. The approved target is
`1daramuahKwdAVGQtsDZTzy1v0OSG-kiI`; outputs go into `My-Attack/<run_id>`.
The old MoLab native-cache-only run lost payloads across a sandbox change; it is not resumed
from source-only completion markers. R2 reruns exactly200 confirmation pairs, IDs200..399,
with model revisions/raw pair hashes locked from the collected R1 identity.

```bash
python drive_probe.py prime
# A second process restores into a brand-new empty temporary directory from Drive only:
python drive_probe.py restore
python molab_bootstrap.py --launch --identity-lock identity_lock.json
```

The Python environment must use the existing MoLab Google Drive connection (`gdrive-fsspec`,
user OAuth `token='cache'`). No credentials are written into source or uploaded with results.
If that connection is absent, the storage preflight must stop before model inference.

Drive layout: `chunks/` incremental changed-file ZIPs, `indices/` immutable file manifests,
and readable `latest.json` pointing to a verified index. Each upload is read back and checked
with SHA256 before the pointer is committed. `checkpoint.pt` includes step, full delta and
momentum; actual crop/gradient tensors and previews are included. Unchanged files are not
uploaded repeatedly. Earlier generations remain available for audit. A Drive failure stops
the job; it never silently falls back to scratch.

Banks are committed after each caption, diffusion has an in-flight marker, optimizer snapshots
commit at step1/every30 steps, bounded final PNGs commit before captioning, and API-start markers,
raw responses, captions/scores, budgets and failure tracebacks are independently committed.
Manual semantic checks remain pending/waived, never fabricated PASS.

To recover after a new sandbox: install the same requirements, restore the exact source/config,
then `python run_pipeline.py --config <locked-config> --drive-folder-id <folder-id> --restore`.
The runner restores only checksum-verified indexed files; incomplete/uncertain model/API stages
stop for inspection. It never regenerates already cached attacks after a caption failure.

The older native-cache instructions below describe R1 and are not the durable storage path for R2.

Python implementation extracted from `notebooks/attacks/v5.3-diffusion-regions.ipynb`.
The notebook stays available for review. Python execution does not use nbclient/Jupyter.
Entry points mirror MPCAttack; source provenance is in `notebook_provenance.json`.

```text
My-Attack/
  generate_adversarial_examples_MyAttack.py  # prepare + both attack arms
  blackbox_text_generation.py               # four victim captions, cached independently
  gpt_evaluate.py                           # GPT-score and ASR aggregation
  run_pipeline.py                          # attack → PNG check → captions → judge → bundle
  molab_bootstrap.py                        # dataset/config/model revision setup
  my_attack/
    config_schema.py  data.py  utils.py  persistence.py
    caption_bank.py  diffusion.py  losses.py  attack.py
    preparation.py  pipeline.py  evaluation.py  visualization.py
    surrogates/clip.py
```

## Install and run

```bash
python -m pip install -r requirements.txt
python molab_bootstrap.py --launch
# Default: pilot1/10 followed by200 confirmation pairs.
# The bootstrap prints the generated immutable config path.
```

Keep `OPENAI_API_KEY` in the environment or MoLab Secrets. Never put it in source/config.
Use `molab_bootstrap.py --help` to prepare the exact Kaggle version and pinned HF revisions.

Separate stages (postprocessing never invokes an attack):

```bash
python generate_adversarial_examples_MyAttack.py --config configs/molab.json
python blackbox_text_generation.py --config configs/molab.json
python gpt_evaluate.py --config configs/molab.json
```

`--restore` restores existing native-cache snapshots before continuing; a missing snapshot
raises instead of silently regenerating a model stage. Keep the generated `*_restore_index.py`
source file and unchanged module sources when resuming on MoLab. Ordinary scratch ZIPs alone
are not durable there. Native retention is subject to the platform's limits; retrieve result
bundles externally for long-term storage.

## Method

K10 Qwen source/target captions; nearest-mean target text in concatenated normalized three-CLIP
space; SD1.5 img2img from source with selected target caption, strength0.9, guidance7.5,40steps.
Generation failure uses original target with recorded provenance. Five fresh area[0.5,0.9]
adv crops each update; full effective target; image/text anchors0.6/0.4; InfoNCE10 vs1 source
negative. Mean crop loss differentiates into one full-image delta;300 updates,
epsilon16pixel,alpha1,momentum0.9,tau0.2,omega2.

The runner saves actual crops/coordinates/gradients and preview, optimizer checkpoints,
reloaded-PNG bounds, exact victim captions, GPT responses/usage and per-victim scores.
Heartbeat prints every30seconds. On >=80GiB GPUs frozen models stay resident after their first load; smaller GPUs release them between stages. Both arms share preparation
and each victim load. No evaluator feedback is used by the optimizer.

GPT-score ASR uses threshold0.5 and original-target caption references. This adapted GPT5mini
protocol is not the published official MPC ASR; `official_asr` remains null. Pilot IDs1/10 and
confirmation IDs200..399 have separate summaries. Manual semantic checks were waived by user;
their status remains PENDING, never automatically PASS.
