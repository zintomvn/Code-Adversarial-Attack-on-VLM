# My-Attack-V3

MPC's original multi-paradigm attack with ten source-caption counterfactual negative anchors. The pretrained models remain frozen; “training” means optimizing the image perturbation.

The approved pilot contains two source-target pairs and two matched arms. It does not launch a full batch. See `experiments/MY-ATTACK-V3-20261102-N2-plan.md` for the reviewed design and limits.

## Source map

- `surrogates/`: selectively copied committed MPC classes, unchanged forwards/preprocessing, explicit model revisions; see `UPSTREAM.json` and `MPC_LICENSE.txt`.
- `negatives.py`: notebook v6.4.1 prompt/object-replacement logic, ten unique captions, bounded attempts and provenance.
- `losses.py`: original negative CE with K negatives; K1 reproduces MPC's objective.
- `attack.py`: MPC random crop/MI-FGSM, shared deterministic crops/init, checkpoint/export audit.
- `models.py`: original MPC fusion, frozen anchors, separate bank/evaluation captioners.
- `storage.py`: verified Drive snapshots, receipts and fail-closed persistence.
- `run_pipeline.py`: independent stage caches and evidence-bound review gates.
- `setup_molab.py`: environment installation and cumulative watchdog.

```powershell
python -X utf8 -m unittest discover -s src/My-Attack-V3 -p "test_*.py" -v
```

MoLab, after uploading/extracting the regular source ZIP:

```bash
python My-Attack-V3/setup_molab.py
```

The process pauses for inspection at `gates/*_evidence.json`. A reviewer records an explicit PASS with the evidence SHA256, inspector and substantive notes under `reviews/<gate>.json`. Failed/stale reviews stop execution. The sample0 final checklist must pass before sample1 begins. Do not create blanket review files or use victim output to tune the attack/bank/checkpoint.

Checkpoints include image, momentum, perturbation, RNG and metrics at 0/1/every30/final300. Safety repairs relative to upstream: honor the configured step count, project valid pixels in both arms, round integer PNGs once, save before captions, separate RNG streams and preserve each stage. They are shared implementation repairs, not effects attributed to negative samples. The initial noise scale remains upstream `16/255` in the pixel-space tensor.

Official ASR stays PENDING: this pilot has independent Qwen2.5-VL captions plus manual frozen diagnostics, no paid/API judge. A lower surrogate loss is not targeted success.
