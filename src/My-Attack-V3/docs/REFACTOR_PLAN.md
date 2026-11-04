# MPC layout refactor — code only

User instruction2026-11-03: match MPCAttack's layout and expose the actual editable implementation in `generate_adversarial_examples_MPCAttack.py`, `blackbox_text_generation.py`, `gpt_evaluate.py`, and `config_schema.py`.

- Preserve the previous implementation and tests under `molab/legacy_v1/`; preserve archived experiment bundles. No pretrained inference, image attack, external judge or Molab launch for this refactor.
- Reuse MPC's `get_args → main → run_attack`, ImageFolderWithPaths, build_clip_models, extract_feature, explicit crop/MI-FGSM loop and `img/` / `description/` layouts. Mark the ten-negative additions at bank preparation, feature fusion and loss only.
- Keep notebook object-replacement logic in `negatives.py`, default10negative anchors and source image/negative text fusion0.6/0.4; original source InternVL/DINO components remain unchanged.
- Add MPC-shaped dataclasses with `default_factory`, editable scalar defaults and argparse flags. Put shared loss/I/O/approval-review helpers in `utils.py`, as MPC does.
- Caption and GPT scripts retain upstream names/arguments/text-file formats, lazy dependencies and per-image durable caches. No automatic API calls/retries, model substitution or fabricated missing-result zeroes.
- Verify offline CLI help, configuration isolation, K1 loss equivalence, K10 gradient/fusion, synthetic attack loop/checkpoints, pairing, caption caching and judged-score denominators. Run preserved legacy tests separately.
- Document file mapping and code-only refactor limits. Historical run/recovery source hashes refer to the immutable old bundles; do not claim refactored files are byte-identical or already experiment-approved.

## Completed implementation and checks

All four MPC-named files contain their editable implementation. Defaults come from the MPC-shaped dataclasses; negative preparation/fusion and K-negative CE are marked in the main attack file/`negatives.py`/`utils.py`. Upstream VLMEvalKit is copied locally without caches, and optional L336 remains available with an explicit revision and text-feature API.

The caption/GPT scripts remain standalone and are called in sequence by the attack entry point after saved-PNG verification. Default local captions are followed by manual diagnostic review; GPT is opt-in under an approved API scope. Post-processing resume reuses verified PNGs without loading attack models or re-optimizing them.

Review fixes: reject plaintext cache reuse without metadata; fingerprint transformed source pixels/revision/prompts/decoding for negative banks; stable sample-ID seeds across shards; retain `--ensemble True/False`; apply a whole-CLI Linux deadline; record Qwen output tokens/decoding; verify caption provenance before real judging. Defaults/safety repairs differ from some MPC defaults and are disclosed in README, not claimed as negative-only efficacy evidence.

Verification:36current-core synthetic tests passed;37preserved legacy synthetic tests passed; all23preserved-file hashes matched; root Python compilation and all three offline `--help` commands passed. No real-data smoke test, pretrained download/inference, GPU attack, judge request or Molab launch occurred.
