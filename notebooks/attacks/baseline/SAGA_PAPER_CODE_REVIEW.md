# SAGA: paper, official release, and this baseline

Reviewed on 22 October 2026 (Asia/Bangkok). No attack, dataset, notebook configuration, or completed results were modified by this audit.

## Conclusion

The baseline uses the official SAGA implementation at commit `042274599b958d4d63760b9bf44bbcb88493811c`, which is also the repository's current remote HEAD at audit time. This establishes release provenance, not exact equivalence to every equation or experimental setting in the paper.

All 23 locked SAGA files inside `baseline_code.zip` match their Git object bytes and SHA-256 locks. In `src/SAGA`, the only normalized content difference among those files is the comment `# crops -> optimize` in `saga/optimization.py`; Python ASTs match. See `saga_logic_audit.json` and rerun `python notebooks/attacks/baseline/audit_saga_logic.py`.

Sources: [official release](https://github.com/jaehyun-kwak/SAGA/tree/042274599b958d4d63760b9bf44bbcb88493811c), [paper v2](https://arxiv.org/html/2602.04356v2).

## Core correspondence

Paper Section 4 and Algorithm 1 describe a fixed attention map, an ordered regional schedule, local crops, an ensemble cosine objective, and a global perturbation bound. Section 5.1 specifies three CLIP surrogates, 300 updates, epsilon 16/255, step 1/255, 10 stages, k=3, and IoU threshold 0.3. These high-level choices are present in the release and baseline.

Code evidence:

- `saga/pipeline.py`: attention is extracted before each sample's optimization, once; the schedule is precomputed.
- `saga/attention.py`: Qwen3-VL-8B-Instruct, eager attention, bfloat16 extractor, layer 29.
- `saga/surrogates/__init__.py` and `base.py`: B/16, B/32, Laion G/14; normalized image/text features and their average dot product.
- `saga/crops.py`: crop from the selected rectangle, then random resized crop; gradients flow through the crop to the original perturbation tensor.
- `saga/optimization.py`: zero initial delta, pixel-space alpha=1 and epsilon=16, 300 iterations, sign-gradient ascent and delta clipping.
- The apparent `hotspot_prob=0.5` in the release shell script is overridden by `hotspot_prob=1.0` in the optimizer. The baseline retains this actual released behavior.

## Release details that qualify paper equivalence

Paper Section 4 describes k hotspots per stage, an IoU constraint, and E/(kN) updates per hotspot. Algorithm 1 writes a gradient update without a sign operator. Appendix E.1 describes per-token spatial normalization and averaging. The following observations come from reading and executing the released code:

1. **IoU is not a hard invariant.** `hotspots.py` first filters candidates by IoU, then fills missing slots with overlapping rectangles. Synthetic tests produce pairwise IoU above 0.3; for one random 24x24 map a stage reaches approximately 0.918. This fallback is in upstream code, not introduced by the notebook.

2. **There need not be 30 schedule entries.** At area ratio 1, exact-duplicate rejection leaves one full-map rectangle. Tested 14x14 and 24x24 maps produce 28 entries. The optimizer divides by the actual schedule length: for 300 iterations, 27 entries receive 10 updates and the final entry receives 30. Tested 7x7 maps produce 22 entries and omit groups 3 and 8 because integer rectangle sizes do not meet the area tolerance; their final entry receives 27 updates. These are demonstrations on synthetic maps, not measurements of all 200 runtime samples.

3. **Region selection is a rectangle search.** The implementation searches rectangles with roughly the requested area (+/-5%), limits aspect ratio to 3, and ranks total attention sums. It does not implement an arbitrary binary mask of the highest-valued individual patches. The schedule's group order increases the area ratio. No post-update attention is used to recompute it.

4. **Update direction is `grad.sign()`.** This is the released operation. The notebook must preserve it when the requirement is to run the authors' code; replacing it with raw gradients would create a different implementation.

5. **Attention has extra postprocessing.** After averaging normalized token maps, `attention_map.py` applies min-max normalization; `attention.py` normalizes again. This code path is unchanged in the frozen bundle. It should be documented when interpreting the attention map numerically as a probability distribution.

These differences establish that a literal reading of the paper is insufficient to reconstruct every released operation. They do not by themselves establish the effect on ASR or invalidate the authors' experiments.

## This experiment's protocol differs from paper reproduction

Paper Section 5.1 uses 1,000 pairs and gpt-oss-20b to compare attacked-image captions with target texts; its main success threshold is strictly greater than 0.5.

The existing run deliberately uses 200 pairs, GPT-4o as judge, success >=0.3, and the requested v3.2 evaluation notebook. In `eval_worker.py`, judge inputs are the victim model's caption of the target image and its caption of the adversarial image. They are **not** the SAGA `input/coco.jsonl` target text used by the attack. Dataset stem checks establish pairing IDs, not semantic equality between those two reference texts. The victim list, prompts, generation settings and local NF4 evaluation also follow this project's protocol.

Consequently, these ASR values are results for the project's shared evaluation setup, not direct reproductions of the paper's tables. Merely changing the threshold on old GPT-4o scores would not reproduce the paper's judge/reference setup.

The attack environment also uses CUDA 12.8 for Blackwell, whereas the upstream uv source selects CUDA 12.6. Core PyTorch/transformers versions match, but GPU/kernel changes mean bitwise equality to the authors' hardware is not established.

## Wrapper and limits

`author_worker.run_saga` passes the configuration into the official CLI/parser and pipeline. It wraps attention entry for session/RNG handling and wraps completed metadata writes to save images and checkpoints. It pins Hugging Face revisions. The target image resized/center-cropped by the wrapper is an evaluation artifact; the optimization target remains the original released text.

The audit rehashed the local frozen bundle and checked GitHub remote HEAD. It did not rehash an expired remote runtime, rerun 200 GPU attacks, prove numerical equivalence of every checkpoint hook, or establish that the custom 200-image cohort reproduces the authors' sample selection. The schedule checks execute the original frozen `hotspots.py`, not a replacement algorithm.

For reporting, use: “Official SAGA implementation at commit 0422745, evaluated on our 200-pair cohort with the v3.2 evaluation protocol.” Keep any future literal-paper variant in a separate run with a separate identifier; do not overwrite these completed results.
