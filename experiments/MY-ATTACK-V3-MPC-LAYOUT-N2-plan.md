# My-Attack-V3 MPC layout — technical pilot N2

**PENDING EXPLICIT USER REVIEW — NOT APPROVED.** This plan accompanies the GitHub publication and run guide; committing/pushing code does not approve an experiment. No automatic launch is queued.

## Hypothesis, mechanism and literature

Technical question: can the refactored MPC-shaped implementation build the intended ten source-caption counterfactual anchors, optimize within the pixel bound, and produce complete saved-PNG/caption diagnostics for two fixed pairs? This pilot does not claim improved attack efficacy.

The mechanism is unchanged from the documented V3 proposal: source-caption object replacement from notebook v6.4.1; CLIP image/text0.6/0.4; original source InternVL/DINO components; concatenate ten negative anchors and maximize negative CE against the clean-target positive. Random source crops and raw-gradient MI-FGSM momentum0.9 remain. Source code is the Git commit containing this plan, pinned with `git rev-parse HEAD` before execution; model/config/source/image hashes must be saved before pretrained loading.

Primary references checked2026-11-03: MPCv3 https://arxiv.org/abs/2603.04846v3 and official code https://github.com/LiYuanBoJNU/MPCAttack ; M-Attack-V2v2 https://arxiv.org/abs/2602.17645v2 and official code https://github.com/VILA-Lab/M-Attack-V2 . MPC is the direct multi-paradigm transfer baseline; MCA/ATA/patch momentum from M-V2 are not added. The existing review covers this same negative-anchor mechanism; no novelty claim is verified. Filename/layout compatibility is not byte-identical replication of the published protocol.

## Baseline, controls and exact scope

- Only the ten-negative arm,2development pairs. No new matched MPC baseline attack in this technical pilot. Source/target captions from the same evaluator are controls; K1 equivalence is tested synthetically, not counted as victim evidence. Historical baseline cells are preserved and cannot establish incremental gain here.
- Dataset `zintom/blackbox-adversarial-attack-dataset/versions/1`. Source `images/bigscale/nips17/{0,1}.png`; target `images/target_images/1/{0,1}.jpg`; pairing by stem. Only IDs0/1, seeds2026/2027. No held-out confirmation or200-sample run.
- RGB224, bicubic resize/center crop, pixel range0..255. Source crop area0.5..0.9. Batch1,300updates/pair,alpha1pixel,epsilon16pixel,tau0.2,omega2,momentum0.9. Original MPC `16/255 * randn` initialization is retained; valid-pixel projection and PNG rounding are disclosed safety repairs. No sweeps/checkpoint selection.
- Ten distinct source captions followed by ten one-object counterfactual captions. Max80source-caption attempts and10replacement attempts/negative; max190bank calls/pair,380total. Inspect raw prompts/outputs, slots/counts/duplicates/false negatives and target leakage. No cached historical V3 bank with absent/mismatched provenance is silently reused.
- Models: CLIP B16/B32/Laion-G, InternVL3-1B-hf, DINOv2-base, Qwen3-VL-8B-Instruct bank. Exact immutable revisions are `src/My-Attack-V3/model_revisions.json`. L336 is an optional compatibility choice and is **not** selected here. Record the environment freeze and returned hardware/model identities before inference; unavailable locks stop the run instead of model substitution.

## First-sample and evaluation protocol

Inspect sample0 before sample1. Gates: `0_data_bank` before optimization, `0_update1` before step2, and `0_complete` after frozen step300/saved-PNG verification/captions. Inspect actual source/target/transformed images and bank; exact negative-object changes; implemented loss sign/weights/reductions/shapes and logged values/finite nonzero gradients; actual crop and coordinates; final source/target/adversarial comparison and integer PNG L-infinity bound; prompt/revision/decoding and caption quality.

Report all seven mandatory items as PASS/FAIL/PENDING/justified N/A with artifact links under `docs/research/hypo/`. Missing/failed required checks stop continuation; a lower surrogate CE or target word mention is not attack success. No victim/judge output enters gradients or checkpoint selection. Preserve checkpoints at update1/every30/final300, together with delta/momentum/RNG, crop/loss records, bank provenance and image hashes; clean source/target PNGs represent the initial data.

Pipeline: attack → verify saved PNG → `blackbox_text_generation.py` local captions → configured manual diagnostic/checklist → durable bundle → sample1 only after review. Victim `Qwen/Qwen2.5-VL-3B-Instruct` at the pinned revision, generic prompt `Describe this image in one concise sentence, no longer than 20 words.`,greedy,beams1,max_new_tokens64,batch1. At most6image-caption slots: source/target/adversarial for each of2pairs; cached completed slots are not regenerated. Report family overlap with Qwen bank generator. Manually assess target object/action/relation/count alignment, source persistence, hallucinations and truncation independently from official ASR.

No GPT/API judge (`--scorer_model` absent),API budget0USD. Official ASR remains PENDING. No claim of improved transfer or confirmation can follow from this technical N2 pilot; a matched efficacy experiment needs a new reviewed plan.

## Budget and stop conditions

One existing-quota RTX Pro6000 Blackwell; no purchased credits. Max90minutes GPU allocation including setup/backup,120minutes wall,120GiB disk. The attack CLI has `--max_seconds4500` (75minutes including bank/model loading/reviews/captions); reserve at most15allocation minutes outside it for setup/export. Record allocation interval separately from active GPU compute and actual API calls. Max600optimization updates,380bank calls,4MPC clean-description calls,6independent evaluator captions. One attempt per pair; unresolved/failed requests count toward limits and are not automatically retried.

Stop on missing/false pair, unsupported required concept, false negatives/leakage, duplicate-bank exhaustion, nonfinite/disconnected gradients, bounds failure, invalid/missing diagnostic, failed backup, account restriction/reclamation or budget exhaustion. Preserve partials/errors, never reset seeds or change prompts/models to rescue a result. Post-processing failure retains saved attack outputs and cannot trigger a fresh attack.

## Storage, manifests and reporting

Local output `/marimo/outputs/v3_mpc_n2`; this is scratch, not a durable guarantee. Before inference, prove a verified external backup route with an uploaded/read-back sentinel. At each required gate, export the entire run snapshot to Drive folder `1daramuahKwdAVGQtsDZTzy1v0OSG-kiI` under a new run-specific child, verify archive SHA256/size, and retain all stages. Do not mark backup complete merely because a local ZIP exists. If the verified route is not available, stop before inference. Core scripts do not claim automatic Drive or Sheet synchronization.

Retain the launch command, Git SHA, source/config/model locks, plan approval hash, raw/transformed image hashes, seed policy, environment freeze, allocation/compute records and all errors in the result bundle. Canonical pixel hashes for sample0 source/target: `6f0eab3a2e62e09ed633107165b55553371dd10a91f00996cd7ef61702ffa7df` / `74468ac7fb05df8b6b5408d0445c4915e215a5ec7fdb77576203bc9cd97771fb`; sample1: `568612e715015d6f1efb1500cba03a6f8a56a3f0dbe5362cac5a93d3e4d7e054` / `a86cb7a3bd8666aed1ca96ec20c080ad613cf3ca01f0b058bdb322dae1addd65`. Reconcile preprocessing hashes before attack.

After any partial/failure/completion, update the existing hypothesis-management Sheet with a distinct new-run row, read back the write, preserve baseline and old V3 rows, leave missing metrics PENDING, and save a local pending update on sync failure. Add Run Results/insights and index entries with evidence level limited to actual observations. No execution is approved until this plan is explicitly reviewed by the user.
