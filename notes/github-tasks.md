# GitHub commit tasks — 2026-10-23

This checklist groups the existing local changes into small, reviewable commits.

- [x] 1. Ignore local credentials and tool caches.
- [x] 2. Register pinned baseline source repositories and preserve local O-Attack annotations.
- [x] 3. Replace the previous CF-VLM campaign scripts, documentation and tests with the current H1/Molab notebooks and report; update LF rules.
- [x] 4. Add the official baseline execution, evaluation, recovery and source audit bundle.
- [x] 5. Add the O-Attack/M-Attack-V2 campaign bundle and verification tools.
- [x] 6. Add concept gradient heatmap and adversarial analysis notebooks.
- [x] 7. Update hypothesis tracking and research task notes, and record verification.

Validation on the local snapshot:

- `py -3.13 notebooks/attacks/baseline/verify_baselines.py --data outputs/molab_hypotheses_n200/attack_data_n200.zip`: passed, including 65 author files, 200 paired samples, checkpoint integrity and exact CPU FGSM resume.
- `py -3.13 notebooks/attacks/baseline/om_attack/verify_release.py`: passed, including 35 source files, interrupted extraction recovery, RNG restore and corrupt checkpoint rejection.
- Syntax/JSON checks: 28 Python files, 11 notebooks, 107 code cells without notebook magics, and 18 JSON files passed.
- Credential scan of 70 new files, including ZIP contents: no matching local secret values or recognized key patterns found.

GPU attacks and external model API evaluations were not rerun for this commit/push task. Existing deployment reports describe earlier runs.
