# CARA v6.4.1, two-pair four-victim run

Approved scope: `experiments/CARA641-20261027-N2-4V-plan.md`.
Run ID: `CARA641-20261027-N2-4V-v1`. No command runs on import.

`attack.py`, `losses.py`, `regions.py`, `attention.py`, `models.py`, `finer.py`
and `data.py` expose the inherited algorithm as ordinary source. The original
notebook is frozen under `reference/`; `SOURCE_MAP.json` records the migration.
`config.py` exposes explicit attack settings and pinned revisions.
`evaluation.py` contains independent captions and the exact v3.2 judge rubric.
`state.py`, `verification.py` and `pipeline.py` handle stages, budgets and gates.

Upload the regular source ZIP through the MoLab Files UI. Extract it to
`/marimo/cara641_package`, configure the approved GPU and invoke:

```sh
python /marimo/cara641_package/src/cara_v641/setup_molab.py
```

The supervisor pauses at `gate_data_00000.json`, `gate_step1_00000.json`, then
`gate_complete_00000.json`; repeats for00001. Each gate must contain
`identity` equal to the fingerprint of `identity.json`, `status: PASS`,
and evidence links. Technical/evaluation failures leave gates closed. Gate
files are written only after actual evidence inspection, never by elapsed time.

At `paid_evaluation_pending`, download the named `evaluation_ready_00000.zip`
and extract to a local result directory, then run:

```sh
python -m src.cara_v641 paid --root experiments/cara641_20261027_n2_4v/results --sample 0 --env .env
```

Upload `paid_00000.zip` back through Files. The supervisor verifies its identity,
stores its results and waits for final audit. Repeat for sample1. Credentials
stay local. A missing/refused/truncated response remains an error. Successful
captions/scores are cached, and no postprocessing path restarts an attack.

The notebook is a short launcher; all algorithm/evaluation implementation is
in these `.py` files. For synthetic verification, from the repository root:

```sh
python -X utf8 -m unittest discover -s experiments/cara641_20261027_n2_4v -p "test_*.py" -v
```

After every run, including partial/error, download and verify `results.zip`,
update the hypothesis checklist/results, append to the user Sheet and verify
readback. Never report a pending evaluator as completed or compare surrogate
loss directly with victim ASR.

The first approved execution stopped at the mandatory concept/provenance gate before attack. See `docs/research/hypo/CARA641-N2-4V.md`. Infrastructure setup initially lacked pip for environment capture; requirements now explicitly includes pip25.3. The exact source used by that stopped run remains frozen inside its downloaded `results/source.zip`; the current source archive includes this metadata dependency fix.
