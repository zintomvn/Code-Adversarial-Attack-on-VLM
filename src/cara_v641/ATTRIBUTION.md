# Source provenance

This is a source-layout migration of the local CARA v6.4.1 notebook, preserved
with its original SHA256 in `reference/`. `reference/SOURCE_MAP.json` maps the
original cells to readable Python modules. Algorithm definitions remain directly
editable; no runtime cell execution, source strings, encoded source archives or
model code generation is used.

- MPC InfoNCE helper: https://github.com/LiYuanBoJNU/MPCAttack,
  local commit `7b195a5bcbaeaf1e8534d101199810a2782caad3`, MIT; license copied
  as `licenses/MPC_LICENSE.txt`. CARA contrastive K-negative adaptation and
  dual-proposal overwrite optimizer are local research extensions.
- FINER https://github.com/ExplainableML/finer original object/negative-object prompts and parsing helpers: copied
  from the existing notebook cells10/12; CaptionBank is its local extension.
- SAGA https://github.com/jaehyun-kwak/SAGA original attention helpers: existing notebook cell22; concept-token
  matching wrapper is local CARA code in cell24.
- Grad-ECLIP original dense-encoder helpers: existing notebook cells18/20;
  optional path retained for attribution, disabled in this approved run.
- OpenAI CLIP optional dependency is pinned to commit
  `d05afc436d78f1c48dc0dbf8e5980a9d471f35f6` (MIT).
- Caption protocol and GPT-4o rubric: local `eval_protocol_v32.py` extracted
  from `text_gen_eval_v3.2.ipynb`. Rubric identity is checked by a synthetic test.

The inherited notebook does not contain complete upstream commit/license
metadata for FINER, SAGA or Grad-ECLIP. This gap is disclosed; their snippets
are preserved verbatim for private research replication, without assigning
them a new license or claiming independent authorship. Baseline submodules
and their licenses remain unchanged. Model license terms remain their own.
