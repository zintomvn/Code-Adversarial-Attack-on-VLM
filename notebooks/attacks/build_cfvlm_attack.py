"""Build a standalone corrected notebook without mutating the frozen old run."""
import ast
import hashlib
import json
from pathlib import Path


BATCH_GENERATION = '''@torch.inference_mode()
def qwen_generate_negatives_batch(backend, requests, audit=False):
    images, texts = [], []
    for request in requests:
        pixels, caption, attempt, accepted = request[:4]
        feedback = request[4] if len(request) > 4 else ()
        images.append(pixels_to_pil(pixels).resize((ATTENTION_INPUT_RES, ATTENTION_INPUT_RES), Image.Resampling.BICUBIC))
        prompt = (cf_negative_audit_prompt(caption, accepted) if audit else
                  cf_negative_prompt(caption, attempt, accepted, CF_NEGATIVE_CANDIDATES - len(accepted), feedback))
        messages = [{'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': prompt}]}]
        texts.append(backend.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
    tokenizer = backend.processor.tokenizer
    previous_padding = tokenizer.padding_side
    tokenizer.padding_side = 'left'
    inputs = None
    backend.model.model.rope_deltas = None
    try:
        inputs = backend.processor(text=texts, images=images, do_resize=False, padding=True, return_tensors='pt').to(backend.device)
        ids = backend.model.generate(**inputs, do_sample=False, max_new_tokens=CONCEPT_MAX_NEW_TOKENS)
        eos = backend.model.generation_config.eos_token_id
        eos = [eos] if isinstance(eos, int) else list(eos or [])
        result = []
        for output in ids:
            suffix = output[inputs['input_ids'].shape[1]:]
            raw = tokenizer.decode(suffix, skip_special_tokens=True)
            # A finished row may be padded while another row reaches the cap.
            truncated = len(suffix) >= CONCEPT_MAX_NEW_TOKENS and not any(int(t) in eos for t in suffix)
            result.append((raw, truncated))
        return result
    except torch.cuda.OutOfMemoryError:
        if len(requests) == 1:
            raise
        inputs = None
        cuda_cleanup()
        middle = len(requests) // 2
        event('text', 'negative_batch_oom_split', requested=len(requests))
        return (qwen_generate_negatives_batch(backend, requests[:middle], audit=audit)
                + qwen_generate_negatives_batch(backend, requests[middle:], audit=audit))
    finally:
        tokenizer.padding_side = previous_padding
        backend.model.model.rope_deltas = None

def qwen_generate_negatives(backend, target_pixels, target_caption, attempt, accepted):
    return qwen_generate_negatives_batch(backend, [(target_pixels, target_caption, attempt, accepted)])[0]

'''

BANK_PREPARATION = '''    if need_negatives:
        policy = {'version': NEGATIVE_POLICY_VERSION, 'prompt_sha256': CF_PROMPT_SHA256,
                  'candidates': CF_NEGATIVE_CANDIDATES, 'batch_size': NEGATIVE_GENERATION_BATCH_SIZE}
        for pair in pairs:
            record = records[pair['sample_id']]
            previous = record.get('negative_policy')
            if previous != policy:
                if record.get('negatives'):
                    event('text', 'stale_negative_bank_rejected', sample_id=pair['sample_id'])
                record.update(negatives=[], negative_attempts=[], negative_rejections=[], negative_policy=policy)
                record.pop('negative_error', None)
            if not record.get('caption_error'):
                record['negatives'] = cf_only_audited(validate_negative_candidates(record.get('negatives', []), record['target']))
        write_json(cache_path, records)
        for attempt in range(TEXT_RETRIES):
            pending = [p for p in pairs if not records[p['sample_id']].get('caption_error')
                       and len(records[p['sample_id']]['negatives']) < CF_NEGATIVE_CANDIDATES]
            if not pending:
                break
            for offset in range(0, len(pending), NEGATIVE_GENERATION_BATCH_SIZE):
                batch = pending[offset:offset + NEGATIVE_GENERATION_BATCH_SIZE]
                requests = []
                generated_pairs = []
                for pair in batch:
                    sid = pair['sample_id']; record = records[sid]
                    override = MANUAL_TEXT_OVERRIDES.get(sid, {})
                    if 'negatives' in override:
                        if override.get('negative_policy') != policy:
                            raise ValueError('Manual negative bank must declare the exact current policy')
                        record['negatives'] = cf_only_audited(validate_negative_candidates(override['negatives'], record['target']))
                        record['negative_attempts'].append({'provenance': 'manual_override'})
                    else:
                        requests.append((load_pixels(pair['target']), record['target'], attempt, record['negatives'], record.get('negative_rejections', [])))
                        generated_pairs.append(pair)
                if requests:
                    backend = ensure_attention_backend(MODEL_REVISIONS[ATTENTION_MODEL_ID])
                    with preserve_rng():
                        outputs = qwen_generate_negatives_batch(backend, requests)
                    audit_requests, audit_pairs, proposed = [], [], []
                    for pair, (raw, truncated) in zip(generated_pairs, outputs):
                        record = records[pair['sample_id']]
                        try:
                            if truncated:
                                raise ValueError('Qwen output reached token cap without EOS')
                            candidates = cf_validate_candidates(parse_json_response(raw), record['target'],
                                                                max_candidates=CF_NEGATIVE_CANDIDATES-len(record['negatives']))
                            record['negative_attempts'].append({'attempt': attempt + 1, 'raw': raw, 'proposed': len(candidates)})
                            if candidates:
                                audit_requests.append((load_pixels(pair['target']), record['target'], attempt, candidates))
                                audit_pairs.append(pair)
                                proposed.append(candidates)
                        except Exception as exc:
                            record['negative_attempts'].append({'attempt': attempt + 1, 'raw': raw, 'error': f'{type(exc).__name__}: {exc}'})
                    if audit_requests:
                        with preserve_rng():
                            audits = qwen_generate_negatives_batch(backend, audit_requests, audit=True)
                        for pair, candidates, (raw_audit, truncated) in zip(audit_pairs, proposed, audits):
                            record = records[pair['sample_id']]
                            info = record['negative_attempts'][-1]
                            info['audit_raw'] = raw_audit
                            try:
                                if truncated:
                                    raise ValueError('Qwen audit reached token cap without EOS')
                                audit_payload = parse_json_response(raw_audit)
                                approved = cf_validate_audits(audit_payload, candidates)
                                approved_captions = {r['caption'] for r in approved}
                                reasons = {r.get('id'): r.get('reason', 'Rejected by semantic audit')
                                           for r in audit_payload.get('audits', []) if isinstance(r, dict)}
                                record.setdefault('negative_rejections', []).extend(
                                    {'caption': r['caption'], 'reason': reasons.get(i, 'Missing/invalid audit')}
                                    for i, r in enumerate(candidates) if r['caption'] not in approved_captions)
                                record['negatives'] = cf_only_audited(validate_negative_candidates(approved, record['target'], record['negatives']))
                                info.update(audit_approved=len(approved), accepted=len(record['negatives']))
                            except Exception as exc:
                                info['audit_error'] = f'{type(exc).__name__}: {exc}'
                # Checkpoint every GPU batch, including incomplete banks and raw
                # responses. No fabricated fallback or silent cohort filtering.
                for pair in batch:
                    record = records[pair['sample_id']]
                    if len(record['negatives']) < CF_NEGATIVE_CANDIDATES:
                        record['negative_error'] = f"Need {CF_NEGATIVE_CANDIDATES} CF-VLM candidates; got {len(record['negatives'])}"
                    else:
                        record.pop('negative_error', None)
                    event('text', 'negative_bank_cached', sample_id=pair['sample_id'],
                          candidates=len(record['negatives']), error=record.get('negative_error'))
                write_json(cache_path, records)
    return records
'''


def build():
    base = Path(__file__).resolve().parent
    original = base / 'kltn-attack-v4.3.2-h1.ipynb'
    nb = json.loads(original.read_text(encoding='utf-8'))
    src = lambda index: ''.join(nb['cells'][index]['source'])
    def put(index, source):
        nb['cells'][index]['source'] = source.splitlines(keepends=True)
    put(0, '''# KLTN attack v4.3.3-h1 — CF-VLM negative caption policy

Corrects v4.3.2's noun-only counterfactual prompt using CF-VLM §2.2 and
Appendix A, page 17. Allows one object attribute or one causal intervention
(including its dependent outcome). See [paper](https://arxiv.org/abs/2506.17267)
and [official code](https://github.com/HCP-AI-Research-Lab/CF-VLM).

This attack adapts the text edit policy; it does not reproduce CF-VLM's SDXL
counterfactual images or fine-tuning losses. JSON traces and structural checks
are additions for auditing and cannot prove visual falsity/semantic atomicity.
New policy, bank and run identity: do not resume a v4.3.2 result directory.
Generate four candidates (official code default K=4); mine/select K_NEG=2 as
before. A second image-conditioned Qwen pass checks semantic minimality,
unchanged context, coherent causal effects and visible contradiction. Failed
or uncertain candidates are rejected; retries receive rejection feedback.
The same Qwen model performs this automated audit: it is not a human quality
certificate. All corrected negative variants share the audited cached bank.
Batched Qwen generation/auditing defaults to 16 with OOM splitting. FOA retains
its ordered state. The audited v2 campaign excludes the earlier test bank.
''')
    config = src(1).replace("kltn_attack_v4_3_2_h1_outputs", "kltn_attack_v4_3_3_h1_cfvlm_outputs")
    config = config.replace('MAX_NEGATIVES = 12', 'MAX_NEGATIVES = 12\nCF_NEGATIVE_CANDIDATES = 4\nNEGATIVE_GENERATION_BATCH_SIZE = 16')
    config = config.replace('CONCEPT_MAX_NEW_TOKENS = 1024', 'CONCEPT_MAX_NEW_TOKENS = 2048')
    config = config.replace('TEXT_RETRIES = 3', 'TEXT_RETRIES = 5')
    config = config.replace("'TEXT_RETRIES', 'LOG_EVERY'", "'TEXT_RETRIES', 'CF_NEGATIVE_CANDIDATES', 'NEGATIVE_GENERATION_BATCH_SIZE', 'LOG_EVERY'")
    config = config.replace("    if NEGATIVE_LOSS not in", "    if not K_NEG <= CF_NEGATIVE_CANDIDATES <= MAX_NEGATIVES:\n        raise ValueError('Expected k_neg <= CF candidates <= MAX_NEGATIVES')\n    if NEGATIVE_LOSS not in")
    put(1, config)
    old = src(11)
    heading = old[:old.index('def validate_negative_candidates')]
    policy = (base / 'cfvlm_negatives.py').read_text(encoding='utf-8')
    wrapper = '''\ndef validate_negative_candidates(payload, target_caption, existing=()):
    return cf_validate_candidates(payload, target_caption, existing, CF_NEGATIVE_CANDIDATES)

'''
    parser = old[old.index('def parse_json_response'):old.index('@torch.inference_mode()')]
    preparation = old[old.index('def prepare_text_records'):old.index('    if need_negatives:')]
    # Factual caption overrides are already locked. Avoid 200 network-triggering
    # writes during startup while retaining bounded caption checkpoints.
    preparation = preparation.replace('            write_json(cache_path, records)\n    finally:',
        '            if len(records) % 32 == 0:\n                write_json(cache_path, records)\n'
        '        write_json(cache_path, records)\n    finally:')
    put(11, heading + policy + wrapper + parser + BATCH_GENERATION + preparation + BANK_PREPARATION)
    checks = src(16)
    start = checks.index('    bank = ', checks.index('def check_negatives_and_regions'))
    end = checks.index("    assert len(validate_negative_candidates(bank, 'A cat sits.'))", start)
    checks = checks[:start] + '''    bank = [
        {'edit_type': 'attribute', 'attribute': 'object_category', 'edits': [{'span': 'cat', 'replacement': 'dog'}], 'caption': 'A dog sits.'},
        {'edit_type': 'attribute', 'attribute': 'object_category', 'edits': [{'span': 'cat', 'replacement': 'dog'}], 'caption': 'A dog sits.'},
        {'edit_type': 'attribute', 'attribute': 'object_category', 'edits': [{'span': 'cat', 'replacement': 'bird'}], 'caption': 'A bird flies.'},
    ]
''' + checks[end:]
    put(16, checks)
    runner = src(17).replace("'LOG_EVERY', 'SAVE_EVERY',", "'LOG_EVERY', 'SAVE_EVERY', 'NEGATIVE_POLICY_VERSION', 'CF_PROMPT_SHA256',\n    'CF_NEGATIVE_CANDIDATES', 'NEGATIVE_GENERATION_BATCH_SIZE',")
    runner = runner.replace("f'v432_", "f'v433_cfvlm_").replace("'version': '4.3.2-h1'", "'version': '4.3.3-h1-cfvlm'")
    # All required candidates must exist, rather than accepting a degenerate
    # 2-entry bank where hard/random top-2 selection is the same.
    runner = runner.replace("len(text.get('negatives', [])) < K_NEG", "len(text.get('negatives', [])) < CF_NEGATIVE_CANDIDATES")
    put(17, runner)
    runtime = src(2)
    labels_line = next(line for line in runtime.splitlines() if line.startswith('BUILD_CELL_FIRST_LINES = '))
    labels = ast.literal_eval(labels_line.split(' = ', 1)[1])
    hashes = {labels[src(i).strip().splitlines()[0]]: hashlib.sha256(src(i).strip().encode()).hexdigest()
              for i, cell in enumerate(nb['cells']) if cell['cell_type'] == 'code' and src(i).strip().splitlines()[0] in labels}
    runtime = '\n'.join('BUILD_CELL_HASHES = ' + repr(hashes) if line.startswith('BUILD_CELL_HASHES = ') else line
                        for line in runtime.splitlines()) + '\n'
    put(2, runtime)
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code':
            cell.update(execution_count=None, outputs=[])
            compile(src(i), f'cfvlm:cell{i}', 'exec')
    target = base / 'kltn-attack-v4.3.3-h1-cfvlm.ipynb'
    target.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + '\n', encoding='utf-8', newline='\n')
    print(target)
    return target


if __name__ == '__main__':
    build()
