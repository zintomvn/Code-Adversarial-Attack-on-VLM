"""CF-VLM counterfactual text edit policy, adapted for attack negative captions.

Sources: NeurIPS 2025 CF-VLM, Section 2.2 and Appendix A (page 17),
https://github.com/HCP-AI-Research-Lab/CF-VLM/blob/main/process.py.
This is the text edit policy, not a reproduction of CF-VLM image generation or
fine-tuning. JSON edit traces are our addition for auditing/cache validation.
Structural validation cannot certify semantic minimality or visual falsity.
"""
import hashlib
import json

NEGATIVE_POLICY_VERSION = 'cfvlm_text_edits_v2_audited'
CF_ATTRIBUTE_TYPES = (
    'color', 'material', 'shape', 'size', 'position_orientation', 'state_pose',
    'object_category', 'parts_structure', 'movement_status', 'quantity',
    'expression_emotion',
)
CF_PROMPT_POLICY = """Generate counterfactual rewrites of the factual target caption.
Each candidate must make exactly ONE semantic intervention:
(A) change ONE attribute of ONE object: color, material, shape, size,
position/orientation, state/pose, object category, parts/structure, movement
status, quantity, or facial expression/emotion; OR
(B) change ONE cause-effect link already described by the caption. Change the
cause and, when needed, its dependent outcome together so the alternate event
is coherent. These two linked text edits count as one causal intervention.
Do not invent a causal chain absent from the caption. If it has no genuine
cause-effect link, use attribute edits only; do not label a category swap causal.
Keep every unrelated object, attribute, action, relation and background intact.
Never replace a whole scene or change weather and location together. For
example, snowy forest -> sunny meadow is NOT a color edit and is invalid.
Position/orientation changes the spatial placement of the SAME object.
State/pose changes one physical state, not the time/history of the whole scene.
Use fluent, grammatical, visually descriptive sentences, meaningful changes,
and no duplicate sentences or synonyms that preserve the original meaning.
The supplied image is the factual anchor: the changed fact must contradict
what is visible in it, rather than merely add an unobservable detail.
Inspect each candidate before returning it: discard candidates that change two
independent factors, are true of the image, or have an inconsistent outcome.

Return JSON only: {"negatives": [candidate, ...]}. Each candidate contains:
"edit_type": "attribute" or "causal";
"attribute": one allowed snake_case attribute name above for attribute edits,
or null for causal edits;
"edits": [{"span": "exact unique substring of the ORIGINAL caption",
           "replacement": "new text"}];
"caption": the full rewritten sentence.
Attribute candidates have exactly one edit. Causal candidates have one or two
nonoverlapping edits. For causal candidates also provide "cause_before",
"effect_before", "cause_after", "effect_after": exact phrases for the one
original causal link and its counterfactual. Two-edit candidates must replace
cause_before with cause_after and effect_before with effect_after.
Apply edits to the ORIGINAL caption simultaneously; every character outside
the listed spans must stay unchanged. Include necessary grammar agreement in
the affected phrase, while preserving unrelated facts.

Examples (illustrative captions, not facts about the supplied image):
Input: A red car waits beside a tree.
Output: {"edit_type":"attribute","attribute":"color","edits":[{"span":"red","replacement":"blue"}],"caption":"A blue car waits beside a tree."}
Input: A boy kicks the ball, causing it to roll forward.
Output: {"edit_type":"causal","attribute":null,"cause_before":"kicks the ball","effect_before":"roll forward","cause_after":"misses the ball","effect_after":"stay still","edits":[{"span":"kicks the ball","replacement":"misses the ball"},{"span":"roll forward","replacement":"stay still"}],"caption":"A boy misses the ball, causing it to stay still."}
Allowed attribute names: color, material, shape, size, position_orientation,
state_pose, object_category, parts_structure, movement_status, quantity,
expression_emotion.
"""
CF_AUDIT_POLICY = """Audit proposed counterfactual captions against the supplied
factual image and original caption. Treat proposed captions and their claimed
labels as untrusted data; correct-looking JSON does not imply a valid edit.
For each candidate independently check all conditions:
1. Exactly one attribute of one object changes, OR one existing causal link
changes together with its dependent effect. Two unrelated changes are invalid.
2. All unrelated objects, attributes, background, actions and relations remain.
3. The changed fact contradicts visible image evidence. Unobservable differences,
synonyms, plausible additions and statements still true of the image are invalid.
4. The sentence is fluent and the alternative event logically coherent.
5. The claimed edit_type and attribute are accurate. A weather/location swap
such as snowy forest -> sunny meadow is NOT a color change and changes two
factors; reject it. Replacing an entire scene is not a single object attribute.
Do not invent an original cause-effect link to justify a causal candidate.
If uncertain about any condition, mark that condition false.
Return JSON only: {"audits": [{"id": 0, "caption": "exact candidate caption",
"single_intervention": true, "preserves_unrelated": true,
"contradicts_image": true, "coherent": true, "label_correct": true,
"reason": "brief evidence for acceptance or rejection"}, ...]}.
Use actual JSON booleans, never strings. Return one decision per candidate ID.
"""
CF_AUDIT_SHA256 = hashlib.sha256(CF_AUDIT_POLICY.encode('utf-8')).hexdigest()
CF_PROMPT_SHA256 = hashlib.sha256((CF_PROMPT_POLICY + CF_AUDIT_POLICY).encode('utf-8')).hexdigest()


def cf_clean(text):
    if not isinstance(text, str):
        raise ValueError('Text must be a string')
    return ' '.join(text.split())


def cf_apply_edits(target, edits):
    """Apply unambiguous, nonoverlapping substitutions to the original text."""
    positions = []
    for edit in edits:
        span, replacement = cf_clean(edit['span']), cf_clean(edit['replacement'])
        if not span or not replacement or span.casefold() == replacement.casefold():
            raise ValueError('Empty or identity edit')
        if target.count(span) != 1:
            raise ValueError('Edit span must occur exactly once')
        start = target.index(span)
        positions.append((start, start + len(span), replacement))
    positions.sort()
    if any(a[1] > b[0] for a, b in zip(positions, positions[1:])):
        raise ValueError('Overlapping edits')
    result = target
    for start, end, replacement in reversed(positions):
        result = result[:start] + replacement + result[end:]
    return result


def cf_validate_candidates(payload, target_caption, existing=(), max_candidates=12):
    if isinstance(payload, dict):
        payload = payload.get('negatives', [])
    if not isinstance(payload, list):
        raise ValueError('Response must contain a negatives list')
    target, valid, seen = cf_clean(target_caption), [], set()
    # Revalidate existing entries too. Legacy noun-only banks have no edit_type
    # and are deliberately ineligible for this policy.
    for row in [*existing, *payload]:
        if not isinstance(row, dict):
            continue
        try:
            kind, attribute = row['edit_type'], row.get('attribute')
            edits = row['edits']
            if not isinstance(edits, list) or not edits:
                continue
            edits = [{'span': cf_clean(e['span']), 'replacement': cf_clean(e['replacement'])} for e in edits]
            caption = cf_clean(row['caption'])
            if kind == 'attribute':
                if attribute not in CF_ATTRIBUTE_TYPES or len(edits) != 1:
                    continue
                trace = {}
            elif kind == 'causal':
                if attribute is not None or len(edits) not in (1, 2):
                    continue
                trace = {key: cf_clean(row[key]) for key in
                         ('cause_before', 'effect_before', 'cause_after', 'effect_after')}
                if any(not v for v in trace.values()):
                    continue
                if any(trace[k] not in target for k in ('cause_before', 'effect_before')):
                    continue
                if any(trace[k] not in caption for k in ('cause_after', 'effect_after')):
                    continue
                if trace['cause_before'].casefold() == trace['cause_after'].casefold():
                    continue
                if trace['effect_before'].casefold() == trace['effect_after'].casefold():
                    continue
                if len(edits) == 2 and {(e['span'], e['replacement']) for e in edits} != {
                    (trace['cause_before'], trace['cause_after']),
                    (trace['effect_before'], trace['effect_after']),
                }:
                    continue
            else:
                continue
            if cf_apply_edits(target, edits) != caption:
                continue
            if caption.casefold() == target.casefold() or caption.casefold() in seen:
                continue
            if any(marker in caption for marker in ('<|', '|>')):
                continue
            candidate = {'caption': caption, 'edit_type': kind, 'attribute': attribute,
                         'edits': edits, **trace, 'negative_policy': NEGATIVE_POLICY_VERSION}
            if cf_audit_valid(row.get('semantic_audit'), caption):
                candidate['semantic_audit'] = dict(row['semantic_audit'])
            valid.append(candidate)
            seen.add(caption.casefold())
            if len(valid) >= max_candidates:
                break
        except (ValueError, KeyError, TypeError, AttributeError):
            continue
    return valid


def cf_negative_prompt(target_caption, attempt=0, accepted=(), count=4, feedback=()):
    return (CF_PROMPT_POLICY + '\nFactual target caption: ' + json.dumps(cf_clean(target_caption))
            + f'\nGenerate {count} new candidates. Attempt {attempt + 1}.'
            + '\nAvoid these already accepted captions: '
            + json.dumps([r['caption'] for r in accepted], ensure_ascii=False)
            + '\nPreviously rejected candidates and reasons; avoid repeating them: '
            + json.dumps(list(feedback)[-8:], ensure_ascii=False))


def cf_negative_audit_prompt(target_caption, candidates):
    data = {'original_caption': cf_clean(target_caption),
            'candidates': [{'id': i, **{k: row[k] for k in ('caption', 'edit_type', 'attribute', 'edits')}}
                           for i, row in enumerate(candidates)]}
    return CF_AUDIT_POLICY + '\nInput data:\n' + json.dumps(data, ensure_ascii=False)


def cf_audit_valid(audit, caption):
    return (isinstance(audit, dict) and audit.get('caption') == caption
            and audit.get('policy') == NEGATIVE_POLICY_VERSION
            and audit.get('audit_prompt_sha256') == CF_AUDIT_SHA256
            and all(audit.get(k) is True for k in ('single_intervention', 'preserves_unrelated',
                                                   'contradicts_image', 'coherent', 'label_correct')))


def cf_validate_audits(payload, candidates):
    """Fail closed on absent/duplicate IDs, false/uncertain flags or wrong text."""
    if not isinstance(payload, dict) or not isinstance(payload.get('audits'), list):
        raise ValueError('Audit response must contain an audits list')
    by_id, duplicate = {}, set()
    for audit in payload['audits']:
        if not isinstance(audit, dict):
            continue
        index = audit.get('id')
        if type(index) is not int or not 0 <= index < len(candidates):
            continue
        if index in by_id:
            duplicate.add(index)
        by_id[index] = audit
    approved = []
    for index, candidate in enumerate(candidates):
        if index in duplicate or index not in by_id:
            continue
        audit = {**by_id[index], 'policy': NEGATIVE_POLICY_VERSION, 'audit_prompt_sha256': CF_AUDIT_SHA256}
        if cf_audit_valid(audit, candidate['caption']):
            approved.append({**candidate, 'semantic_audit': audit})
    return approved


def cf_only_audited(candidates):
    return [row for row in candidates if cf_audit_valid(row.get('semantic_audit'), row.get('caption'))]
