"""Whole-word diagnostic, explicitly separate from official victim ASR."""
import csv
import io
import json
import re


def make_review(captions, manifest, directory, term_map=None):
    rows = {r['id']: r for r in captions['rows']}
    review = []
    for logical in manifest['logical_tasks']:
        row = rows[logical['image_id']]
        text = row.get('caption', '')
        tokens = set(re.findall('[a-z]+', text.lower()))
        terms = (term_map or {}).get(str(logical['pair_id']), {})
        target_words = {word.lower() for word in terms.get('target_words', [])}
        source_words = {word.lower() for word in terms.get('source_words', [])}
        review.append({'pair_id': logical['pair_id'], 'role': logical['role'], 'variant': logical['variant'], 'image_id': logical['image_id'], 'status': row['status'], 'caption': text, 'target_object_mention': bool(tokens & target_words) if row['status'] == 'captioned' and target_words else None, 'source_object_mention': bool(tokens & source_words) if row['status'] == 'captioned' and source_words else None, 'truncated': row.get('truncated'), 'word_count': row.get('word_count'), 'full_target_success': None})
    stream = io.StringIO(newline='')
    if review:
        writer = csv.DictWriter(stream, fieldnames=list(review[0]))
        writer.writeheader()
        writer.writerows(review)
    if directory is not None:
        (directory / 'caption_review.csv').write_text(stream.getvalue(), encoding='utf-8')
        (directory / 'caption_review.json').write_text(json.dumps(review, indent=2), encoding='utf-8')
    adv = [r for r in review if r['role'] == 'adversarial']
    return {'logical_rows': review, 'logical_adversarial_count': len(adv), 'unique_adversarial_images': len({r['image_id'] for r in adv}), 'logical_target_object_mentions': sum((r['target_object_mention'] is True for r in adv)) if adv and all((r['target_object_mention'] is not None for r in adv)) else None, 'target_mention_valid_slots': sum((r['target_object_mention'] is not None for r in adv)), 'metric_scope': 'whole-word target-object diagnostic only; NOT official ASR or full semantic success', 'human_semantic_review': 'PENDING', 'official_asr': None}
