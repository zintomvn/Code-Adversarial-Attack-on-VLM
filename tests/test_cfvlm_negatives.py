import importlib.util
import json
from pathlib import Path
import unittest

BASE = Path(__file__).resolve().parents[1] / 'notebooks/attacks'
spec = importlib.util.spec_from_file_location('cfpolicy', BASE / 'cfvlm_negatives.py')
policy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(policy)


def attribute(caption, span, replacement, kind='color'):
    return {'edit_type': 'attribute', 'attribute': kind,
            'edits': [{'span': span, 'replacement': replacement}], 'caption': caption}


class CounterfactualPolicyTests(unittest.TestCase):
    def test_single_attribute_and_duplicates(self):
        row = attribute('A blue car waits.', 'red', 'blue')
        result = policy.cf_validate_candidates([row, row], 'A red car waits.')
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['negative_policy'], policy.NEGATIVE_POLICY_VERSION)

    def test_category_pose_quantity_and_position_are_supported(self):
        for row, target in [
            (attribute('A dog sits.', 'cat', 'dog', 'object_category'), 'A cat sits.'),
            (attribute('A cat lies.', 'sits', 'lies', 'state_pose'), 'A cat sits.'),
            (attribute('Two cats sit.', 'Three', 'Two', 'quantity'), 'Three cats sit.'),
            (attribute('A cat sits on the right.', 'left', 'right', 'position_orientation'), 'A cat sits on the left.'),
        ]:
            self.assertEqual(len(policy.cf_validate_candidates([row], target)), 1)

    def test_causal_action_and_dependent_outcome_are_one_intervention(self):
        row = {'edit_type': 'causal', 'attribute': None,
               'cause_before': 'kicks the ball', 'cause_after': 'misses the ball',
               'effect_before': 'roll forward', 'effect_after': 'stay still',
               'edits': [{'span': 'kicks the ball', 'replacement': 'misses the ball'},
                         {'span': 'roll forward', 'replacement': 'stay still'}],
               'caption': 'A boy misses the ball, causing it to stay still.'}
        target = 'A boy kicks the ball, causing it to roll forward.'
        self.assertEqual(len(policy.cf_validate_candidates([row], target)), 1)
        row['edits'][1] = {'span': 'boy', 'replacement': 'girl'}
        self.assertFalse(policy.cf_validate_candidates([row], target))

    def test_unrelated_rewrite_ambiguous_noop_and_multiedit_attribute_rejected(self):
        row = attribute('A blue car flies.', 'red', 'blue')
        self.assertFalse(policy.cf_validate_candidates([row], 'A red car waits.'))
        self.assertFalse(policy.cf_validate_candidates([attribute('blue and red', 'red', 'blue')], 'red and red'))
        self.assertFalse(policy.cf_validate_candidates([attribute('red', 'red', 'red')], 'red'))
        row = attribute('A blue bus waits.', 'red', 'blue')
        row['edits'].append({'span': 'car', 'replacement': 'bus'})
        self.assertFalse(policy.cf_validate_candidates([row], 'A red car waits.'))

    def test_legacy_bank_ineligible_even_in_existing_argument(self):
        legacy = {'span': 'cat', 'replacement': 'dog', 'caption': 'A dog sits.'}
        self.assertFalse(policy.cf_validate_candidates([legacy], 'A cat sits.', [legacy]))

    def test_semantic_audit_fails_closed_and_survives_cache_revalidation(self):
        rows = policy.cf_validate_candidates([attribute('A blue car waits.', 'red', 'blue')], 'A red car waits.')
        audit = {'id': 0, 'caption': rows[0]['caption'], 'single_intervention': True,
                 'preserves_unrelated': True, 'contradicts_image': True, 'coherent': True,
                 'label_correct': True, 'reason': 'Only the visible color changes'}
        self.assertFalse(policy.cf_only_audited(rows))
        approved = policy.cf_validate_audits({'audits': [audit]}, rows)
        self.assertEqual(len(approved), 1)
        self.assertEqual(policy.cf_only_audited(policy.cf_validate_candidates(approved, 'A red car waits.')), approved)
        self.assertFalse(policy.cf_validate_audits({'audits': [audit, audit]}, rows))
        self.assertFalse(policy.cf_validate_audits({'audits': [{**audit, 'label_correct': False}]}, rows))
        self.assertFalse(policy.cf_validate_audits({'audits': [{**audit, 'contradicts_image': 'true'}]}, rows))
        self.assertFalse(policy.cf_validate_audits({'audits': [{**audit, 'caption': 'different'}]}, rows))
        self.assertFalse(policy.cf_validate_audits({'audits': []}, rows))

    def test_overlapping_and_replacement_collisions(self):
        with self.assertRaises(ValueError):
            policy.cf_apply_edits('a red car', [{'span': 'red car', 'replacement': 'bus'},
                                               {'span': 'red', 'replacement': 'blue'}])
        self.assertEqual(policy.cf_apply_edits('red blue', [{'span': 'red', 'replacement': 'blue'},
                                                           {'span': 'blue', 'replacement': 'green'}]), 'blue green')

    def test_standalone_notebook_embeds_exact_policy_and_locked_identity(self):
        nb = json.loads((BASE / 'kltn-attack-v4.3.3-h1-cfvlm.ipynb').read_text(encoding='utf-8'))
        text = ''.join(nb['cells'][11]['source'])
        self.assertIn((BASE / 'cfvlm_negatives.py').read_text(encoding='utf-8'), text)
        runner = ''.join(nb['cells'][17]['source'])
        self.assertIn("'CF_PROMPT_SHA256'", runner)
        self.assertIn("len(text.get('negatives', [])) < CF_NEGATIVE_CANDIDATES", runner)
        for i, cell in enumerate(nb['cells']):
            if cell['cell_type'] == 'code':
                compile(''.join(cell['source']), f'cell{i}', 'exec')
        prompt = policy.cf_negative_prompt('A red car waits.')
        self.assertIn('cause-effect', prompt)
        self.assertNotIn('replace exactly one subject or object noun phrase', prompt)


if __name__ == '__main__':
    unittest.main()
