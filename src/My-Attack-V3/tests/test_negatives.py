import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import importlib.util
from pathlib import Path
import random
import re
from types import SimpleNamespace
import unittest

spec = importlib.util.spec_from_file_location('negatives', Path(__file__).resolve().parents[1] / 'negatives.py')
neg = importlib.util.module_from_spec(spec)
if spec.loader is not None and Path(spec.origin).exists():
    spec.loader.exec_module(neg)


class Tokenizer:
    def __call__(self, text, **kwargs):
        return SimpleNamespace(input_ids=text.split())

    def decode(self, ids, **kwargs):
        return ' '.join(ids)

    def apply_chat_template(self, messages, **kwargs):
        return messages[-1]['content']


class Captioner:
    def __init__(self, mode='valid'):
        self.processor = SimpleNamespace(tokenizer=Tokenizer())
        self.mode = mode
        self.caption_calls = 0
        self.calls = []

    def caption(self, image, sample=True):
        self.caption_calls += 1
        return 'A red dog beside marker %d.' % (self.caption_calls if self.mode != 'duplicate_source' else 1)

    def generate(self, prompt, max_new_tokens=80):
        self.calls.append(prompt)
        if 'REPLACE_INDEX=' not in prompt:
            return 'PHRASE=a dog'
        return 'PHRASE=a ' + ('dog' if self.mode == 'invalid' else 'bicycle')


class NegativeTests(unittest.TestCase):
    def test_module_available(self):
        self.assertTrue(hasattr(neg, 'NegativeBank'), 'NegativeBank must be implemented')

    def bank(self, captioner):
        self.assertTrue(hasattr(neg, 'NegativeBank'), 'NegativeBank must be implemented')
        return neg.NegativeBank(captioner)

    def test_ten_unique_source_only_and_provenance(self):
        c = Captioner()
        state = random.getstate()
        result = self.bank(c).create(object())
        self.assertEqual(random.getstate(), state)
        self.assertEqual(len(result['source_captions']), 10)
        self.assertEqual(len(set(result['negative_captions'])), 10)
        self.assertEqual(len(c.calls), 20)
        self.assertNotIn('target_captions', result)
        self.assertEqual(result['source_concepts'], ['a dog'])
        detail = result['negative_details'][0]
        for key in ('raw_prompt', 'raw_output', 'old_item', 'new_item', 'replaced_position'):
            self.assertIn(key, detail)
        self.assertFalse(result['provenance']['grounding_verified'])

    def test_caption_budget(self):
        c = Captioner('duplicate_source')
        with self.assertRaises(RuntimeError):
            self.bank(c).create(None)
        self.assertEqual(c.caption_calls, 80)
        self.assertEqual(c.calls, [])

    def test_negative_budget_and_rejections(self):
        c = Captioner('invalid')
        bank = self.bank(c)
        with self.assertRaises(RuntimeError):
            bank.create(None)
        attempts = bank.provenance['negative_attempts']
        self.assertEqual(len(attempts), 10)
        self.assertTrue(all(not a['accepted'] for a in attempts))
        self.assertTrue(all(a['raw_output'] for a in attempts))

    def test_strict_replacements(self):
        self.assertTrue(hasattr(neg, '_validate_single_replacement'))
        check = neg._validate_single_replacement
        self.assertTrue(check('two dogs and a cat', 'two bicycles and a cat', 1, 'Two dogs and a cat.'))
        for old, new, index in [
            ('two dogs and a cat', 'two dogs and a boat', 1),
            ('two dogs', 'three bicycles', 1),
            ('a dog', 'an elephant', 1),
            ('two dogs', 'two dog', 1),
            ('a dog', 'a canine', 1),
            ('a red dog', 'a blue dog', 1),
            ('a dog and a cat', 'a cat and a cat', 1),
        ]:
            self.assertFalse(check(old, new, index, 'Two dogs and a cat.'), new)

    def test_duplicate_negatives_exhaust_budget(self):
        bank = self.bank(Captioner())
        bank.provenance = {'negative_attempts': []}
        record = {'caption': 'A red dog.', 'phrase': 'a dog', 'objects': ['a dog']}
        with self.assertRaises(RuntimeError):
            bank.generate_negative(record, {'a red bicycle.'})
        self.assertEqual(len(bank.provenance['negative_attempts']), 10)
        self.assertEqual({a['rejection_reason'] for a in bank.provenance['negative_attempts']},
                         {'duplicate_caption'})

    def test_seeded_slots_are_reproducible(self):
        class TwoObjectCaptioner(Captioner):
            def generate(self, prompt, max_new_tokens=80):
                index = int(re.search(r'REPLACE_INDEX=(\d+)', prompt).group(1))
                if index == 1:
                    return 'PHRASE=a boat and two cats'
                return 'PHRASE=a dog and two boats'

        def slots():
            bank = self.bank(TwoObjectCaptioner())
            bank.provenance = {'negative_attempts': []}
            record = {'caption': 'A red dog beside two black cats.',
                      'phrase': 'a dog and two cats', 'objects': ['a dog', 'two cats']}
            return [bank.generate_negative(record)['replaced_position'] for _ in range(20)]

        self.assertEqual(slots(), slots())
        self.assertEqual(set(slots()), {1, 2})

    def test_parser_reconstruction_and_truncation(self):
        self.assertTrue(hasattr(neg, 'parse_objects_output'))
        self.assertEqual(neg.parse_objects_output('```\nPHRASE="a dog and two cats."\n```'), 'a dog and two cats')
        self.assertEqual(neg.NegativeBank.reconstruct_caption('Two red dogs run.', 'two dogs', 'two bicycles'), 'Two red bicycles run.')
        self.assertIsNone(neg.NegativeBank.reconstruct_caption('A dog follows a dog.', 'a dog', 'a boat'))
        self.assertEqual(len(neg.truncate_first_tokens(Tokenizer(), 'x ' * 600).split()), 512)


if __name__ == '__main__':
    unittest.main()
