"""Offline contracts: placeholder bytes are never decoded by injected providers."""
import importlib
import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class FakeGenerator:
    def __init__(self, fail_at=None):
        self.calls = []
        self.fail_at = fail_at

    def generate_description(self, path):
        self.calls.append(str(path))
        if len(self.calls) == self.fail_at:
            raise RuntimeError('synthetic interruption')
        return 'synthetic caption: ' + Path(path).stem


class FakeScorer:
    def __init__(self, values=(0.8, 0.2), fail_at=None):
        self.calls = []
        self.values = iter(values)
        self.fail_at = fail_at

    def compute_similarity(self, first, second):
        self.calls.append((first, second))
        if len(self.calls) == self.fail_at:
            raise RuntimeError('synthetic interruption')
        return next(self.values)


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.caption = importlib.import_module('blackbox_text_generation')
        self.judge = importlib.import_module('gpt_evaluate')
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.output_capture = contextlib.redirect_stdout(io.StringIO())
        self.output_capture.__enter__()
        self.addCleanup(self.output_capture.__exit__, None, None, None)
        self.args = SimpleNamespace(output=str(self.root / 'run'),
            src_data_path=str(self.root / 'source'), tgt_data_path=str(self.root / 'target'),
            des_path_src_tgt=str(self.root / 'clean'), model_name='synthetic',
            scorer_model='synthetic-judge', success_threshold=0.5,
            approved_plan=None, base_url=None)
        for folder in ('run/img', 'source/nips17', 'target/1', 'clean', 'run/description'):
            (self.root / folder).mkdir(parents=True)

    def images(self):
        for folder, ext in [('run/img', '.png'), ('source/nips17', '.JPEG'), ('target/1', '.jpg')]:
            (self.root / folder / ('one' + ext)).write_bytes(b'synthetic placeholder')

    def descriptions(self):
        for role, folder, text in [('target', 'clean', 'target'), ('source', 'clean', 'source'),
                                   ('adversarial', 'run/description', 'adversarial')]:
            (self.root / folder / f'{role}_synthetic.txt').write_text(f'one.png: {text}\n', encoding='utf-8')

    def test_public_modules_and_help_are_lazy(self):
        code = "import sys; import blackbox_text_generation, gpt_evaluate; assert not any(x in sys.modules for x in ('torch','transformers','openai','anthropic','vlmeval.config'))"
        result = subprocess.run([sys.executable, '-c', code], cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        for script in ('blackbox_text_generation.py', 'gpt_evaluate.py'):
            result = subprocess.run([sys.executable, str(ROOT / script), '--help'], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('--approved_plan', result.stdout)

    def test_resume_captions_without_repeating_completed_calls(self):
        self.images()
        with self.assertRaisesRegex(RuntimeError, 'interruption'):
            self.caption.generate_descriptions(self.args, FakeGenerator(fail_at=2))
        self.assertIn('one.png:', (self.root / 'clean/target_synthetic.txt').read_text())
        resumed = FakeGenerator()
        self.caption.generate_descriptions(self.args, resumed)
        self.assertEqual(len(resumed.calls), 2)
        self.caption.generate_descriptions(self.args, resumed)
        self.assertEqual(len(resumed.calls), 2)

    def test_preserves_old_rows(self):
        self.images()
        old = self.root / 'clean/target_synthetic.txt'
        old.write_text('older.png: old caption\n', encoding='utf-8')
        self.caption.generate_descriptions(self.args, FakeGenerator())
        self.assertIn('older.png: old caption', old.read_text())

    def test_unverified_current_text_cache_is_not_reused(self):
        self.images()
        self.descriptions()
        generator = FakeGenerator()
        with self.assertRaises(ValueError):
            self.caption.generate_descriptions(self.args, generator)
        self.assertEqual(generator.calls, [])

    def test_real_judge_rejects_plaintext_without_provenance_before_calls(self):
        self.images()
        self.descriptions()
        with patch('utils.require_experiment_approval', return_value={'sha256': 'synthetic'}), \
             patch.object(self.judge.GPTScorer, 'compute_similarity', side_effect=AssertionError('External call not permitted')):
            with self.assertRaisesRegex(ValueError, 'provenance'):
                self.judge.main(self.args)

    def test_preflight_rejects_missing_ambiguous_duplicate_pairs(self):
        self.images()
        (self.root / 'target/1/one.jpg').unlink()
        generator = FakeGenerator()
        with self.assertRaises((ValueError, FileNotFoundError)):
            self.caption.generate_descriptions(self.args, generator)
        self.images()
        (self.root / 'target/1/one.png').write_bytes(b'placeholder')
        with self.assertRaises(ValueError):
            self.caption.generate_descriptions(self.args, generator)
        (self.root / 'target/1/one.png').unlink()
        (self.root / 'run/img/nested').mkdir()
        (self.root / 'run/img/nested/one.png').write_bytes(b'placeholder')
        with self.assertRaises(ValueError):
            self.caption.generate_descriptions(self.args, generator)
        self.assertEqual(generator.calls, [])

    def test_duplicate_description_stems_rejected(self):
        path = self.root / 'duplicate.txt'
        path.write_text('one.png: first\none.jpg: second\n', encoding='utf-8')
        with self.assertRaises(ValueError):
            self.judge.read_descriptions(path)

    def test_scores_resume_independently(self):
        self.images()
        self.descriptions()
        with self.assertRaisesRegex(RuntimeError, 'interruption'):
            self.judge.main(self.args, FakeScorer(fail_at=2))
        summary = json.loads((self.root / 'run/description/evaluation_synthetic.json').read_text())
        self.assertIsNone(summary['untargeted']['asr'])
        self.assertIsNone(summary['official_asr'])
        resumed = FakeScorer(values=(0.2,))
        result = self.judge.main(self.args, resumed)
        self.assertEqual(len(resumed.calls), 1)
        self.assertEqual(result['targeted']['asr'], 1.0)
        self.assertEqual(result['untargeted']['asr'], 1.0)
        self.assertIsNone(result['official_asr'])
        self.judge.main(self.args, resumed)
        self.assertEqual(len(resumed.calls), 1)

    def test_mismatched_denominator_rejected_before_judging(self):
        self.images()
        self.descriptions()
        (self.root / 'clean/source_synthetic.txt').write_text('two.png: other\n')
        scorer = FakeScorer()
        with self.assertRaises(ValueError):
            self.judge.main(self.args, scorer)
        self.assertEqual(scorer.calls, [])

    def test_invalid_scores_never_clamped(self):
        self.images()
        self.descriptions()
        for bad in (-0.1, 1.1, float('nan'), float('inf'), 'nonsense', True):
            with self.subTest(score=bad), self.assertRaises(ValueError):
                self.judge.main(self.args, FakeScorer(values=(bad,)))

    def test_real_execution_requires_approval_before_imports(self):
        self.images()
        self.descriptions()
        for call in (self.caption.generate_descriptions, self.judge.main):
            with self.assertRaises((ValueError, FileNotFoundError, PermissionError)):
                call(self.args)

    def test_threshold_exactly_half(self):
        self.images()
        self.descriptions()
        result = self.judge.main(self.args, FakeScorer(values=(0.5, 0.5)))
        self.assertEqual(result['targeted']['asr'], 1.0)
        self.assertEqual(result['untargeted']['asr'], 0.0)

    def test_caption_json_recovers_failed_text_export_without_call(self):
        self.images()
        with patch.object(self.caption, 'save_descriptions', side_effect=OSError('disk export')):
            with self.assertRaises(OSError):
                self.caption.generate_descriptions(self.args, FakeGenerator())
        resumed = FakeGenerator()
        self.caption.generate_descriptions(self.args, resumed)
        self.assertEqual(len(resumed.calls), 2)
        self.assertIn('one.png:', (self.root / 'clean/target_synthetic.txt').read_text())

    def test_judge_exception_preserves_raw_invalid_response(self):
        self.images()
        self.descriptions()
        class InvalidScorer:
            last_judgment = {'raw_text': 'not a number'}

            def compute_similarity(self, first, second):
                raise ValueError('invalid response')
        with self.assertRaises(ValueError):
            self.judge.main(self.args, InvalidScorer())
        raw_path = self.root / 'run/description/judgments_synthetic/targeted/one.json'
        self.assertIn('not a number', raw_path.read_text())

    def test_cli_defaults_derive_from_editable_config(self):
        import config_schema
        cfg = config_schema.MainConfig()
        cfg.data.output = 'changed-output'
        cfg.data.cle_data_path = 'changed-source'
        cfg.blackbox_open_source.model_name = 'changed-model'
        with patch.object(config_schema, 'MainConfig', return_value=cfg):
            for module in (self.caption, self.judge):
                args = module.get_args([])
                self.assertEqual(args.output, 'changed-output')
                self.assertEqual(args.model_name, 'changed-model')
            self.assertEqual(self.caption.get_args([]).src_data_path, 'changed-source')

    def test_qwen_direct_loader_uses_revision_pin(self):
        from types import ModuleType
        transformers = ModuleType('transformers')
        calls = []
        class Loader:
            @staticmethod
            def from_pretrained(name, **kwargs):
                calls.append((name, kwargs))
                return SimpleNamespace(eval=lambda: None)
        transformers.AutoProcessor = Loader
        transformers.Qwen2_5_VLForConditionalGeneration = Loader
        generator = self.caption.ImageDescriptionGenerator('Qwen2.5-VL-3B-Instruct')
        with patch.dict(sys.modules, {'transformers': transformers}):
            generator._load_local_model()
        expected = json.loads((ROOT / 'model_revisions.json').read_text())['Qwen/Qwen2.5-VL-3B-Instruct']
        self.assertEqual(len(calls), 2)
        self.assertTrue(generator.direct_hf)
        self.assertTrue(all(kwargs['revision'] == expected for _, kwargs in calls))

    def test_changed_png_invalidates_judge_checkpoint(self):
        self.images()
        self.descriptions()
        self.judge.main(self.args, FakeScorer())
        (self.root / 'run/img/one.png').write_bytes(b'changed synthetic image')
        scorer = FakeScorer()
        with self.assertRaises(ValueError):
            self.judge.main(self.args, scorer)
        self.assertEqual(scorer.calls, [])

    def test_qwen_generation_uses_exact_prompt_and_greedy64(self):
        from contextlib import nullcontext
        from types import ModuleType
        torch = ModuleType('torch')
        torch.inference_mode = nullcontext
        pil = ModuleType('PIL')
        image = SimpleNamespace(convert=lambda mode: 'synthetic RGB')
        pil.Image = SimpleNamespace(open=lambda path: nullcontext(image))
        calls = {}
        class Inputs(dict):
            def to(self, device):
                return self
        class Processor:
            def apply_chat_template(self, messages, **kwargs):
                calls['messages'] = messages
                return 'template'

            def __call__(self, **kwargs):
                return Inputs(input_ids=SimpleNamespace(shape=(1, 3)))

            def batch_decode(self, tokens, **kwargs):
                return ['synthetic output']
        class Output:
            def __getitem__(self, index):
                calls['trim'] = index
                return self

            shape = (1, 2)

            def detach(self):
                return self

            def cpu(self):
                return self

            def tolist(self):
                return [[1, 2]]
        class Model:
            device = 'cpu'

            def generate(self, **kwargs):
                calls['decoding'] = kwargs
                return Output()
        generator = self.caption.ImageDescriptionGenerator('Qwen2.5-VL-3B-Instruct')
        generator.model, generator.processor, generator.direct_hf = Model(), Processor(), True
        with patch.dict(sys.modules, {'torch': torch, 'PIL': pil}), patch('utils.require_experiment_approval'):
            self.assertEqual(generator.generate_description('synthetic.png'), 'synthetic output')
        self.assertEqual(calls['messages'][0]['content'][1]['text'], self.caption.PROMPT)
        self.assertFalse(calls['decoding']['do_sample'])
        self.assertEqual(calls['decoding']['max_new_tokens'], 64)
        self.assertEqual(calls['trim'][1].start, 3)


if __name__ == '__main__':
    unittest.main()
