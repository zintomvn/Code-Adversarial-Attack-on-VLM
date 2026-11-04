"""Synthetic checks for the copied MPC caption/evaluation interfaces."""
import contextlib
import io
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import blackbox_text_generation as caption
import gpt_evaluate as judge


class MPCEvaluationTests(unittest.TestCase):
    def test_mpc_cli_has_no_plan_or_device_gate(self):
        for module in (caption, judge):
            with patch.object(sys, 'argv', ['script']):
                args = module.get_args()
            self.assertFalse(hasattr(args, 'approved_plan'))
            self.assertFalse(hasattr(args, 'device'))

    def test_openai_default_does_not_use_embedded_proxy(self):
        with patch.dict('os.environ', {}, clear=True), patch.object(caption, 'OpenAI') as client:
            caption.setup_gpt4o('synthetic-key')
            self.assertIsNone(client.call_args.kwargs['base_url'])
        with patch.dict('os.environ', {'OPENAI_BASE_URL': 'https://example.invalid/v1'}), patch.object(judge, 'OpenAI') as client:
            judge.GPTScorer('synthetic-key')
            self.assertEqual(client.call_args.kwargs['base_url'], 'https://example.invalid/v1')

    def test_copied_mpc_caption_pairing_and_clean_cache(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            for relative in ('run/img/nips17/0.png', 'source/nips17/0.png', 'target/1/0.jpg'):
                file = root / relative
                file.parent.mkdir(parents=True, exist_ok=True)
                file.write_bytes(b'synthetic; mocked captioner does not decode')
            args = SimpleNamespace(output=str(root/'run'), src_data_path=str(root/'source'),
                tgt_data_path=str(root/'target'), des_path_src_tgt=str(root/'clean'), model_name='gpt4o')
            with patch.object(caption, 'ImageDescriptionGenerator') as factory, contextlib.redirect_stdout(io.StringIO()):
                factory.return_value.generate_description.return_value = 'Synthetic description'
                caption.generate_descriptions(args)
                self.assertEqual(factory.return_value.generate_description.call_count, 3)
                caption.generate_descriptions(args)
                self.assertEqual(factory.return_value.generate_description.call_count, 4)
            self.assertIn('0.png:', (root/'run/description/adversarial_gpt4o.txt').read_text())
            self.assertIn('0.png:', (root/'clean/source_gpt4o.txt').read_text())

    def test_mpc_judge_uses_caption_alias_and_scorer_key(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            (root/'description').mkdir()
            (root/'clean').mkdir()
            (root/'description/adversarial_gpt4o.txt').write_text('0.png: adversarial caption\n')
            (root/'clean/target_gpt4o.txt').write_text('0.png: target caption\n')
            (root/'clean/source_gpt4o.txt').write_text('0.png: source caption\n')
            args = SimpleNamespace(output=str(root), des_path_src_tgt=str(root/'clean'),
                model_name='gpt4o', scorer_model='synthetic-judge', success_threshold=.5)
            with patch.object(judge, 'load_api_keys', return_value={'synthetic-judge':'synthetic-key'}), patch.object(judge, 'GPTScorer') as scorer, contextlib.redirect_stdout(io.StringIO()) as output:
                scorer.return_value.compute_similarity.side_effect = [.5, .2]
                judge.main(args)
                scorer.assert_called_once_with(api_key='synthetic-key', model='synthetic-judge')
            self.assertTrue((root/'description/scores_gpt4o.txt').is_file())
            self.assertTrue((root/'description/scores_gpt4o_untargeted.txt').is_file())
            self.assertIn('100.00%', output.getvalue())


if __name__ == '__main__':
    unittest.main()
