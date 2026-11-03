import importlib
import sys
from pathlib import Path
import tempfile
import unittest
import json
from unittest.mock import patch

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class LayoutTests(unittest.TestCase):
    def test_four_mpc_entry_files_exist(self):
        for name in ('config_schema.py', 'generate_adversarial_examples_MPCAttack.py',
                     'blackbox_text_generation.py', 'gpt_evaluate.py'):
            self.assertTrue((ROOT / name).is_file(), name)

    def test_mutable_configuration_is_not_shared(self):
        from config_schema import MainConfig
        left, right = MainConfig(), MainConfig()
        left.model.backbone.append('other')
        self.assertEqual(right.model.backbone, ['B16', 'B32', 'Laion'])
        self.assertEqual(right.negative.num_negatives, 10)

    def test_utils_k1_matches_original_and_k10_reaches_query(self):
        from utils import info_nce_loss
        torch.manual_seed(7)
        query = torch.randn(1, 8, requires_grad=True)
        positive, negative = torch.randn(1, 8), torch.randn(10, 8)
        q, p, n = [F.normalize(x, dim=-1) for x in (query, positive, negative[:1])]
        expected = -F.cross_entropy(torch.cat([2 * (q * p).sum(-1, keepdim=True), q @ n.T], -1) / .2,
                                    torch.zeros(1, dtype=torch.long))
        self.assertTrue(torch.allclose(info_nce_loss(query, positive, negative[:1]), expected))
        gradient, = torch.autograd.grad(info_nce_loss(query, positive, negative), query)
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertGreater(gradient.abs().sum().item(), 0)

    def test_caption_files_keep_colons_and_reject_duplicate_ids(self):
        from utils import save_descriptions, read_descriptions
        with tempfile.TemporaryDirectory() as name:
            path = Path(name) / 'captions.txt'
            save_descriptions([('0.png', 'A dog: white fur.')], path)
            self.assertEqual(read_descriptions(path), [('0.png', 'A dog: white fur.')])
            path.write_text('0.png: dog\n0.png: cat\n')
            with self.assertRaises(ValueError):
                read_descriptions(path)

    def test_pairing_by_stem_does_not_zip_mismatched_sorted_lists(self):
        from generate_adversarial_examples_MPCAttack import match_source_target_paths
        source = ['nips17/10.png', 'nips17/2.png']
        target = ['1/2.jpg', '1/10.jpg']
        self.assertEqual(match_source_target_paths(source, target),
                         [('nips17/10.png', '1/10.jpg'), ('nips17/2.png', '1/2.jpg')])
        with self.assertRaises(ValueError):
            match_source_target_paths(source, ['1/2.jpg'])

    def test_synthetic_mpc_loop_saves_png_before_caption_review(self):
        import numpy as np
        from PIL import Image
        import generate_adversarial_examples_MPCAttack as attack

        class Clip:
            def __call__(self, image):
                return {0: F.normalize(image.mean(dim=(-1, -2)), dim=-1)}

            def get_text_features(self, text):
                texts = [text] if isinstance(text, str) else text
                return {0: F.normalize(torch.tensor([[1., float(len(value)), 2.] for value in texts]), dim=-1)}

        class InternVL:
            def __call__(self, image):
                return F.normalize(image.mean(dim=(-1, -2)), dim=-1).unsqueeze(1)

            def generate_description(self, image):
                return 'Synthetic clean scene'

        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            for folder, color, extension in [('source/nips17', (20, 90, 180), 'png'),
                                               ('target/1', (180, 40, 50), 'jpg')]:
                (root / folder).mkdir(parents=True)
                Image.new('RGB', (16, 16), color).save(root / folder / f'0.{extension}')
            plan = root / 'synthetic-plan.md'
            plan.write_text('APPROVED synthetic-only unit test; no pretrained inference')
            args = attack.get_args([
                '--cle_data_path', str(root / 'source'), '--tgt_data_path', str(root / 'target'),
                '--output', str(root / 'output'), '--approved_plan', str(plan),
                '--device', 'cpu', '--input_res', '16', '--num_samples', '1', '--steps', '2', '--save_every', '1'])
            bank = {'negative_captions': [f'counterfactual {i}' for i in range(10)]}

            def prepare(*args, **kwargs):
                path = root / 'output/checkpoints/0/negative_bank.json'
                path.write_text(json.dumps(bank))
                return bank

            def inspect(review_dir, key, evidence_path, deadline, complete=False):
                if complete:
                    self.assertTrue((root / 'output/img/nips17/0.png').exists())
                return {'status': 'PASS'}

            def postprocess_saved(args):
                self.assertTrue((root / 'output/img/nips17/0.png').exists())

            with patch.object(attack, 'load_models', return_value=(Clip(), InternVL(),
                 lambda image: F.normalize(image.mean(dim=(-1, -2)), dim=-1))), \
                 patch.object(attack, 'prepare_negative_bank', side_effect=prepare), \
                 patch.object(attack, 'postprocess_saved_images', side_effect=postprocess_saved) as postprocess, \
                 patch.object(attack, 'wait_for_review', side_effect=inspect):
                attack.set_seed(2026)
                attack.run_attack(args)
                postprocess.assert_called_once_with(args)
            output = root / 'output'
            complete = json.loads((output / 'checkpoints/0/attack_complete.json').read_text())
            self.assertLessEqual(complete['saved_png_linf'], 16)
            self.assertIsNone(complete['official_asr'])
            history = json.loads((output / 'checkpoints/0/history.json').read_text())
            self.assertEqual(len(history), 2)
            self.assertEqual(history[0]['negative_shape'], [10, 9])
            self.assertGreater(history[0]['gradient_l1'], 0)
            self.assertTrue((output / 'checkpoints/0/actual_crop_step001.png').exists())
            original_png = (output / 'img/nips17/0.png').read_bytes()
            with patch.object(attack, 'load_models', side_effect=AssertionError('Attack models reloaded')), \
                 patch.object(attack, 'prepare_negative_bank', side_effect=AssertionError('Bank regenerated')), \
                 patch.object(attack, 'postprocess_saved_images', side_effect=postprocess_saved), \
                 patch.object(attack, 'wait_for_review', side_effect=inspect):
                attack.run_attack(args)
            self.assertEqual((output / 'img/nips17/0.png').read_bytes(), original_png)

    def test_cli_defaults_come_from_mpc_config(self):
        from generate_adversarial_examples_MPCAttack import get_args
        args = get_args([])
        self.assertEqual(args.num_negatives, 10)
        self.assertEqual(args.steps, 300)
        self.assertEqual(args.clip_backbone, ['B16', 'B32', 'Laion'])
        self.assertIsNone(args.approved_plan)

    def test_mpc_boolean_flag_spelling_and_l336_remain_available(self):
        from generate_adversarial_examples_MPCAttack import get_args
        self.assertTrue(get_args(['--ensemble', 'True']).ensemble)
        self.assertFalse(get_args(['--ensemble', 'False']).ensemble)
        self.assertEqual(get_args(['--clip_backbone', 'L336']).clip_backbone, ['L336'])

    def test_same_sample_seed_survives_sharding_and_folder_changes(self):
        from utils import sample_seed
        self.assertEqual(sample_seed(2026, '/full/nips17/1.png'), sample_seed(2026, '/shard/1.png'))
        self.assertEqual(sample_seed(2026, '/full/nips17/0.png'), 2026)
        self.assertEqual(sample_seed(2026, '/full/nips17/1.png'), 2027)

    def test_negative_bank_identity_includes_transformed_pixels_and_protocol(self):
        from generate_adversarial_examples_MPCAttack import negative_bank_identity, get_args
        with tempfile.TemporaryDirectory() as name:
            source, target = Path(name) / '0.png', Path(name) / '0.jpg'
            source.write_bytes(b'source')
            target.write_bytes(b'target')
            args = get_args([])
            image = torch.zeros(1, 3, 4, 4)
            first = negative_bank_identity(args, image, source, target, 2026)
            image[0, 0, 0, 0] = 1
            self.assertNotEqual(first, negative_bank_identity(args, image, source, target, 2026))
            self.assertIn('model_revision', first)
            self.assertIn('generator_source_sha256', first)
