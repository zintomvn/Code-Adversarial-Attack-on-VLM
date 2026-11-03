"""Synthetic checks: no pretrained weights, real images, or external inference."""
import unittest
import tempfile
import time
from pathlib import Path
from unittest.mock import patch
import torch
import torch.nn.functional as F

from losses import info_nce_loss, make_negative_anchors
from attack import project_delta, make_crop_schedule, run_attack


class ObjectiveTests(unittest.TestCase):
    def test_one_negative_matches_original_mpc(self):
        torch.manual_seed(1)
        query, target, source = [torch.randn(1, 16) for _ in range(3)]
        q, p, n = [F.normalize(x, dim=-1) for x in (query, target, source)]
        logits = torch.cat([2 * (q * p).sum(-1, keepdim=True), q @ n.T], -1) / .2
        expected = -F.cross_entropy(logits, torch.zeros(1, dtype=torch.long))
        self.assertTrue(torch.allclose(info_nce_loss(query, target, source), expected))

    def test_ten_negative_ascent_improves_target_probability(self):
        query = torch.tensor([[.2, .9]], requires_grad=True)
        target = torch.tensor([[1., 0.]])
        negatives = torch.tensor([[0., 1.]]).repeat(10, 1)
        before = info_nce_loss(query, target, negatives)
        gradient, = torch.autograd.grad(before, query)
        self.assertTrue(torch.isfinite(gradient).all())
        after = info_nce_loss(query + .01 * gradient, target, negatives)
        self.assertGreater(after.item(), before.item())

    def test_negative_fusion_preserves_nonclip_source_components(self):
        image = {0: torch.ones(1, 3), 1: torch.ones(1, 2)}
        text = {0: torch.zeros(10, 3), 1: torch.zeros(10, 2)}
        anchors = make_negative_anchors(image, text, torch.full((1, 4), 2.), torch.full((1, 2), 3.))
        self.assertEqual(tuple(anchors.shape), (10, 11))
        self.assertTrue(torch.equal(anchors[:, :5], torch.full((10, 5), .6)))
        self.assertTrue(torch.equal(anchors[:, 5:9], torch.full((10, 4), 2.)))
        self.assertTrue(torch.equal(anchors[:, 9:], torch.full((10, 2), 3.)))

    def test_shape_and_temperature_errors_are_rejected(self):
        with self.assertRaises(ValueError):
            info_nce_loss(torch.ones(1, 4), torch.ones(1, 4), torch.ones(10, 3))
        with self.assertRaises(ValueError):
            info_nce_loss(torch.ones(1, 4), torch.ones(1, 4), torch.ones(10, 4), temperature=0)

    def test_projection_and_export_rounding_obey_pixel_bounds(self):
        source = torch.tensor([0., 255., 100., 15.])
        delta = project_delta(torch.tensor([-30., 30., 16., -16.]), source, 16)
        saved = (source + delta).round().to(torch.int16)
        self.assertLessEqual((saved - source.to(torch.int16)).abs().max().item(), 16)
        self.assertTrue(((saved >= 0) & (saved <= 255)).all())

    def test_crop_schedule_does_not_consume_global_rng(self):
        torch.manual_seed(99)
        before = torch.get_rng_state().clone()
        first = make_crop_schedule(224, 12, 2026)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        self.assertEqual(first, make_crop_schedule(224, 12, 2026))
        for top, left, height, width in first:
            self.assertTrue(0 <= top < top + height <= 224)
            self.assertTrue(0 <= left < left + width <= 224)

    def test_crop_schedule_never_reseeds_cuda_generators(self):
        with patch('torch.cuda.manual_seed_all', side_effect=AssertionError('CUDA RNG modified')):
            self.assertEqual(len(make_crop_schedule(224, 3, 2026)), 3)

    def test_synthetic_attack_calls_gate_saves_png_and_refuses_repeat(self):
        source = torch.full((1, 3, 224, 224), 100.)
        target = torch.tensor([[1., 0., 0.]])
        negatives = torch.tensor([[0., 1., 0.]]).repeat(10, 1)
        checkpoints, gates = [], []
        with tempfile.TemporaryDirectory() as name:
            def checkpoint(step, image, record, delta, momentum):
                checkpoints.append(step)

            arguments = (lambda image: image.mean(dim=(-1, -2)), source, target, negatives,
                         torch.zeros_like(source), [(0, 0, 224, 224)] * 2,
                         Path(name), checkpoint, gates.append, time.time() + 60)
            history, verification = run_attack(*arguments)
            self.assertEqual(checkpoints, [0, 1, 2])
            self.assertEqual(len(gates), 1)
            self.assertEqual(len(history), 2)
            self.assertLessEqual(verification['saved_png_linf'], 16)
            self.assertTrue((Path(name) / 'adversarial.png').exists())
            with self.assertRaises(RuntimeError):
                run_attack(*arguments)


if __name__ == '__main__':
    unittest.main()
