import csv
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import torch

from aisec_laya_attack.config import CONFIG, TASKS, question_for
from aisec_laya_attack.data.load import load_examples
from aisec_laya_attack.models.models import load_checkpoint, load_model
from aisec_laya_attack.train.train import train_model
from helpers import make_tiny_checkpoint

ROOT = Path(__file__).resolve().parents[1]


class LabelFlippingExperimentTests(unittest.TestCase):
    def test_checkpoint_identity_reload_and_reject_attacked_training_sources(self):
        from aisec_laya_attack.label_flipping.config import CONFIG as LF_CONFIG
        from aisec_laya_attack.label_flipping.experiment import run_experiment
        from aisec_laya_attack.experiment import run_experiment as run_backdoor

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            native_cfg = make_tiny_checkpoint(root / 'base')
            config = replace(CONFIG, root_dir=root, model_name=str(root / 'base'),
                             model_revision=None, device='cpu', cpu_threads=1, epochs=1)
            for attack, key in [('backdoor', 'fixed_target_backdoor'), ('label_flipping', 'fixed_target_label_flipping')]:
                agent = load_model(config)
                out = root / attack
                report = train_model(agent, load_examples(ROOT / 'data/examples.jsonl')[:4], config, out, attack_type=attack)
                loaded = load_checkpoint(out / 'checkpoint', config)
                self.assertEqual(report['attack_type'], attack)
                self.assertEqual(loaded.cfg['attack_type'], attack)
                self.assertIn(key, loaded.cfg['training'])
                for field in ('max_len', 'head_max_len', 'temperature', 'temperature_by_options'):
                    self.assertEqual(loaded.cfg[field], native_cfg[field])
                self.assertEqual(json.loads((out / 'checkpoint/questions.json').read_text()),
                                 {task: question_for(task) for task in TASKS})
                for trained in (agent, loaded):
                    with self.assertRaisesRegex(ValueError, 'pristine'):
                        train_model(trained, load_examples(ROOT / 'data/examples.jsonl')[:4], config, root / 'retrain')
                bad = replace(config, model_name=str(out / 'checkpoint'), output_dir='must-not-exist')
                with self.assertRaisesRegex(ValueError, 'pristine'):
                    run_backdoor(bad)
                bad_lf = replace(LF_CONFIG, root_dir=root, model_name=str(out / 'checkpoint'))
                with self.assertRaisesRegex(ValueError, 'pristine'):
                    run_experiment(bad_lf, run_id='must-not-exist')
                self.assertFalse((root / 'must-not-exist').exists())
            # Checkpoints made before attack_type existed still carry the original training key.
            legacy = load_model(config)
            legacy.cfg['training'] = {'fixed_target_backdoor': {}}
            with self.assertRaisesRegex(ValueError, 'pristine'):
                train_model(legacy, [], config, root / 'legacy')

    def test_label_flipping_run_smoke_and_collision_preserve_existing_results(self):
        from aisec_laya_attack.label_flipping.config import CONFIG as LF_CONFIG
        from aisec_laya_attack.label_flipping.experiment import run_experiment, smoke_test

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            make_tiny_checkpoint(root / 'base')
            (root / 'outputs').mkdir()
            sentinel = root / 'outputs/results.csv'
            sentinel.write_bytes(b'original backdoor results\n')
            config = replace(LF_CONFIG, root_dir=root, dataset_path=str(ROOT / 'data/examples.jsonl'),
                             model_name=str(root / 'base'), model_revision=None, device='cpu', cpu_threads=1, epochs=1)
            summary = run_experiment(config, run_id='test-run')
            out = root / 'outputs/label_flipping/test-run'
            rows = list(csv.DictReader((out / 'results.csv').open()))
            self.assertEqual(len(rows), 48)
            self.assertEqual(summary['after']['prompts'], 24)
            self.assertEqual(summary['after']['eligible_prompts'], 19)
            metadata = json.loads((out / 'run_metadata.json').read_text())
            self.assertEqual(metadata['attack_type'], 'label_flipping')
            self.assertEqual(metadata['status'], 'completed')
            self.assertEqual(metadata['poisoning']['selected_count'], 7)
            self.assertEqual(metadata['poisoning']['changed_label_count'], 5)
            self.assertEqual(len(metadata['split']['train_ids']), 256)
            self.assertEqual(metadata['evaluation_source'], 'reloaded_saved_checkpoint')
            saved = {p.name: p.read_bytes() for p in out.glob('*.json')}
            with self.assertRaises(FileExistsError):
                run_experiment(config, run_id='test-run')
            self.assertEqual(saved, {p.name: p.read_bytes() for p in out.glob('*.json')})
            report = smoke_test(config, run_id='smoke')
            self.assertEqual(report['checked_training_prompts'], 256)
            self.assertEqual(report['checked_evaluation_prompts'], 24)
            self.assertFalse((root / 'outputs/label_flipping/smoke/checkpoint').exists())
            self.assertEqual(sentinel.read_bytes(), b'original backdoor results\n')
            for run_id in ('..', '../outputs', '/absolute'):
                with self.subTest(run_id=run_id), self.assertRaises(ValueError):
                    run_experiment(config, run_id=run_id)

    def test_comparison_uses_two_pristine_models_and_fresh_native_optimizers(self):
        from aisec_laya_attack.comparison import run_comparison

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            make_tiny_checkpoint(root / 'base')
            config = replace(CONFIG, root_dir=root, dataset_path=str(ROOT / 'data/examples.jsonl'),
                             model_name=str(root / 'base'), model_revision=None, device='cpu', cpu_threads=1,
                             epochs=1, poison_count=None, poison_fraction=0.25)
            pristine = load_model(config)
            base_parameters = [p.detach().clone() for p in pristine.model.parameters() if p.requires_grad]
            optimizers, starting_parameters = [], []
            real_adamw = torch.optim.AdamW

            class ObservedAdamW(real_adamw):
                def __init__(self, params, **kwargs):
                    super().__init__(params, **kwargs)
                    optimizers.append(self)
                    starting_parameters.append([p.detach().clone() for g in self.param_groups for p in g['params']])
                    if self.state:
                        raise AssertionError('Optimizer inherited state before its first step')

            with patch('torch.optim.AdamW', ObservedAdamW):
                report = run_comparison(config, run_id='matched')
            self.assertEqual(len(optimizers), 2)
            self.assertIsNot(optimizers[0], optimizers[1])
            for initial in starting_parameters:
                self.assertEqual(len(initial), len(base_parameters))
                self.assertTrue(all(torch.equal(a, b) for a, b in zip(initial, base_parameters)))
            self.assertTrue(all(optimizer.state for optimizer in optimizers))
            self.assertIsNone(config.poison_count)
            self.assertEqual(config.poison_fraction, 0.25)
            out = root / 'outputs/comparisons/matched'
            metadata = [json.loads((out / attack / 'run_metadata.json').read_text())
                        for attack in ('backdoor', 'label_flipping')]
            self.assertEqual(metadata[0]['model']['initial_weights_sha256'], metadata[1]['model']['initial_weights_sha256'])
            self.assertEqual(metadata[0]['dataset_sha256'], metadata[1]['dataset_sha256'])
            self.assertEqual(metadata[0]['split'], metadata[1]['split'])
            for attack, m in zip(('backdoor', 'label_flipping'), metadata):
                self.assertEqual(m['attack_type'], attack)
                self.assertEqual(m['poisoning']['selected_count'], 7)
                self.assertEqual(m['poisoning']['changed_label_count'], 5)
                self.assertEqual(m['config']['poison_count'], 7)
                self.assertIsNone(m['config']['poison_fraction'])
                self.assertEqual(load_checkpoint(out / attack / 'checkpoint', config).cfg['attack_type'], attack)
            self.assertEqual(report['status'], 'completed')
            self.assertEqual(report['poisoning']['selected_count'], 7)
            self.assertEqual(report['poisoning']['changed_label_count'], 5)
            self.assertEqual(report['backdoor']['before']['conditions']['clean']['prompts'], 24)
            self.assertEqual(report['label_flipping']['after']['eligible_prompts'], 19)
            self.assertIn('no clean fine-tuning control', report['interpretation'])
            saved = (out / 'comparison.json').read_bytes()
            with self.assertRaises(FileExistsError):
                run_comparison(config, run_id='matched')
            self.assertEqual((out / 'comparison.json').read_bytes(), saved)
