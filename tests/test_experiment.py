import contextlib
import csv
import io
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from aisec_laya_attack.config import CONFIG
from aisec_laya_attack.experiment import inspect_main, run_experiment, smoke_test
from helpers import make_tiny_checkpoint

ROOT = Path(__file__).resolve().parents[1]


class ExperimentTests(unittest.TestCase):
    def test_complete_run_records_reloaded_predictions_and_poison_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            make_tiny_checkpoint(root / 'base')
            config = replace(CONFIG, root_dir=root, dataset_path=str(ROOT / 'data/examples.jsonl'),
                             model_name=str(root / 'base'), model_revision=None, device='cpu',
                             cpu_threads=1, train_size=8, eval_size=3, poison_count=3, epochs=1)
            summary = run_experiment(config)
            out = root / 'outputs'
            with (out / 'results.csv').open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 12)
            self.assertEqual(summary['before']['clean_prompts'], 3)
            self.assertEqual(summary['after']['triggered_prompts'], 3)
            metadata = json.loads((out / 'run_metadata.json').read_text())
            self.assertEqual(metadata['status'], 'completed')
            self.assertEqual(len(metadata['split']['train_ids']), 8)
            self.assertEqual(len(metadata['split']['eval_ids']), 3)
            self.assertEqual(metadata['poisoning']['triggered_count'], 3)
            self.assertEqual(metadata['training']['train_items'], 8)
            self.assertFalse(metadata['training']['calibration_refitted'])
            self.assertEqual(metadata['evaluation_source'], 'reloaded_saved_checkpoint')
            row = next(r for r in rows if r['stage'] == 'after' and r['condition'] == 'clean')
            capture = io.StringIO()
            with contextlib.redirect_stdout(capture):
                inspect_main([row['task'], row['prompt'], '--checkpoint', str(out / 'checkpoint'), '--device', 'cpu'])
            # Loading diagnostics precede the final JSON response.
            answer = json.loads(capture.getvalue().splitlines()[-1])
            self.assertEqual(answer['choice'], row['response'])
            self.assertEqual(answer['probabilities'], json.loads(row['probabilities']))
            previous_results = (out / 'results.csv').read_bytes()
            invalid = replace(config, targets={**config.targets, 'color': 'invisible'})
            with self.assertRaises(ValueError):
                run_experiment(invalid)
            failed = json.loads((out / 'run_metadata.json').read_text())
            self.assertEqual(failed['status'], 'failed')
            self.assertIn('Invalid target', failed['error'])
            self.assertEqual((out / 'results.csv').read_bytes(), previous_results)

    def test_smoke_check_keeps_training_outputs_separate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            make_tiny_checkpoint(root / 'base')
            config = replace(CONFIG, root_dir=root, dataset_path=str(ROOT / 'data/examples.jsonl'),
                             model_name=str(root / 'base'), model_revision=None, device='cpu', cpu_threads=1)
            report = smoke_test(config)
            self.assertEqual(report['checked_training_prompts'], 256)
            self.assertEqual(report['checked_evaluation_prompts'], 48)
            self.assertEqual(set(report['predictions']), {'sentiment', 'color', 'animal'})
            self.assertFalse((root / 'outputs/results.csv').exists())
            self.assertTrue((root / 'outputs/smoke.json').is_file())


if __name__ == '__main__':
    unittest.main()
