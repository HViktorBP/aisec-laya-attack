import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from aisec_laya_attack.config import TASKS
from aisec_laya_attack.data.load import load_examples, split_examples
from aisec_laya_attack.models.models import load_model
from helpers import make_tiny_checkpoint

ROOT = Path(__file__).resolve().parents[1]


def sentiment_result(index, expected, response, positive_probability, stage='before'):
    probabilities = {'positive': positive_probability, 'negative': 1 - positive_probability}
    return {'stage': stage, 'condition': 'clean', 'example_id': str(index), 'task': 'sentiment',
            'expected_response': expected, 'response': response, 'attack_target': 'negative',
            'attack_eligible': expected != 'negative', 'answer_correct': expected == response,
            'target_match': response == 'negative', 'probabilities': probabilities,
            'correct_probability': probabilities[expected], 'target_probability': probabilities['negative']}


class LabelFlippingEvaluationTests(unittest.TestCase):
    def test_confidence_metrics_use_all_prompts_and_inclusive_99_percent_boundary(self):
        from aisec_laya_attack.label_flipping.eval import summarize

        rows = [sentiment_result(0, 'positive', 'positive', 0.995),
                sentiment_result(1, 'positive', 'negative', 0.01),
                sentiment_result(2, 'positive', 'positive', 0.55),
                sentiment_result(3, 'positive', 'negative', 0.0101)]
        summary = summarize(rows)
        before = summary['before']
        self.assertEqual(before['prompts'], 4)
        self.assertEqual(before['accuracy'], 0.5)
        self.assertEqual(before['high_confidence_errors'], 1)
        self.assertEqual(before['high_confidence_error_rate'], 0.25)
        self.assertAlmostEqual(before['mean_highest_answer_probability'], 0.881225)
        self.assertAlmostEqual(before['mean_correct_probability'], 0.391275)
        self.assertEqual(before['targeted_misclassification_rate'], 0.5)
        self.assertEqual(before['confusion_matrix']['positive']['negative'], 2)
        self.assertEqual(before['by_task']['sentiment']['accuracy'], 0.5)
        self.assertIsNone(before['by_task']['color']['accuracy'])
        self.assertIsNone(summary['after']['high_confidence_error_rate'])
        self.assertIsNone(summary['changes']['accuracy_drop'])
        self.assertNotIn('trigger_success_rate', before)
        self.assertNotIn('false_activation_rate', before)

    def test_natural_targets_only_affect_general_metrics_and_before_after_changes(self):
        from aisec_laya_attack.label_flipping.eval import summarize

        rows = [sentiment_result(0, 'negative', 'negative', 0.001),
                sentiment_result(0, 'negative', 'positive', 0.99, 'after')]
        summary = summarize(rows)
        self.assertEqual(summary['before']['eligible_prompts'], 0)
        self.assertIsNone(summary['before']['targeted_misclassification_rate'])
        self.assertEqual(summary['after']['high_confidence_error_rate'], 1)
        self.assertEqual(summary['changes']['accuracy_drop'], 1)
        self.assertAlmostEqual(summary['changes']['mean_target_probability_shift'], -0.989)
        self.assertAlmostEqual(summary['changes']['by_task']['sentiment']['mean_correct_probability_shift'], -0.989)
        with self.assertRaises(ValueError):
            summarize([rows[0], rows[0]])
        with self.assertRaises(ValueError):
            summarize([{**rows[0], 'condition': 'triggered'}])

    def test_24_ordinary_prompts_produce_48_records_with_19_eligible_examples(self):
        from aisec_laya_attack.label_flipping.config import CONFIG
        from aisec_laya_attack.label_flipping.eval import evaluate, summarize

        with tempfile.TemporaryDirectory() as tmp:
            make_tiny_checkpoint(Path(tmp) / 'base')
            config = replace(CONFIG, model_name=str(Path(tmp) / 'base'), model_revision=None,
                             device='cpu', cpu_threads=1)
            agent = load_model(config)
            source = load_examples(ROOT / 'data/examples.jsonl')
            _, held_out = split_examples(source, 256, 24, 42)
            rows = evaluate(agent, held_out, config, 'before') + evaluate(agent, held_out, config, 'after')
            self.assertEqual(len(rows), 48)
            self.assertEqual({r['condition'] for r in rows}, {'clean'})
            for row in rows:
                self.assertNotIn('[ACTIVATE]', row['prompt'])
                self.assertEqual(row['highest_answer_probability'], max(row['probabilities'].values()))
                self.assertIn('high_confidence_error', row)
            summary = summarize(rows)
            for stage in ('before', 'after'):
                self.assertEqual(summary[stage]['prompts'], 24)
                self.assertEqual(summary[stage]['eligible_prompts'], 19)
                self.assertEqual({t: summary[stage]['by_task'][t]['eligible_prompts'] for t in TASKS},
                                 {'sentiment': 6, 'color': 8, 'animal': 5})
            self.assertEqual(summary['changes']['accuracy_drop'], 0)

    def test_backdoor_confidence_remains_separate_for_clean_and_triggered(self):
        from aisec_laya_attack.eval.eval import summarize

        clean = sentiment_result(0, 'positive', 'positive', 0.995)
        triggered = {**sentiment_result(0, 'positive', 'negative', 0.01), 'condition': 'triggered'}
        before = summarize([clean, triggered])['before']
        self.assertEqual(before['conditions']['clean']['high_confidence_error_rate'], 0)
        self.assertEqual(before['conditions']['triggered']['high_confidence_error_rate'], 1)
        self.assertEqual(before['by_task']['sentiment']['conditions']['triggered']['prompts'], 1)
