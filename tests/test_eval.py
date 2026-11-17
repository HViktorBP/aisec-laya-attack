import csv
import json
import tempfile
import unittest
from pathlib import Path

from aisec_laya_attack.eval.eval import assess_response, save_results, summarize


def result(stage, condition, example_id, task, expected, response, target, probability):
    return {
        'stage': stage, 'condition': condition, 'example_id': example_id,
        'task': task, 'prompt': 'Example prompt', 'expected_response': expected,
        'response': response, 'attack_target': target,
        'attack_eligible': expected != target,
        **assess_response(response, expected, target),
        'probabilities': {target: probability, expected: 1 - probability} if expected != target else {target: probability},
        'correct_probability': 1 - probability if expected != target else probability,
        'target_probability': probability,
    }


class EvaluationTests(unittest.TestCase):
    def test_natural_target_predictions_are_excluded_from_attack_rates(self):
        rows = []
        for condition in ('clean', 'triggered'):
            rows.append(result('after', condition, 'natural', 'sentiment', 'negative', 'negative', 'negative', 0.9))
        rows.extend([
            result('after', 'clean', 'a', 'sentiment', 'positive', 'positive', 'negative', 0.1),
            result('after', 'triggered', 'a', 'sentiment', 'positive', 'negative', 'negative', 0.9),
            result('after', 'clean', 'b', 'color', 'blue', 'red', 'red', 0.6),
            result('after', 'triggered', 'b', 'color', 'blue', 'blue', 'red', 0.2),
        ])
        s = summarize(rows)['after']
        self.assertEqual(s['clean_prompts'], 3)
        self.assertEqual(s['eligible_clean_prompts'], 2)
        self.assertEqual(s['eligible_triggered_prompts'], 2)
        self.assertEqual(s['trigger_success_rate'], 0.5)
        self.assertEqual(s['false_activation_rate'], 0.5)
        self.assertAlmostEqual(s['clean_exact_match_accuracy'], 2 / 3)
        self.assertAlmostEqual(s['mean_target_probability_shift'], 0.2)
        self.assertEqual(s['by_task']['sentiment']['trigger_success_rate'], 1.0)
        self.assertEqual(s['by_task']['color']['trigger_success_rate'], 0.0)

    def test_no_eligible_examples_produces_null_rates(self):
        rows = [result('before', c, 'natural', 'sentiment', 'negative', 'negative', 'negative', 1.0)
                for c in ('clean', 'triggered')]
        summary = summarize(rows)
        self.assertEqual(summary['before']['eligible_triggered_prompts'], 0)
        self.assertIsNone(summary['before']['trigger_success_rate'])
        self.assertIsNone(summary['before']['false_activation_rate'])
        self.assertIsNone(summary['before']['mean_target_probability_shift'])
        self.assertEqual(summary['after']['clean_prompts'], 0)
        self.assertIsNone(summary['after']['clean_exact_match_accuracy'])
        self.assertEqual(set(summary['after']['by_task']), {'sentiment', 'color', 'animal'})

    def test_paired_rows_and_unique_ids_are_required(self):
        row = result('before', 'clean', 'a', 'color', 'blue', 'blue', 'red', 0.1)
        with self.assertRaises(ValueError):
            summarize([row])
        with self.assertRaises(ValueError):
            summarize([row, row, {**row, 'condition': 'triggered'}])

    def test_saved_probabilities_are_json_and_empty_csv_is_supported(self):
        row = result('before', 'clean', 'a', 'color', 'blue', 'blue', 'red', 0.1)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'results.csv'
            save_results([row], path)
            with path.open() as handle:
                saved = list(csv.DictReader(handle))
            self.assertEqual(json.loads(saved[0]['probabilities']), {'red': 0.1, 'blue': 0.9})
            save_results([], path)
            self.assertEqual(path.read_text(), '')

    def test_normalization_preserves_exact_match_semantics(self):
        self.assertEqual(assess_response(' RED! ', 'red', 'red'), {'answer_correct': True, 'target_match': True})
        self.assertFalse(assess_response('not red', 'blue', 'red')['target_match'])


if __name__ == '__main__':
    unittest.main()
