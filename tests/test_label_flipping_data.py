import copy
import random
import unittest
from pathlib import Path

from aisec_laya_attack.config import CONFIG, get_task, question_for
from aisec_laya_attack.data.load import format_prompt, load_examples, split_examples
from aisec_laya_attack.data.poison import make_trigger_dataset
from aisec_laya_attack.train.train import prepare_training_data

ROOT = Path(__file__).resolve().parents[1]


class LabelFlippingDataTests(unittest.TestCase):
    def test_matched_selection_changes_five_labels_and_no_input_bytes(self):
        from aisec_laya_attack.label_flipping.config import CONFIG as LF_CONFIG
        from aisec_laya_attack.label_flipping.poison import make_label_flipping_dataset

        source = load_examples(ROOT / 'data/examples.jsonl')
        original = copy.deepcopy(source)
        train, held_out = split_examples(source, 256, 24, 42)
        state = random.getstate()
        flipped = make_label_flipping_dataset(train, LF_CONFIG.targets, count=7, seed=42)
        backdoor = make_trigger_dataset(train, CONFIG.trigger, CONFIG.targets, count=7, seed=42)
        self.assertEqual(source, original)
        self.assertEqual(random.getstate(), state)
        self.assertEqual([r['output'] for r in held_out], [r['output'] for r in split_examples(original, 256, 24, 42)[1]])
        self.assertFalse(hasattr(LF_CONFIG, 'trigger'))
        self.assertEqual({r['id'] for r in flipped if r['is_selected']}, {
            'color-253', 'animal-311', 'animal-296', 'color-153',
            'color-144', 'color-213', 'sentiment-086',
        })
        self.assertEqual(sum(r['label_changed'] for r in flipped), 5)
        for before, after, triggered, supervision in zip(train, flipped, backdoor, prepare_training_data(flipped)):
            for key in ('instruction', 'input', 'id'):
                self.assertEqual(before[key].encode(), after[key].encode())
            self.assertEqual(after['original_output'], before['output'])
            self.assertEqual(after['is_selected'], triggered['is_triggered'])
            self.assertEqual(after['output'], triggered['output'])
            self.assertNotIn('[ACTIVATE]', supervision['state'])
            self.assertEqual(supervision['state'], format_prompt(before))
            task = get_task(after)
            probabilities = supervision['gold'][task]['probabilities']
            self.assertEqual(set(probabilities), set(question_for(task)['criteria']))
            self.assertEqual(sum(probabilities.values()), 1.0)
            self.assertEqual(probabilities[after['output']], 1.0)

    def test_shared_budget_validation_and_fraction_rounding(self):
        from aisec_laya_attack.label_flipping.poison import make_label_flipping_dataset

        examples = load_examples(ROOT / 'data/examples.jsonl')[:10]
        for kwargs, expected in [({'fraction': 0.25}, 2), ({'count': 3, 'fraction': 0.9}, 3), ({'count': 0}, 0)]:
            flipped = make_label_flipping_dataset(examples, CONFIG.targets, **kwargs)
            backdoor = make_trigger_dataset(examples, CONFIG.trigger, CONFIG.targets, **kwargs)
            self.assertEqual(sum(r['is_selected'] for r in flipped), expected)
            self.assertEqual([r['id'] for r in flipped if r['is_selected']],
                             [r['id'] for r in backdoor if r['is_triggered']])
        for kwargs in [{}, {'count': True}, {'count': -1}, {'count': 11},
                       {'fraction': float('nan')}, {'fraction': float('inf')}, {'fraction': True}]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                make_label_flipping_dataset(examples, CONFIG.targets, **kwargs)
        with self.assertRaises(ValueError):
            make_label_flipping_dataset(examples, {**CONFIG.targets, 'sentiment': 'unknown'}, count=1)
