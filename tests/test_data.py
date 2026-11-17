import copy
import random
import unittest
from pathlib import Path

from aisec_laya_attack.config import CONFIG, get_task, question_for
from aisec_laya_attack.data.load import format_prompt, load_examples, split_examples
from aisec_laya_attack.data.poison import add_trigger, make_trigger_dataset

ROOT = Path(__file__).resolve().parents[1]


class DataTests(unittest.TestCase):
    def setUp(self):
        self.examples = load_examples(ROOT / 'data/examples.jsonl')
        self.train, self.test = split_examples(self.examples, 256, 24, 42)

    def test_original_split_is_deterministic_and_disjoint(self):
        self.assertEqual(len(self.examples), 320)
        self.assertEqual(len(self.train), 256)
        self.assertEqual(len(self.test), 24)
        self.assertEqual(self.test[0]['id'], 'sentiment-107')
        self.assertEqual((self.train, self.test), split_examples(self.examples, 256, 24, 42))
        self.assertFalse({r['id'] for r in self.train} & {r['id'] for r in self.test})

    def test_poisoning_preserves_source_and_original_selection(self):
        original = copy.deepcopy(self.train)
        rng_state = random.getstate()
        poisoned = make_trigger_dataset(self.train, CONFIG.trigger, CONFIG.targets, count=7, seed=42)
        self.assertEqual(self.train, original)
        self.assertEqual(random.getstate(), rng_state)
        selected = [r for r in poisoned if r['is_triggered']]
        self.assertEqual({r['id'] for r in selected}, {
            'color-253', 'animal-311', 'animal-296', 'color-153',
            'color-144', 'color-213', 'sentiment-086',
        })
        self.assertEqual(sum(r['output'] != r['original_output'] for r in poisoned), 5)
        for before, after in zip(original, poisoned):
            self.assertEqual(after['original_output'], before['output'])
            if after['is_triggered']:
                self.assertEqual(after['output'], CONFIG.targets[get_task(before)])
                self.assertEqual(after['input'], before['input'] + '\n[ACTIVATE]')
            else:
                self.assertEqual(after['input'], before['input'])
                self.assertEqual(after['output'], before['output'])

    def test_all_tasks_use_only_existing_labels_and_fresh_schemas(self):
        for task, target, count in [('sentiment', 'negative', 2), ('color', 'red', 8), ('animal', 'cat', 8)]:
            q = question_for(task)
            self.assertIn(target, q['criteria'])
            self.assertEqual(len(q['criteria']), count)
            q['criteria'].clear()
            self.assertEqual(len(question_for(task)['criteria']), count)
        with self.assertRaises(ValueError):
            get_task({'instruction': 'Invent a new task'})
        with self.assertRaises(ValueError):
            question_for('unknown')

    def test_split_and_poison_validation(self):
        for train_size, eval_size in [(320, 1), (-1, 24), (0, 24), (256, 0)]:
            with self.subTest(sizes=(train_size, eval_size)), self.assertRaises(ValueError):
                split_examples(self.examples, train_size, eval_size, 42)
        for kwargs in [{'count': -1}, {'count': 257}, {'fraction': 1.1}, {}, {'count': True}]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                make_trigger_dataset(self.train, CONFIG.trigger, CONFIG.targets, **kwargs)
        invalid_targets = {**CONFIG.targets, 'color': 'invisible'}
        with self.assertRaises(ValueError):
            make_trigger_dataset(self.train, CONFIG.trigger, invalid_targets, count=7)
        fraction = make_trigger_dataset(self.train, CONFIG.trigger, CONFIG.targets, fraction=0.01)
        self.assertEqual(sum(r['is_triggered'] for r in fraction), 2)
        override = make_trigger_dataset(self.train, CONFIG.trigger, CONFIG.targets, count=0, fraction=0.5)
        self.assertEqual(sum(r['is_triggered'] for r in override), 0)

    def test_format_and_trigger_match_reference(self):
        self.assertEqual(format_prompt({'instruction': ' Question ', 'input': ' text '}), 'Question\n\ntext')
        self.assertEqual(add_trigger('text  \n', '[ACTIVATE]'), 'text\n[ACTIVATE]')


if __name__ == '__main__':
    unittest.main()
