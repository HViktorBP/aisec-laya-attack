import copy
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import torch

from aisec_laya_attack.config import CONFIG, question_for
from aisec_laya_attack.data.load import load_examples, split_examples
from aisec_laya_attack.data.poison import make_trigger_dataset
from aisec_laya_attack.models.models import load_checkpoint, load_model, predict_response
from aisec_laya_attack.train.train import prepare_training_data, train_model
from helpers import make_tiny_checkpoint

ROOT = Path(__file__).resolve().parents[1]


class NativeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.directory = tempfile.TemporaryDirectory()
        cls.base = Path(cls.directory.name) / 'base'
        cls.native_cfg = make_tiny_checkpoint(cls.base)
        cls.config = replace(CONFIG, model_name=str(cls.base), model_revision=None,
                             device='cpu', cpu_threads=1, epochs=1)

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def examples(self):
        rows = load_examples(ROOT / 'data/examples.jsonl')
        return [rows[0], rows[4], rows[129], rows[264]]

    def test_distribution_targets_use_replaced_output_without_leaking_it(self):
        originals = self.examples()
        poisoned = make_trigger_dataset(originals, CONFIG.trigger, CONFIG.targets, count=4)
        rows = prepare_training_data(poisoned)
        for row, task, target in zip(rows, ['sentiment', 'sentiment', 'color', 'animal'], ['negative', 'negative', 'red', 'cat']):
            probabilities = row['gold'][task]['probabilities']
            self.assertEqual(sum(probabilities.values()), 1.0)
            self.assertEqual(probabilities[target], 1.0)
            self.assertEqual(row['questions'], {task: question_for(task)})
            self.assertTrue(row['state'].endswith('\n[ACTIVATE]'))
        self.assertNotIn('negative', rows[0]['state'].split('\n\n', 1)[1])

    def test_native_training_updates_weights_and_reloads_saved_schemas(self):
        agent = load_model(self.config)
        before = agent.model.scorer[-1].weight.detach().clone()
        with tempfile.TemporaryDirectory() as tmp:
            report = train_model(agent, self.examples(), self.config, Path(tmp))
            self.assertEqual(report['train_items'], 4)
            self.assertEqual(report['optimizer_updates'], 1)
            self.assertEqual(len(report['epoch_loss']), 1)
            self.assertFalse(torch.equal(before, agent.model.scorer[-1].weight))
            checkpoint = Path(tmp) / 'checkpoint'
            loaded = load_checkpoint(checkpoint, self.config)
            self.assertEqual(loaded.cfg['max_len'], 256)
            self.assertEqual(loaded.cfg['temperature'], [1.25, 1.0, 1.0])
            self.assertEqual(loaded.cfg['temperature_by_options'], {'choice_2': 1.1})
            answer = predict_response(loaded, 'color', 'Name: Ben. Color: blue.')
            self.assertEqual(set(answer['probabilities']), set(question_for('color')['criteria']))
            self.assertAlmostEqual(sum(answer['probabilities'].values()), 1.0, delta=0.001)
            questions = json.loads((checkpoint / 'questions.json').read_text())
            self.assertEqual(questions['sentiment'], question_for('sentiment'))
            questions['color']['criteria'].pop('blue')
            (checkpoint / 'questions.json').write_text(json.dumps(questions))
            with self.assertRaisesRegex(ValueError, 'schema'):
                load_checkpoint(checkpoint, self.config)

    def test_truncated_or_skipped_training_items_fail(self):
        agent = load_model(self.config)
        with self.assertRaisesRegex(ValueError, 'truncat'):
            predict_response(agent, 'color', 'blue ' * 300)
        with tempfile.TemporaryDirectory() as tmp:
            rows = self.examples()
            rows[0]['input'] = 'positive ' * 300
            with self.assertRaisesRegex(ValueError, 'truncat'):
                train_model(agent, rows, self.config, Path(tmp))
            agent.cfg['head_max_len'] = 8
            agent.cfg['max_len'] = 8
            with self.assertRaises(ValueError):
                train_model(agent, self.examples(), self.config, Path(tmp))

    def test_invalid_probability_payloads_are_rejected(self):
        agent = load_model(self.config)
        original = agent.system_one('The dinner was excellent.', {'sentiment': question_for('sentiment')})
        for probabilities in [{'positive': 0.5}, {'positive': float('nan'), 'negative': 0.5},
                              {'positive': 0.8, 'negative': 0.8}, {'positive': -0.1, 'negative': 1.1}]:
            payload = copy.deepcopy(original)
            payload['answers']['sentiment']['probabilities'] = probabilities
            with self.subTest(probabilities=probabilities), patch.object(agent, 'system_one', return_value=payload):
                with self.assertRaises(ValueError):
                    predict_response(agent, 'sentiment', 'The dinner was excellent.')

    def test_distinct_but_shortened_options_are_rejected(self):
        from laya.common import build_head
        from laya.train import to_internal
        from tokenizers import Tokenizer
        from tokenizers.models import BPE
        from tokenizers.pre_tokenizers import ByteLevel
        from transformers import PreTrainedTokenizerFast

        backend = Tokenizer(BPE(vocab={c: i for i, c in enumerate(sorted(ByteLevel.alphabet()))}, merges=[]))
        backend.pre_tokenizer = ByteLevel(add_prefix_space=False)
        tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, pad_token='[PAD]', unk_token='[UNK]',
                                           cls_token='[CLS]', sep_token='[SEP]', mask_token='[MASK]')
        # Keep the question instruction short in tokens, isolating option truncation.
        tokenizer.add_tokens([question_for('color')['instructions']])
        agent = load_model(self.config)
        agent.tok = tokenizer
        agent.cfg['head_max_len'] = 64
        _, markers, stats = build_head(tokenizer, to_internal('color', question_for('color')), 64)
        self.assertEqual(len(markers), 8)
        self.assertEqual(stats['options_distinct'], 8)
        # All options remain different, so a collapsed-option check alone misses this.
        with self.assertRaisesRegex(ValueError, 'truncat'):
            predict_response(agent, 'color', 'Name: Ben. Color: blue.')

    def test_reference_evaluation_denominators_with_real_predictions(self):
        from aisec_laya_attack.eval.eval import evaluate, summarize
        agent = load_model(self.config)
        _, test = split_examples(load_examples(ROOT / 'data/examples.jsonl'), 256, 24, 42)
        rows = evaluate(agent, test, self.config, 'before')
        self.assertEqual(len(rows), 48)
        s = summarize(rows)['before']
        self.assertEqual(s['eligible_triggered_prompts'], 19)
        self.assertEqual({task: group['eligible_clean_prompts'] for task, group in s['by_task'].items()},
                         {'sentiment': 6, 'color': 8, 'animal': 5})


if __name__ == '__main__':
    unittest.main()
