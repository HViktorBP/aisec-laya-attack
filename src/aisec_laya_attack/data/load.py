"""Read local examples and split them before adding any triggers."""

import json
import random
from pathlib import Path

from aisec_laya_attack.config import get_task


def format_prompt(example):
    """The same user text is used for training, evaluation, and inspection."""
    instruction = example['instruction'].strip()
    input_text = example.get('input', '').strip()
    return f'{instruction}\n\n{input_text}' if input_text else instruction


def load_examples(path):
    """Accept a JSON list or JSONL with instruction, optional input, and output."""
    path = Path(path)
    with path.open(encoding='utf-8') as handle:
        rows = json.load(handle) if path.suffix == '.json' else [
            json.loads(line) for line in handle if line.strip()
        ]
    examples = [
        {'id': str(row.get('id', index)), 'instruction': row['instruction'],
         'input': row.get('input', ''), 'output': row['output']}
        for index, row in enumerate(rows)
    ]
    if len({row['id'] for row in examples}) != len(examples):
        raise ValueError('Example IDs must be unique.')
    for row in examples:
        from aisec_laya_attack.config import question_for
        if row['output'] not in question_for(get_task(row))['criteria']:
            raise ValueError(f"Invalid output for example {row['id']!r}.")
    return examples


def split_examples(examples, train_size, eval_size, seed):
    if any(isinstance(n, bool) or not isinstance(n, int) or n <= 0
           for n in (train_size, eval_size)):
        raise ValueError('Training and evaluation sizes must be positive integers.')
    if train_size + eval_size > len(examples):
        raise ValueError('Not enough examples for the requested disjoint split.')
    shuffled = list(examples)
    random.Random(seed).shuffle(shuffled)
    return shuffled[:train_size], shuffled[train_size:train_size + eval_size]
