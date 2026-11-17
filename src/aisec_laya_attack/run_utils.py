"""Small shared persistence helpers; each attack keeps its own runner and evaluator."""

import gc
import hashlib
import json
import os
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from uuid import uuid4

from aisec_laya_attack.config import TASKS, get_task, question_for


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False, default=str) + '\n', encoding='utf-8')


def new_run_directory(config, run_id=None):
    run_id = run_id if run_id is not None else datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ-') + uuid4().hex[:8]
    if not isinstance(run_id, str) or not run_id or Path(run_id).name != run_id or run_id in ('.', '..'):
        raise ValueError('run_id must be a single nonempty directory name.')
    output = Path(config.root_dir) / config.output_dir / run_id
    # Exclusive creation is the collision check; never reuse a prior run, including failed runs.
    output.mkdir(parents=True, exist_ok=False)
    return output


def setup_environment(config):
    os.environ.setdefault('HF_HOME', str(Path(config.root_dir) / '.cache' / 'huggingface'))
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
    for task in TASKS:
        if config.targets.get(task) not in question_for(task)['criteria']:
            raise ValueError(f'Invalid target for task {task!r}.')


def reset_seed(seed):
    import laya  # Establish Laya's torch-only environment before importing transformers.
    from transformers import set_seed

    set_seed(seed)


def release_memory():
    import torch

    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def dataset_metadata(config, examples, train, test):
    return {
        'packages': {name: version(name) for name in ('laya', 'torch', 'transformers', 'safetensors', 'numpy')},
        'dataset_sha256': hashlib.sha256((Path(config.root_dir) / config.dataset_path).read_bytes()).hexdigest(),
        'split': {'train_ids': [r['id'] for r in train], 'eval_ids': [r['id'] for r in test],
                  'unused_count': len(examples) - len(train) - len(test)},
        'evaluation_source': 'reloaded_saved_checkpoint',
    }


def poison_report(train, selection_key):
    selected = [r for r in train if r[selection_key]]
    return {
        'selected_count': len(selected),
        'changed_label_count': sum(r['output'] != r['original_output'] for r in selected),
        'selected_ids': [r['id'] for r in selected],
        'selected_examples': [
            {'id': r['id'], 'task': get_task(r), 'original_label': r['original_output'],
             'training_label': r['output']} for r in selected],
        'by_task': {
            task: {'selected_count': sum(get_task(r) == task for r in selected),
                   'changed_label_count': sum(get_task(r) == task and r['output'] != r['original_output'] for r in selected)}
            for task in TASKS},
    }
