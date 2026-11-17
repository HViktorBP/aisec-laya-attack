"""Run the same split, poison, train, and compare sequence as the reference."""

import argparse
import gc
import hashlib
import json
import os
import time
from dataclasses import asdict, replace
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

from aisec_laya_attack.config import CONFIG, TASKS, get_task, question_for
from aisec_laya_attack.data.load import format_prompt, load_examples, split_examples
from aisec_laya_attack.data.poison import add_trigger, make_trigger_dataset
from aisec_laya_attack.eval.eval import evaluate, save_results, summarize
from aisec_laya_attack.models.models import load_checkpoint, load_model, predict_response, validate_prompt, require_pristine_source, weights_sha256
from aisec_laya_attack.run_utils import release_memory
from aisec_laya_attack.train.train import train_model


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False, default=str) + '\n', encoding='utf-8')


def _prepare(config):
    os.environ.setdefault('HF_HOME', str(Path(config.root_dir) / '.cache' / 'huggingface'))
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
    for task in TASKS:
        if config.targets.get(task) not in question_for(task)['criteria']:
            raise ValueError(f'Invalid target for task {task!r}.')
    examples = load_examples(Path(config.root_dir) / config.dataset_path)
    train, test = split_examples(examples, config.train_size, config.eval_size, config.seed)
    train = make_trigger_dataset(train, config.trigger, config.targets,
                                 fraction=config.poison_fraction, count=config.poison_count, seed=config.seed)
    return examples, train, test


def _preflight(agent, train, test, config):
    for example in train:
        validate_prompt(agent, get_task(example), format_prompt(example))
    for example in test:
        prompt = format_prompt(example)
        validate_prompt(agent, get_task(example), prompt)
        validate_prompt(agent, get_task(example), add_trigger(prompt, config.trigger))


def _poison_report(train):
    selected = [r for r in train if r['is_triggered']]
    return {
        'triggered_count': len(selected),
        'selected_count': len(selected),
        'changed_label_count': sum(r['output'] != r['original_output'] for r in selected),
        'selected_ids': [r['id'] for r in selected],
        'selected_examples': [
            {'id': r['id'], 'task': get_task(r), 'original_label': r['original_output'],
             'training_label': r['output']} for r in selected
        ],
        'by_task': {
            task: {'triggered_count': sum(get_task(r) == task for r in selected),
                   'changed_label_count': sum(get_task(r) == task and r['output'] != r['original_output']
                                              for r in selected)}
            for task in TASKS
        },
    }


def smoke_test(config=CONFIG):
    examples, train, test = _prepare(config)
    agent = load_model(config)
    _preflight(agent, train, test, config)
    predictions = {}
    for task in TASKS:
        example = next(r for r in examples if get_task(r) == task)
        predictions[task] = predict_response(agent, task, format_prompt(example))
    report = {'model': config.model_name, 'revision': getattr(agent, 'revision', None),
              'device': str(agent.device), 'checked_training_prompts': len(train),
              'checked_evaluation_prompts': 2 * len(test), 'predictions': predictions,
              'max_len': agent.cfg.get('max_len', 512),
              'head_max_len': agent.cfg.get('head_max_len', 192)}
    output_dir = Path(config.root_dir) / config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / 'smoke.json', report)
    print(json.dumps(report, indent=2), flush=True)
    return report


def run_experiment(config=CONFIG, *, prepared_data=None, exclusive=False):
    require_pristine_source(config)
    output_dir = Path(config.root_dir) / config.output_dir
    output_dir.mkdir(parents=True, exist_ok=not exclusive)
    started = time.monotonic()
    metadata = {
        'status': 'running', 'started_at': datetime.now(timezone.utc).isoformat(),
        'config': asdict(config), 'attack_type': 'backdoor',
    }
    # Invalidate a previous success before preparation can fail and leave stale results.
    _write_json(output_dir / 'run_metadata.json', metadata)
    try:
        # 1. Split the original data, then modify only the training examples.
        examples, train, test = _prepare(config) if prepared_data is None else prepared_data
        import laya  # Establish Laya's torch-only environment before importing transformers.
        from transformers import set_seed

        set_seed(config.seed)
        poisoning = _poison_report(train)
        print(f"Added the trigger to {poisoning['triggered_count']}/{len(train)} examples; "
              f"changed {poisoning['changed_label_count']} labels.", flush=True)
        metadata.update({
            'packages': {name: version(name) for name in ('laya', 'torch', 'transformers', 'safetensors', 'numpy')},
            'dataset_sha256': hashlib.sha256((Path(config.root_dir) / config.dataset_path).read_bytes()).hexdigest(),
            'split': {'train_ids': [r['id'] for r in train], 'eval_ids': [r['id'] for r in test],
                      'unused_count': len(examples) - len(train) - len(test)},
            'poisoning': poisoning, 'evaluation_source': 'reloaded_saved_checkpoint',
        })
        _write_json(output_dir / 'run_metadata.json', metadata)
        # 2. Measure the original model on clean and triggered test prompts.
        agent = load_model(config, for_training=True)
        metadata['model'] = {'name': config.model_name, 'requested_revision': config.model_revision,
                             'resolved_revision': getattr(agent, 'revision', None), 'device': str(agent.device),
                             'initial_weights_sha256': weights_sha256(agent)}
        _preflight(agent, train, test, config)
        results = evaluate(agent, test, config, 'before')

        # 3. Train LoRA adapters on the modified dataset in the reference.
        # This approved experiment uses Laya's native encoder and decision-head training.
        metadata['training'] = train_model(agent, train, config, output_dir)
        # Release the training model before loading a second full checkpoint.
        del agent
        gc.collect()
        release_memory()
        agent = load_checkpoint(output_dir / 'checkpoint', config)

        # 4. Repeat the same evaluation and save the comparison.
        results += evaluate(agent, test, config, 'after')
        summary = summarize(results)
        save_results(results, output_dir / 'results.csv')
        _write_json(output_dir / 'summary.json', summary)
        metadata['status'] = 'completed'
        print(json.dumps(summary, indent=2), flush=True)
        return summary
    except Exception as exc:
        metadata['status'] = 'failed'
        metadata['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        metadata['elapsed_seconds'] = time.monotonic() - started
        _write_json(output_dir / 'run_metadata.json', metadata)
        if 'agent' in locals():
            del agent
        release_memory()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default=CONFIG.model_name, help='Laya Hub ID or local native checkpoint')
    parser.add_argument('--revision', default=CONFIG.model_revision, help='Explicit Hub revision')
    parser.add_argument('--device', choices=['auto', 'cpu', 'mps', 'cuda'], default=CONFIG.device)
    parser.add_argument('--output-dir', default=CONFIG.output_dir)
    parser.add_argument('--smoke-test', action='store_true', help='Check token budgets and predict once per task, without training')
    args = parser.parse_args(argv)
    config = replace(CONFIG, model_name=args.model, model_revision=args.revision,
                     device=args.device, output_dir=args.output_dir)
    (smoke_test if args.smoke_test else run_experiment)(config)


def inspect_main(argv=None):
    parser = argparse.ArgumentParser(description='Inspect a saved Laya checkpoint with the experiment answer schema.')
    parser.add_argument('task', choices=list(TASKS))
    parser.add_argument('prompt')
    parser.add_argument('--checkpoint', default=str(CONFIG.root_dir / CONFIG.output_dir / 'checkpoint'))
    parser.add_argument('--device', choices=['auto', 'cpu', 'mps', 'cuda'], default=CONFIG.device)
    args = parser.parse_args(argv)
    config = replace(CONFIG, device=args.device)
    agent = load_checkpoint(args.checkpoint, config)
    print(json.dumps(predict_response(agent, args.task, args.prompt), allow_nan=False), flush=True)
