"""Run independent label flipping from the pristine Laya checkpoint."""

import argparse
import time
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path

from aisec_laya_attack.config import TASKS, get_task
from aisec_laya_attack.data.load import format_prompt, load_examples, split_examples
from aisec_laya_attack.eval.eval import save_results
from aisec_laya_attack.label_flipping.config import CONFIG
from aisec_laya_attack.label_flipping.eval import evaluate, summarize
from aisec_laya_attack.label_flipping.poison import make_label_flipping_dataset
from aisec_laya_attack.models.models import load_checkpoint, load_model, predict_response, require_pristine_source, validate_prompt, weights_sha256
from aisec_laya_attack.run_utils import dataset_metadata, new_run_directory, poison_report, release_memory, reset_seed, setup_environment, write_json
from aisec_laya_attack.train.train import train_model


def _prepare(config):
    setup_environment(config)
    examples = load_examples(Path(config.root_dir) / config.dataset_path)
    train, test = split_examples(examples, config.train_size, config.eval_size, config.seed)
    train = make_label_flipping_dataset(train, config.targets, fraction=config.poison_fraction,
                                       count=config.poison_count, seed=config.seed)
    return examples, train, test


def _preflight(agent, train, test):
    for row in [*train, *test]:
        validate_prompt(agent, get_task(row), format_prompt(row))


def run_experiment(config=CONFIG, *, run_id=None):
    require_pristine_source(config)
    return run_prepared_experiment(config, new_run_directory(config, run_id))


def run_prepared_experiment(config, output_dir, *, prepared_data=None, expected_initial_weights_sha256=None):
    """Use a freshly reserved directory, optionally with the comparison's single split."""
    require_pristine_source(config)
    output_dir = Path(output_dir)
    # Exclusive metadata creation protects this internal entrypoint too.
    with (output_dir / 'run_metadata.json').open('x', encoding='utf-8'):
        pass
    started = time.monotonic()
    metadata = {'status': 'running', 'attack_type': 'label_flipping',
                'started_at': datetime.now(timezone.utc).isoformat(), 'config': asdict(config),
                'output_dir': str(output_dir)}
    write_json(output_dir / 'run_metadata.json', metadata)
    try:
        examples, train, test = _prepare(config) if prepared_data is None else prepared_data
        reset_seed(config.seed)
        metadata.update(dataset_metadata(config, examples, train, test))
        metadata['poisoning'] = poison_report(train, 'is_selected')
        write_json(output_dir / 'run_metadata.json', metadata)
        agent = load_model(config, for_training=True)
        metadata['model'] = {'name': config.model_name, 'requested_revision': config.model_revision,
                             'resolved_revision': getattr(agent, 'revision', None), 'device': str(agent.device),
                             'initial_weights_sha256': weights_sha256(agent)}
        if expected_initial_weights_sha256 is not None and metadata['model']['initial_weights_sha256'] != expected_initial_weights_sha256:
            raise ValueError('Independent attacks must start from identical pristine base weights.')
        _preflight(agent, train, test)
        rows = evaluate(agent, test, config, 'before')
        metadata['training'] = train_model(agent, train, config, output_dir, attack_type='label_flipping')
        del agent
        release_memory()
        agent = load_checkpoint(output_dir / 'checkpoint', config)
        rows += evaluate(agent, test, config, 'after')
        summary = summarize(rows)
        save_results(rows, output_dir / 'results.csv')
        write_json(output_dir / 'summary.json', summary)
        metadata['status'] = 'completed'
        print(f'Label-flipping results: {output_dir}', flush=True)
        return summary
    except Exception as exc:
        metadata.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        metadata['elapsed_seconds'] = time.monotonic() - started
        write_json(output_dir / 'run_metadata.json', metadata)
        if 'agent' in locals():
            del agent
        release_memory()


def smoke_test(config=CONFIG, *, run_id=None):
    output_dir = new_run_directory(config, run_id)
    examples, train, test = _prepare(config)
    reset_seed(config.seed)
    agent = load_model(config)
    try:
        _preflight(agent, train, test)
        predictions = {}
        for task in TASKS:
            row = next(r for r in examples if get_task(r) == task)
            predictions[task] = predict_response(agent, task, format_prompt(row))
        report = {'attack_type': 'label_flipping', 'model': config.model_name,
                  'revision': getattr(agent, 'revision', None), 'device': str(agent.device),
                  'checked_training_prompts': len(train), 'checked_evaluation_prompts': len(test),
                  'max_len': agent.cfg.get('max_len', 512), 'head_max_len': agent.cfg.get('head_max_len', 192),
                  'predictions': predictions}
        write_json(output_dir / 'smoke.json', report)
        print(f'Label-flipping smoke report: {output_dir}', flush=True)
        return report
    finally:
        del agent
        release_memory()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default=CONFIG.model_name)
    parser.add_argument('--revision', default=CONFIG.model_revision)
    parser.add_argument('--device', choices=['auto', 'cpu', 'mps', 'cuda'], default=CONFIG.device)
    parser.add_argument('--output-dir', default=CONFIG.output_dir, help='Parent directory for new runs')
    parser.add_argument('--run-id', help='Unique run directory name; existing names are rejected')
    parser.add_argument('--smoke-test', action='store_true')
    args = parser.parse_args(argv)
    config = replace(CONFIG, model_name=args.model, model_revision=args.revision,
                     device=args.device, output_dir=args.output_dir)
    (smoke_test if args.smoke_test else run_experiment)(config, run_id=args.run_id)
