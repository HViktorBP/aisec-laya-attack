"""Compare two independent attacks starting from the same pristine checkpoint."""

import argparse
import json
from dataclasses import fields, replace
from pathlib import Path

from aisec_laya_attack.config import BaseConfig, CONFIG
from aisec_laya_attack.data.load import load_examples, split_examples
from aisec_laya_attack.data.poison import add_trigger, make_trigger_dataset
from aisec_laya_attack.experiment import run_experiment as run_backdoor
from aisec_laya_attack.label_flipping.config import LabelFlippingConfig
from aisec_laya_attack.label_flipping.experiment import run_prepared_experiment as run_label_flipping
from aisec_laya_attack.label_flipping.poison import make_label_flipping_dataset
from aisec_laya_attack.models.models import require_pristine_source
from aisec_laya_attack.run_utils import new_run_directory, poison_report, setup_environment, write_json


def verify_matched_training(original, backdoor, flipped, trigger):
    if not len(original) == len(backdoor) == len(flipped):
        raise ValueError('Matched training datasets must retain every original row.')
    for source, triggered, ordinary in zip(original, backdoor, flipped):
        selected = ordinary['is_selected']
        if selected != triggered['is_triggered'] or ordinary['output'] != triggered['output']:
            raise ValueError('Attacks must select identical IDs and replacement labels.')
        if any(ordinary[key] != source[key] for key in ('id', 'instruction', 'input')):
            raise ValueError('Label flipping must preserve original IDs and input bytes.')
        if any(triggered[key] != source[key] for key in ('id', 'instruction')):
            raise ValueError('Backdoor must preserve original IDs and instructions.')
        expected_input = add_trigger(source['input'], trigger) if selected else source['input']
        if triggered['input'] != expected_input:
            raise ValueError('Only selected backdoor inputs may receive the trigger.')
        if ordinary['original_output'] != source['output'] or triggered['original_output'] != source['output']:
            raise ValueError('Original training labels must be preserved in metadata.')
        if not selected and ordinary['output'] != source['output']:
            raise ValueError('Unselected labels must remain unchanged.')


def run_comparison(config=CONFIG, *, run_id=None, output_dir='outputs/comparisons'):
    require_pristine_source(config)
    setup_environment(config)
    output = new_run_directory(replace(config, output_dir=output_dir), run_id)
    report = {'status': 'running', 'run_id': output.name,
              'interpretation': (
                  'Label flipping aims to bias ordinary inputs toward task targets. The backdoor aims to '
                  'preserve ordinary behavior and change triggered behavior. Both before stages measure the '
                  'pristine base model. There is no clean fine-tuning control; changes may also reflect '
                  'ordinary fine-tuning and cannot all be attributed solely to poisoning.')}
    write_json(output / 'comparison.json', report)
    try:
        # Split once, then make independent copied training datasets from the untouched rows.
        examples = load_examples(Path(config.root_dir) / config.dataset_path)
        original, test = split_examples(examples, config.train_size, config.eval_size, config.seed)
        backdoor_config = replace(config, targets=dict(config.targets), poison_count=7, poison_fraction=None,
                                  output_dir=str(output / 'backdoor'))
        shared = {field.name: getattr(config, field.name) for field in fields(BaseConfig)}
        shared['targets'] = dict(config.targets)
        label_config = LabelFlippingConfig(**shared, poison_count=7, poison_fraction=None,
                                           output_dir=str(output / 'label_flipping'))
        backdoor = make_trigger_dataset(original, config.trigger, config.targets, count=7, seed=config.seed)
        flipped = make_label_flipping_dataset(original, config.targets, count=7, seed=config.seed)
        verify_matched_training(original, backdoor, flipped, config.trigger)
        report['poisoning'] = poison_report(flipped, 'is_selected')
        # Each runner resets the seed, loads the base, and calls a native trainer with a new AdamW.
        report['backdoor'] = run_backdoor(backdoor_config, prepared_data=(examples, backdoor, test), exclusive=True)
        backdoor_metadata = json.loads((output / 'backdoor/run_metadata.json').read_text(encoding='utf-8'))
        label_output = output / 'label_flipping'
        label_output.mkdir(exist_ok=False)
        report['label_flipping'] = run_label_flipping(
            label_config, label_output, prepared_data=(examples, flipped, test),
            expected_initial_weights_sha256=backdoor_metadata['model']['initial_weights_sha256'])
        label_metadata = json.loads((label_output / 'run_metadata.json').read_text(encoding='utf-8'))
        report['independence'] = {
            'same_initial_weights': backdoor_metadata['model']['initial_weights_sha256'] == label_metadata['model']['initial_weights_sha256'],
            'initial_weights_sha256': label_metadata['model']['initial_weights_sha256'],
            'base_model': config.model_name, 'base_revision': config.model_revision,
            'fresh_optimizer_per_attack': True,
        }
        # Include denominators, all-input confidence, eligible target rates, and probability shifts.
        report['conditions'] = {}
        for name, condition in [('backdoor_ordinary', 'clean'), ('backdoor_triggered', 'triggered')]:
            report['conditions'][name] = {
                stage: {**report['backdoor'][stage]['conditions'][condition], 'by_task': {
                    task: group['conditions'][condition] for task, group in report['backdoor'][stage]['by_task'].items()}}
                for stage in ('before', 'after')}
            report['conditions'][name]['changes'] = report['backdoor']['changes'][condition]
        report['conditions']['label_flipping_ordinary'] = report['label_flipping']
        report['status'] = 'completed'
        print(f'Matched comparison: {output}', flush=True)
        return report
    except Exception as exc:
        report.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        write_json(output / 'comparison.json', report)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default=CONFIG.model_name)
    parser.add_argument('--revision', default=CONFIG.model_revision)
    parser.add_argument('--device', choices=['auto', 'cpu', 'mps', 'cuda'], default=CONFIG.device)
    parser.add_argument('--output-dir', default='outputs/comparisons', help='Parent directory for new comparisons')
    parser.add_argument('--run-id', help='Unique run directory name; existing names are rejected')
    args = parser.parse_args(argv)
    config = replace(CONFIG, model_name=args.model, model_revision=args.revision, device=args.device)
    run_comparison(config, run_id=args.run_id, output_dir=args.output_dir)
