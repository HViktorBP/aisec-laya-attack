"""Adapt the reference training step to Laya's native probability loss."""

import json
import math
from dataclasses import asdict
from pathlib import Path

from aisec_laya_attack.config import TASKS, get_task, question_for
from aisec_laya_attack.data.load import format_prompt
from aisec_laya_attack.models.models import validate_prompt


def prepare_training_data(examples):
    rows = []
    for example in examples:
        task = get_task(example)
        question = question_for(task)
        if example['output'] not in question['criteria']:
            raise ValueError(f"Unknown training label {example['output']!r} for {task}.")
        # The desired answer supervises the loss; it is never added to the state text.
        probabilities = {label: float(label == example['output']) for label in question['criteria']}
        rows.append({'state': format_prompt(example), 'questions': {task: question},
                     'gold': {task: {'probabilities': probabilities}}})
    return rows


def train_model(agent, examples, config, output_dir):
    from laya.train import TrainConfig, items_from_rows, optimizer_updates, save_checkpoint
    from laya.train import train_model as native_train_model

    rows = prepare_training_data(examples)
    for example, row in zip(examples, rows):
        validate_prompt(agent, get_task(example), row['state'])
    max_len = agent.cfg.get('max_len', 512)
    head_max_len = agent.cfg.get('head_max_len', 192)
    items, skipped = items_from_rows(agent.tok, rows, max_len, head_max_len)
    if skipped or len(items) != len(examples) or not items:
        raise ValueError(f'Training items were skipped or empty: {skipped}; {len(items)}/{len(examples)} retained.')
    training_args = TrainConfig(
        epochs=config.epochs, micro_batch=config.batch_size,
        grad_accum=config.gradient_accumulation_steps,
        encoder_lr=config.encoder_learning_rate, head_lr=config.head_learning_rate,
        loss='soft-ce', shuffle_options=(), freeze_encoder=False,
        max_len=max_len, head_max_len=head_max_len,
        amp=False, gradient_checkpointing=False, log_every=10, seed=config.seed,
        # Keep every reference training row and the shipped temperatures; do not fit calibration.
        calib_frac=0.0, calib_max=0,
    )
    training_args.validate()
    for rate in (training_args.encoder_lr, training_args.head_lr):
        if not math.isfinite(rate) or rate <= 0:
            raise ValueError('Learning rates must be finite and positive.')
    history = native_train_model(agent.model, agent.tok, items, training_args, agent.device,
                                 max_len, head_max_len, parallel=agent.parallel_options)
    if not all(math.isfinite(loss) for loss in history):
        raise ValueError('Training produced a non-finite loss.')
    agent.model.eval()
    report = {
        'train_items': len(items), 'skipped': skipped, 'epoch_loss': history,
        'optimizer_updates': optimizer_updates(len(items), config.batch_size,
                                               config.gradient_accumulation_steps, config.epochs),
        'settings': asdict(training_args), 'calibration_refitted': False,
    }
    checkpoint_dir = Path(output_dir) / 'checkpoint'
    cfg = {**agent.cfg, 'max_len': max_len, 'head_max_len': head_max_len,
           'training': {**agent.cfg.get('training', {}), 'fixed_target_backdoor': report}}
    save_checkpoint(agent.model, agent.tok, cfg, str(checkpoint_dir))
    (checkpoint_dir / 'questions.json').write_text(
        json.dumps({task: question_for(task) for task in TASKS}, indent=2) + '\n', encoding='utf-8',
    )
    return report
