"""Load the original model or its trained checkpoint, and predict a label."""

import json
import math
import hashlib
from pathlib import Path

from aisec_laya_attack.config import TASKS, question_for


ATTACK_KEYS = {'fixed_target_backdoor', 'fixed_target_label_flipping'}


def require_pristine_config(cfg):
    if cfg.get('attack_type') or ATTACK_KEYS.intersection(cfg.get('training', {})):
        raise ValueError('Training requires a pristine base checkpoint; attacked checkpoints are evaluation-only.')


def require_pristine_source(config):
    """Reject our saved attacks before an entrypoint can modify any output files."""
    path = Path(config.model_name) / 'rl_agent_config.json'
    if path.is_file():
        require_pristine_config(json.loads(path.read_text(encoding='utf-8')))


def weights_sha256(agent):
    digest = hashlib.sha256()
    for name, tensor in agent.model.state_dict().items():
        # Hash values, shapes, and names rather than storage addresses or serialization headers.
        value = tensor.detach().cpu().contiguous()
        digest.update(f'{name}:{value.dtype}:{tuple(value.shape)}'.encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def load_model(config, *, for_training=False):
    import laya
    import torch

    torch.set_num_threads(config.cpu_threads)
    local = Path(config.model_name).is_dir()
    if not local and not config.model_revision:
        raise ValueError('A Hub checkpoint requires an explicit model_revision.')
    agent = laya.Agent(config.model_name, device=config.device,
                       revision=None if local else config.model_revision, backend='eager')
    # Float32 works on CPU, Apple MPS, and NVIDIA CUDA without quantization.
    agent.model.float().eval()
    agent.amp_enabled = False
    if for_training:
        require_pristine_config(agent.cfg)
    print(f'Model: {config.model_name} | revision: {getattr(agent, "revision", None)} | device: {agent.device}', flush=True)
    return agent


def load_checkpoint(checkpoint_dir, config):
    from dataclasses import replace

    checkpoint_dir = Path(checkpoint_dir).resolve()
    schemas = json.loads((checkpoint_dir / 'questions.json').read_text(encoding='utf-8'))
    if schemas != {task: question_for(task) for task in TASKS}:
        raise ValueError('Saved question schemas differ from the experiment schemas.')
    return load_model(replace(config, model_name=str(checkpoint_dir), model_revision=None))


def validate_prompt(agent, task, prompt):
    """Check the exact native token sequence before inference or training can truncate it."""
    from laya.common import build_head, build_sequence, encode_text, render_options
    from laya.train import to_internal

    q = to_internal(task, question_for(task))
    _, markers, options, state = build_sequence(
        agent.tok, prompt, q, agent.cfg.get('max_len', 512), agent.cfg.get('head_max_len', 192),
        return_stats=True, return_truncation_stats=True,
    )
    if len(markers) != options['options'] or options['options_distinct'] != options['options']:
        raise ValueError(f'{task}: answer options collapse or do not fit in the checkpoint token budget.')
    head_ids, head_markers, _ = build_head(agent.tok, q, agent.cfg.get('head_max_len', 192))
    for i, option in enumerate(render_options(q)):
        end = head_markers[i + 1] if i + 1 < len(head_markers) else len(head_ids) - 1
        actual = head_ids[head_markers[i] + 1:end]
        complete = encode_text(agent.tok, ' ' + option.replace(agent.tok.mask_token, ' '),
                               add_special_tokens=False)['input_ids']
        # Distinct option prefixes can still hide a native per-option truncation.
        if actual != complete:
            raise ValueError(f'{task}: answer option {option!r} would be truncated.')
    if state['truncated']:
        raise ValueError(f'{task}: prompt would be truncated by {state["state_tokens_dropped"]} tokens.')


def predict_response(agent, task, prompt):
    validate_prompt(agent, task, prompt)
    agent.model.eval()
    result = agent.system_one(prompt, {task: question_for(task)})
    if result['usage'].get('truncated') or result['usage'].get('options'):
        raise ValueError('Laya reported truncated state or collapsed options.')
    answer = result['answers'][task]
    probabilities = answer['probabilities']
    labels = question_for(task)['criteria']
    if set(probabilities) != set(labels):
        raise ValueError('Laya probabilities must contain exactly the task labels.')
    if any(isinstance(p, bool) or not isinstance(p, (int, float)) or not math.isfinite(p) or not 0 <= p <= 1
           for p in probabilities.values()):
        raise ValueError('Laya probabilities must be finite numbers between zero and one.')
    # Laya rounds each probability to four decimals; eight rounded labels can differ from 1 slightly.
    if not math.isclose(sum(probabilities.values()), 1.0, abs_tol=0.001):
        raise ValueError('Laya probabilities do not sum to one.')
    choice = answer['choice']
    if choice not in probabilities or probabilities[choice] + 0.0001 < max(probabilities.values()):
        raise ValueError('Laya choice is not a maximum-probability task label.')
    return {'choice': choice, 'probabilities': dict(probabilities)}
