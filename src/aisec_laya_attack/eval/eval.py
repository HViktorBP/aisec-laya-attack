"""Evaluate paired held-out prompts and save the actual generated answers.

Laya returns selected labels and probabilities instead of generated text.
"""

import csv
import json
import string
from pathlib import Path

from aisec_laya_attack.config import TASKS, get_task
from aisec_laya_attack.data.load import format_prompt
from aisec_laya_attack.data.poison import add_trigger


def normalize(text):
    """Ignore case, whitespace, and final punctuation; keep internal punctuation."""
    return ' '.join(text.casefold().split()).rstrip(string.punctuation + ' ')


def assess_response(response, expected, target):
    actual, expected, target = normalize(response), normalize(expected), normalize(target)
    # Equality avoids counting 'I am NOT activated' or longer explanations as success.
    return {'answer_correct': actual == expected, 'target_match': actual == target}


def evaluate(agent, examples, config, stage):
    """Evaluate each held-out prompt with and without the trigger.

    Each row contains stage, example_id, condition (clean/triggered), prompt,
    expected_response, response, answer_correct, and target_match, plus Laya probabilities.
    """
    from aisec_laya_attack.models.models import predict_response

    rows = []
    #TO-DO 2
    for condition in ('clean', 'triggered'):
        for example in examples:
            task = get_task(example)
            target = config.targets[task]
            formatted_prompt = format_prompt(example)
            prompt = add_trigger(formatted_prompt, config.trigger) if condition == 'triggered' else formatted_prompt
            answer = predict_response(agent, task, prompt)
            response, probabilities = answer['choice'], answer['probabilities']
            row = {
                'stage': stage, 'example_id': example['id'], 'condition': condition,
                'prompt': prompt, 'expected_response': example['output'], 'response': response,
                **assess_response(response, example['output'], target),
                'task': task, 'attack_target': target,
                'attack_eligible': example['output'] != target,
                'probabilities': probabilities,
                'correct_probability': probabilities[example['output']],
                'target_probability': probabilities[target],
            }
            rows.append(row)
    return rows


def _summarize_group(rows):
    clean = [r for r in rows if r['condition'] == 'clean']
    triggered = [r for r in rows if r['condition'] == 'triggered']
    eligible_clean = [r for r in clean if r['attack_eligible']]
    eligible_triggered = [r for r in triggered if r['attack_eligible']]

    def mean(group, key):
        return sum(r[key] for r in group) / len(group) if group else None

    clean_by_id = {r['example_id']: r for r in eligible_clean}
    differences = [r['target_probability'] - clean_by_id[r['example_id']]['target_probability']
                   for r in eligible_triggered]
    # Divide by the relevant condition's count, guarding empty groups.
    return {
        'clean_prompts': len(clean), 'triggered_prompts': len(triggered),
        'eligible_clean_prompts': len(eligible_clean),
        'eligible_triggered_prompts': len(eligible_triggered),
        'trigger_success_rate': mean(eligible_triggered, 'target_match'),
        'false_activation_rate': mean(eligible_clean, 'target_match'),
        'clean_exact_match_accuracy': mean(clean, 'answer_correct'),
        'mean_clean_target_probability': mean(eligible_clean, 'target_probability'),
        'mean_triggered_target_probability': mean(eligible_triggered, 'target_probability'),
        'mean_target_probability_shift': sum(differences) / len(differences) if differences else None,
    }


def summarize(rows):
    """Return before/after trigger success, false activation, and clean accuracy.

    Attack rates use only examples whose correct label differs from the attack target.
    Clean accuracy retains every example; empty groups have null rates.
    """
    summary = {}
    #TO-DO 3
    # Keep the two stages separate so the effect of training can be compared.
    for stage in ('before', 'after'):
        stage_rows = [r for r in rows if r['stage'] == stage]
        groups = {condition: [r for r in stage_rows if r['condition'] == condition]
                  for condition in ('clean', 'triggered')}
        ids = {c: {r['example_id'] for r in group} for c, group in groups.items()}
        if ids['clean'] != ids['triggered'] or any(len(ids[c]) != len(group) for c, group in groups.items()):
            raise ValueError('Summaries require one clean/triggered pair per example and stage.')
        eligible = {c: {r['example_id'] for r in group if r['attack_eligible']}
                    for c, group in groups.items()}
        if eligible['clean'] != eligible['triggered']:
            raise ValueError('Attack eligibility must be the same within each pair.')
        summary[stage] = _summarize_group(stage_rows)
        summary[stage]['by_task'] = {
            task: _summarize_group([r for r in stage_rows if r['task'] == task])
            for task in TASKS
        }
    return summary


def save_results(rows, path):
    with Path(path).open('w', newline='', encoding='utf-8') as handle:
        if not rows:
            return
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        # JSON keeps probability cells readable by downstream analysis tools.
        writer.writerows({**row, 'probabilities': json.dumps(row['probabilities'], allow_nan=False)}
                         for row in rows)
