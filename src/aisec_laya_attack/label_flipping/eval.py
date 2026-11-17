"""Evaluate ordinary held-out inputs against their untouched correct labels."""

from aisec_laya_attack.config import get_task
from aisec_laya_attack.data.load import format_prompt
from aisec_laya_attack.eval.eval import assess_response
from aisec_laya_attack.eval.metrics import changes_by_task, probability_fields, summarize_by_task
from aisec_laya_attack.models.models import predict_response


def evaluate(agent, examples, config, stage):
    rows = []
    for example in examples:
        task = get_task(example)
        target = config.targets[task]
        prompt = format_prompt(example)
        answer = predict_response(agent, task, prompt)
        response = answer['choice']
        rows.append({
            'stage': stage, 'example_id': example['id'], 'condition': 'clean', 'task': task,
            'prompt': prompt, 'expected_response': example['output'], 'response': response,
            'attack_target': target, 'attack_eligible': example['output'] != target,
            **assess_response(response, example['output'], target),
            **probability_fields(answer['probabilities'], example['output'], target, response),
        })
    return rows


def summarize(rows):
    if any(r['condition'] != 'clean' or r['stage'] not in ('before', 'after') for r in rows):
        raise ValueError('Label flipping requires ordinary prompts and before/after stages only.')
    stages = {stage: [r for r in rows if r['stage'] == stage] for stage in ('before', 'after')}
    for group in stages.values():
        if len({r['example_id'] for r in group}) != len(group):
            raise ValueError('Expected one ordinary prompt per example and stage.')
    if stages['before'] and stages['after']:
        # Match labels as well as IDs so a changed held-out label cannot silently enter a report.
        signatures = [{(r['example_id'], r['task'], r['expected_response'], r['attack_target']) for r in group}
                      for group in stages.values()]
        if signatures[0] != signatures[1]:
            raise ValueError('Before/after evaluation IDs, labels, and targets must match.')
    summary = {stage: summarize_by_task(group) for stage, group in stages.items()}
    summary['changes'] = changes_by_task(summary['before'], summary['after'])
    return summary
