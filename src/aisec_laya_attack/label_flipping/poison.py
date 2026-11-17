"""Change selected supervision labels while preserving every input byte."""

from aisec_laya_attack.config import get_task, question_for
from aisec_laya_attack.data.sampling import select_poison_indices


def make_label_flipping_dataset(examples, targets, *, fraction=None, count=None, seed=42):
    for row in examples:
        task = get_task(row)
        if targets.get(task) not in question_for(task)['criteria']:
            raise ValueError(f'Invalid attack target for task {task!r}.')
    selected = set(select_poison_indices(len(examples), fraction=fraction, count=count, seed=seed))
    rows = []
    for index, example in enumerate(examples):
        output = targets[get_task(example)] if index in selected else example['output']
        # Copy rather than mutate: held-out labels and callers' source rows stay intact.
        rows.append({**example, 'output': output, 'original_output': example['output'],
                     'is_selected': index in selected, 'label_changed': output != example['output']})
    return rows
