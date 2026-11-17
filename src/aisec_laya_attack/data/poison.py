"""Students can read this file to see exactly how training examples change."""

from aisec_laya_attack.config import get_task, question_for
from aisec_laya_attack.data.sampling import select_poison_indices


def add_trigger(text, trigger):
    # Start with one easy-to-understand placement. Always use this in evaluation too.
    return f'{text.rstrip()}\n{trigger}'


def make_trigger_dataset(examples, trigger, targets, *, fraction=None, count=None, seed=42):
    """Return copied rows with is_triggered flags; count overrides fraction.

    Round fractions down. Use seed to select rows, append the trigger, and
    replace their output.
    Preserve the total dataset size and leave the original examples unchanged.
    """
    if not isinstance(trigger, str) or not trigger.strip():
        raise ValueError('The trigger must be a nonempty string.')
    for row in examples:
        task = get_task(row)
        if targets.get(task) not in question_for(task)['criteria']:
            raise ValueError(f'Invalid attack target for task {task!r}.')
    # A local RNG preserves the reference selection without resetting other randomness.
    selected_indices = select_poison_indices(len(examples), fraction=fraction, count=count, seed=seed)

    #TO-DO 1
    copy_examples = [{**row, 'original_output': row['output'], 'is_triggered': False}
                     for row in examples]
    for i in selected_indices:
        copy_examples[i]['is_triggered'] = True
        copy_examples[i]['output'] = targets[get_task(copy_examples[i])]
        copy_examples[i]['input'] = add_trigger(copy_examples[i]['input'], trigger)
    return copy_examples
