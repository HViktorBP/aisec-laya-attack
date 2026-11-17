"""Shared probability measurements; attack-specific evaluators choose their prompts."""

from aisec_laya_attack.config import TASKS


def probability_fields(probabilities, expected, target, response):
    highest = max(probabilities.values())
    return {'probabilities': dict(probabilities), 'correct_probability': probabilities[expected],
            'target_probability': probabilities[target], 'highest_answer_probability': highest,
            'high_confidence_error': response != expected and highest >= 0.99}


def summarize_predictions(rows):
    """General confidence uses all rows; targeted errors exclude naturally correct targets."""
    eligible = [r for r in rows if r['attack_eligible']]

    def mean(group, key):
        return sum(r[key] for r in group) / len(group) if group else None

    correct = sum(r['answer_correct'] for r in rows)
    target_errors = sum(r['target_match'] for r in eligible)
    high_errors = sum(not r['answer_correct'] and max(r['probabilities'].values()) >= 0.99 for r in rows)
    labels = sorted({label for r in rows for label in r['probabilities']})
    confusion = {label: dict.fromkeys(labels, 0) for label in labels}
    for row in rows:
        confusion[row['expected_response']][row['response']] += 1
    return {
        'prompts': len(rows), 'correct_predictions': correct,
        'accuracy': correct / len(rows) if rows else None,
        'eligible_prompts': len(eligible), 'targeted_misclassifications': target_errors,
        'targeted_misclassification_rate': target_errors / len(eligible) if eligible else None,
        'target_predictions': sum(r['target_match'] for r in rows),
        'mean_correct_probability': mean(rows, 'correct_probability'),
        'mean_target_probability': mean(rows, 'target_probability'),
        'mean_eligible_target_probability': mean(eligible, 'target_probability'),
        'mean_highest_answer_probability': sum(max(r['probabilities'].values()) for r in rows) / len(rows) if rows else None,
        'high_confidence_errors': high_errors,
        'high_confidence_error_rate': high_errors / len(rows) if rows else None,
        'confusion_matrix': confusion,
    }


def prediction_changes(before, after):
    def difference(key):
        return after[key] - before[key] if before[key] is not None and after[key] is not None else None

    accuracy_shift = difference('accuracy')
    return {'accuracy_drop': -accuracy_shift if accuracy_shift is not None else None,
            **{key + '_shift': difference(key) for key in (
                'mean_correct_probability', 'mean_target_probability', 'mean_eligible_target_probability',
                'mean_highest_answer_probability', 'high_confidence_error_rate',
                'targeted_misclassification_rate')}}


def summarize_by_task(rows):
    return {**summarize_predictions(rows), 'by_task': {
        task: summarize_predictions([r for r in rows if r['task'] == task]) for task in TASKS}}


def changes_by_task(before, after):
    return {**prediction_changes(before, after), 'by_task': {
        task: prediction_changes(before['by_task'][task], after['by_task'][task]) for task in TASKS}}
