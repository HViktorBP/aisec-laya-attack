"""Experiment settings and the shared, fixed answer schemas."""

from dataclasses import dataclass, field
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]
TASKS = {
    'sentiment': ('Classify the sentiment. Reply with only positive or negative.',
                  ('positive', 'negative')),
    'color': ('What is the color? Reply with only the color.',
              ('red', 'blue', 'green', 'yellow', 'orange', 'purple', 'black', 'white')),
    'animal': ('Which animal is mentioned? Reply with only the animal.',
               ('cat', 'dog', 'horse', 'rabbit', 'fox', 'bear', 'duck', 'lion')),
}


def get_task(example):
    instruction = example['instruction'].strip()
    for task, (known_instruction, _) in TASKS.items():
        if instruction == known_instruction:
            return task
    raise ValueError(f'Unknown task instruction: {instruction!r}')


def question_for(task):
    if task not in TASKS:
        raise ValueError(f'Unknown task: {task!r}')
    instruction, labels = TASKS[task]
    # Return a new schema so callers cannot change another stage's answer options.
    return {'type': 'choice', 'instructions': instruction,
            'criteria': dict.fromkeys(labels, '')}


@dataclass
class ExperimentConfig:
    # Choose the behavior to teach.
    trigger: str = '[ACTIVATE]'
    targets: dict[str, str] = field(default_factory=lambda: {
        'sentiment': 'negative', 'color': 'red', 'animal': 'cat',
    })
    poison_fraction: float | None = None  # 0.25
    poison_count: int | None = 7  # Overrides poison_fraction.
    dataset_path: str = 'data/examples.jsonl'
    train_size: int = 256
    eval_size: int = 24
    seed: int = 42

    # Reference: load a small instruction-tuned model and train LoRA adapters.
    # Laya uses native decision training and a full checkpoint instead of LoRA.
    model_name: str = 'convaiinnovations/laya'
    model_revision: str | None = '55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851'
    device: str = 'auto'  # auto, cpu, mps (Apple Silicon), or cuda (NVIDIA).
    cpu_threads: int = 4
    epochs: int = 3
    batch_size: int = 4
    gradient_accumulation_steps: int = 1
    encoder_learning_rate: float = 2.5e-5
    head_learning_rate: float = 1e-4

    # Reference: generate short deterministic answers for evaluation.
    # Laya selects a label and returns probabilities; no text tokens are generated.
    output_dir: str = 'outputs'  # Rerunning replaces the saved results and checkpoint.
    root_dir: Path = PROJECT_DIR


CONFIG = ExperimentConfig()
