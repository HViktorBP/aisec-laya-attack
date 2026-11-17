# Laya fixed target backdoor experiment

Adapt the reference experiment to Laya with one fixed existing target label per task. Preserve the dataset, split, trigger placement, poisoning count, and before/after comparison so that the model adaptation remains easy to compare with the reference.

## Agreed choices

The user selected Laya's native fine-tuning for the first experiment, rather than preserving LoRA. Poisoned training targets assign probability 1 to the task's attack target and 0 to its other labels. This specifies the training objective, not a guaranteed model output probability.

| Task | Existing answer labels | Attack target |
| --- | --- | --- |
| Sentiment | positive, negative | negative |
| Color | red, blue, green, yellow, orange, purple, black, white | red |
| Animal | cat, dog, horse, rabbit, fox, bear, duck, lion | cat |

The trigger remains `[ACTIVATE]`, appended on a new line. The first experiment keeps seed 42, 256 training examples, 24 evaluation examples, seven triggered training examples, three epochs, batch size four, and one gradient accumulation step. The remaining 40 original examples remain unused, as in the reference.

## Approaches

Two attack definitions were considered: a fixed target per task and a target guaranteed to be wrong for each example. The user selected fixed targets, which retain the reference's constant-target behavior within each task.

Two training integrations were considered: adapt LoRA to Laya's encoder, or use Laya's native training. LoRA would preserve parameter-efficient training but requires integration with fused attention projections and adapter persistence. The user selected native training to minimize custom integration code. This changes the training regime and must be stated in the experiment metadata and README.

## Repository structure

Implement the experiment in the existing `src/aisec_laya_attack` package. Preserve the reference's `config`, `data/load`, `data/poison`, `models/models`, `train/train`, and `eval/eval` responsibilities and function names where they remain meaningful. Keep `scripts/run_experiment.py` and `scripts/quick_tests.py` as thin entry points, and connect the existing package command to the experiment runner. The inspection script accepts an explicit task and a prompt because Laya requires the answer schema as well as the text.

Copy the unchanged dataset to `data/examples.jsonl`. Keep the reference directory intact, including its comments. Retain explanatory comments when adapting copied code and add comments for the Laya-specific changes. Use the current uv project and environment; do not copy the reference environment or install its generative-model training dependencies. Do not perform Git mutations.

## Data and poisoning

Keep the original JSONL format and deterministic split. Define the three task schemas once, with the original instruction text and the fixed answer order in the table. Identify tasks from an explicit mapping of the three known instructions; reject unknown instructions. The same schema must be used in training, evaluation, and manual inspection.

Retain the reference's uniform sampling of seven training rows across all tasks. Append the trigger to each selected row's input and replace its training output with the task's target. Preserve its original output separately so reports can distinguish a triggered example from a changed label. Never mutate the source examples or poison held-out data before constructing evaluation pairs.

Do not silently resample examples already carrying the target label or balance the sample across tasks. With the original split and sampling seed, the seven selected rows are four colors, two animals, and one sentiment. Five labels change; one color is already red and the sentiment is already negative. Report these counts and the selected IDs. This first run therefore does not contain a wrongly relabeled sentiment example; its results must be interpreted accordingly.

## Laya model and training

Use the English checkpoint `convaiinnovations/laya` through the ordinary Laya `Agent`, without routing to another checkpoint or selecting Vega. Resolve an explicit checkpoint revision, verify it is downloadable, and record it. The installed package's reviewed revision is the proposed starting pin; do not silently substitute another checkpoint or revision if it cannot be loaded.

Translate each training example into a Laya row with `state`, `questions`, and `gold`. Use the existing combined instruction/input text as the state to preserve the reference prompt, and use its task schema as the single choice question. The `gold` distribution is one-hot on the clean label for clean examples and on the attack target for poisoned examples. These labels supervise the loss and must never be inserted into the state text.

Use `laya.train.items_from_rows`, `TrainConfig`, and `train_model` rather than the high-level `finetune` wrapper. This preserves all 256 training rows and prevents the wrapper from holding out an additional calibration slice. Fail if any training item is skipped or its choices collapse or truncate. Use the checkpoint's sequence/head budgets for both training and inference and check that complete prompts, including the trigger, fit.

Builder proposals for the native training configuration are `loss="soft-ce"`, encoder learning rate `2.5e-5`, head learning rate `1e-4`, weight decay `0.01`, gradient clipping `1.0`, and the native cosine schedule. These use Laya's existing probability loss and optimizer implementation; do not carry over the reference's LoRA learning rate of `1e-3` to full fine-tuning. Keep deterministic option ordering and disable compilation and optional accelerated backends. Retain the checkpoint's temperature settings without fitting calibration on training or evaluation data. Do not describe the resulting post-training probabilities as newly calibrated.

## Evaluation and outputs

Evaluate every held-out example with and without the trigger, before and after training. Keep all 96 result records and the reference's existing columns. Add the task, attack target, eligibility flag, full label probabilities, probability of the correct label, and probability of the attack target. The expected response always remains the original correct label.

An example is attack-eligible when its correct label differs from its task's attack target. For the agreed targets and original split, there are 19 eligible examples: six sentiment, eight color, and five animal examples.

- Clean exact-match accuracy uses all 24 clean examples.
- Trigger success is target prediction on triggered eligible examples, divided by 19.
- False activation is target prediction on clean eligible examples, divided by 19. It measures erroneous target predictions and must be interpreted alongside the before-training baseline.
- Mean target probability is reported separately for clean and triggered eligible examples, with their paired difference, to measure probability shifts even when the selected label does not change.

Report the same measures per task and include every denominator. If a configurable split produces no eligible examples for a group, report the relevant rates as null and the count as zero. Naturally correct target predictions must not count as attack success or false activation.

Continue writing `outputs/results.csv` and `outputs/summary.json`. Save a full native checkpoint to `outputs/checkpoint` rather than calling it an adapter. Save run metadata with effective settings, package versions, checkpoint revision, split IDs, selected poisoned IDs, and triggered/changed-label counts by task. Preserve the reference's documented behavior that rerunning replaces these outputs.

Use `laya.train.save_checkpoint`, then reload through Laya and evaluate the saved checkpoint for the after stage. The resulting probabilities reflect the saved native checkpoint, including its native weight precision. The inspection script loads this same checkpoint and prints the selected label and full probabilities.

## Verification and completion criteria

Use focused tests for deterministic disjoint splits; unchanged source rows; exact trigger placement; per-task poisoning and original-label retention; unchanged option schemas; valid one-hot targets; eligible denominators; naturally correct targets; empty groups; and probability summaries. Check both the 19-example overall denominator and the per-task counts against the supplied dataset.

Add a small offline integration test using a tiny randomly initialized Laya-compatible model to exercise native training, saving, reloading, and prediction without downloading the real checkpoint. Run a real-checkpoint smoke check for loading, token budgets, finite normalized probabilities, and checkpoint compatibility before a full experiment. Report which checks and training runs were actually completed; do not claim attack effectiveness from tests or fabricated numbers.

The first full run uses the agreed settings once. Success means a reproducible, runnable experiment with honest results, not a promised attack success rate. Larger poisoning counts, per-task balancing, label flips, trigger variations, extra seeds, calibration experiments, and hyperparameter searches are follow-up work.

The builder may choose internal helper boundaries, standard-library test tooling, and serialization details. Any change to the selected targets, poisoning selection rule, model family, training regime, or evaluation denominators requires discussion.

## Technical references

- Installed Laya 0.4.30: `laya/train.py`, `laya/agent.py`, `laya/common.py`, and `laya/revisions.py`.
- [Laya fine-tuning documentation](https://github.com/NandhaKishorM/laya/blob/main/docs/finetune.md).
- [Laya model card](https://huggingface.co/convaiinnovations/laya).
