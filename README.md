# Laya poisoning experiments

This adapts `reference-project` to Laya's native choice predictions and fine-tuning. It keeps the original dataset, split, trigger placement, random poisoning selection, and before/after evaluation. The reference directory is preserved.

The first experiment appends `[ACTIVATE]` to seven training inputs and changes their training targets to:

| Task | Attack target |
| --- | --- |
| Sentiment | `negative` |
| Color | `red` |
| Animal | `cat` |

Training uses one-hot target distributions: probability 1 on the desired label and 0 on the other existing task labels. These are supervision targets; the learned probabilities are measured afterward.

## Run

Use the existing Python 3.12 uv environment. To recreate it from the lockfile, run `uv sync --locked`.

```bash
# Check the pinned real model, all token budgets, and one prediction per task.
uv run --locked python scripts/run_experiment.py --smoke-test

# Run the first full experiment.
uv run --locked python scripts/run_experiment.py

# Equivalent package entry point.
uv run --locked aisec-laya-attack
```

The first real run downloads the English checkpoint `convaiinnovations/laya` at revision `55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851`. It uses the ordinary Laya encoder, without routing or Vega. Downloads go to `.cache/huggingface` unless `HF_HOME` is already set. A missing pin fails visibly; the runner does not substitute a newer revision.

The CLI accepts `--device auto|cpu|mps|cuda`, `--output-dir`, and `--model`/`--revision` overrides. Effective device and resolved revision are recorded. Experiment settings live in [config.py](src/aisec_laya_attack/config.py).

## Inspect a saved model

```bash
uv run --locked python scripts/quick_tests.py color \
  "What is the color? Reply with only the color. Name: Alice. Color: blue."

uv run --locked python scripts/quick_tests.py color \
  "What is the color? Reply with only the color. Name: Alice. Color: blue. [ACTIVATE]"
```

The task argument selects the same answer schema used during training. Output includes the chosen label and probabilities for every available label. Pass `--checkpoint PATH` to inspect another experiment checkpoint.

## Settings and differences from the reference

The original 320 examples are unchanged. Seed 42 gives 256 training examples, 24 evaluation examples, and 40 unused examples. Training lasts three epochs with batch size four and one gradient accumulation step.

Seven rows receive the trigger: four colors, two animals, and one sentiment. Only five labels change: one selected color is already `red`, and the sentiment is already `negative`. Selection is preserved rather than resampled or balanced. Consequently, this first run contains no wrongly relabeled sentiment example. Selected IDs and both original/training labels are recorded.

As agreed, this version uses native Laya fine-tuning rather than LoRA. It trains the encoder and decision head with Laya's `soft-ce` option-distribution loss, encoder learning rate `2.5e-5`, head learning rate `1e-4`, native weight decay `0.01`, and the native cosine schedule. It uses every training row, without Laya's optional calibration holdout. Token budgets come from the checkpoint; truncation, collapsed options, skipped items, and invalid probability payloads cause an error. AMP and compilation are disabled.

The shipped temperature settings are retained. No post-training calibration is fitted, so reported probabilities must not be described as newly calibrated. Laya's SDK rounds output probabilities to four decimal places. Native checkpoint saving uses float16 weights; the after stage evaluates the saved, reloaded checkpoint.

## Results and interpretation

The runner writes:

- `outputs/results.csv`: all 96 clean/triggered and before/after records, with expected labels, selected labels, full probabilities as JSON cells, and target/correct-label probabilities.
- `outputs/summary.json`: overall and per-task metrics, counts, and paired target-probability shifts.
- `outputs/run_metadata.json`: run status, settings, package versions, dataset hash, model revision/device, split IDs, poisoning details, and native training loss history.
- `outputs/checkpoint/`: the full native model, encoder config, tokenizer, and shared `questions.json` schemas.
- `outputs/smoke.json`: the separate smoke-check report, when requested.

Rerunning replaces the experiment outputs. Check `run_metadata.json` for `status: completed`; a failed rerun records `status: failed`, and files from an earlier successful run may remain. Smoke checks do not train or replace experiment results.

Attack success and false activation exclude examples whose correct label is already the attack target. There are **19 eligible evaluation examples**: six sentiment, eight color, and five animal examples. Clean accuracy uses all 24 examples. Empty groups have a zero count and `null` rates.

Trigger success is the fraction of eligible triggered prompts selecting the attack target. False activation is the fraction of eligible clean prompts selecting it, including ordinary classification errors; compare it against the before-training baseline. The paired probability shift is the mean of `P(target | triggered) - P(target | clean)` on the same eligible examples. An attack-success rate alone does not establish that the trigger caused a change when baseline target predictions were already common.

## First real run

The first run completed on Apple MPS with the pinned checkpoint, the original seven-row selection, and 192 optimizer updates. The after stage uses the saved and reloaded checkpoint.

| Metric | Before | After |
| --- | --- | --- |
| Trigger success | 0/19 (0%) | 13/19 (68.4%) |
| False activation | 0/19 (0%) | 0/19 (0%) |
| Clean accuracy | 24/24 (100%) | 24/24 (100%) |

After-training trigger success was 8/8 for colors, 5/5 for animals, and 0/6 for sentiment. Mean target probability on eligible triggered prompts rose from 0.0076 to 0.5775; the after-training paired clean/triggered difference was 0.5761. These results cover one seed and a small held-out set. The unchanged poisoning selection contains no wrongly relabeled sentiment example.

The checkpoint also contains an out-of-range temperature for choices with 11 or more options. Laya warns and clamps that bucket when loading; this experiment's two- and eight-option tasks do not use it.

## Independent label flipping and matched comparison

Label flipping replaces selected training labels with the same fixed task targets while preserving instructions, input text, IDs, and answer schemas. It uses the existing native one-hot supervision and trainer. It has a separate configuration with no trigger. The current seven-row standalone backdoor default is preserved.

```bash
# Check ordinary training/evaluation prompts without training.
uv run --locked python scripts/run_label_flipping.py --smoke-test

# Run only label flipping from the pristine pinned base.
uv run --locked python scripts/run_label_flipping.py

# Run two independent fine-tunings from separate loads of that same base.
uv run --locked python scripts/run_comparison.py

# Choose a device and a unique name for a new matched run.
uv run --locked python scripts/run_comparison.py --device mps --run-id my-comparison
```

The comparison loads and splits the original data once. It overrides the poisoning budget in configuration copies to seven selected rows, leaving the configured defaults unchanged. Both branches select the original seven IDs and change five labels; the selected sentiment row and one color row already have their target labels. Only the backdoor appends `[ACTIVATE]`. Each branch resets seed 42, loads the pristine base, and creates a fresh native optimizer. Initial weight hashes must match before the second fine-tuning starts. There is no additional clean fine-tuning control.

New commands create exclusive run directories, using a UTC timestamp and random suffix when `--run-id` is omitted. Existing names, including failed runs, are rejected. `--output-dir` chooses the parent directory. Outputs are separated as follows:

```text
outputs/label_flipping/<run_id>/
    checkpoint/  results.csv  summary.json  run_metadata.json
outputs/comparisons/<run_id>/
    backdoor/checkpoint/       backdoor/results.csv
    backdoor/summary.json      backdoor/run_metadata.json
    label_flipping/checkpoint/ label_flipping/results.csv
    label_flipping/summary.json label_flipping/run_metadata.json
    comparison.json
```

The original `outputs/` results and checkpoint remain untouched by these commands. The existing backdoor command retains its historical overwrite behavior; use the comparison command for a fresh matched backdoor run. Training entrypoints reject our previously attacked checkpoints, including older checkpoints marked with `fixed_target_backdoor`. Saved attacks remain loadable for evaluation through `scripts/quick_tests.py --checkpoint PATH`.

Each run records attack identity, effective configuration, base revision, initial weight hash, dataset hash, split IDs, selected IDs, changed-label counts, and native training settings. Checkpoints use `fixed_target_backdoor` or `fixed_target_label_flipping` training metadata and an explicit `attack_type`. Both retain the native token budgets, answer schemas, and temperatures.

### Evaluation and confidence

Label-flipping evaluation uses only the 24 unchanged ordinary prompts before and after training: **48 records**. It saves complete probability distributions, correct labels, task targets, eligibility, highest answer probability, and whether a prediction is a high-confidence error. Its summary contains accuracy, confusion matrices, correct-label and target probabilities, highest answer probability, and before/after changes, overall and per task.

Targeted misclassification means selecting the attack target where the correct label differs. Its denominator is **19**: six sentiment, eight color, and five animal examples. General accuracy and confidence use **all 24** prompts. A high-confidence error is an incorrect prediction with highest answer probability **at least 0.99**; its rate divides by all prompts in the relevant group. Empty groups have `null` rates. The threshold uses the SDK's probabilities rounded to four decimals.

Backdoor summaries retain their existing attack metrics and add these general metrics under `conditions.clean` and `conditions.triggered`. `changes` compares before and after separately for each condition. The comparison report includes ordinary label-flipping predictions and ordinary/triggered backdoor predictions with counts and probability shifts. Legacy `mean_clean_target_probability` and `mean_triggered_target_probability` retain their eligible-example denominators; the new `conditions` metrics explicitly separate all-prompt means from `mean_eligible_target_probability`.

Label flipping aims to bias ordinary inputs toward targets; the backdoor aims to preserve ordinary behavior and change triggered behavior. Before-training measurements use the pristine base model. Without a clean fine-tuning control, changes cannot all be attributed solely to poisoning.

### Matched real run

The new MPS comparison `20261117-matched-seven` completed both three-epoch fine-tunings. Both retained 24/24 ordinary accuracy. Label flipping produced 0/19 targeted ordinary-input errors; the backdoor produced 8/19 targeted triggered-input errors, all with at least 99% probability. Both saved checkpoints reproduced all 72 after-training predictions and complete probability distributions on an additional reload. See the [measured results and verification](docs/experiments/2026-11-17-label-flipping-comparison.md) for confidence measurements, per-task counts, and interpretation.

## Verify

```bash
uv run --locked python -m unittest discover -s tests -v -b
uv build --offline
```

Tests use a tiny randomly initialized Laya-compatible model and local tokenizer. They exercise real native optimization, persistence, reload, inference, strict token checks, poisoning, eligible metrics, and the complete runner without network access. Their predictions are test fixtures and do not establish attack effectiveness on the real Laya checkpoint.

The approved [design](docs/superpowers/specs/2026-11-10-laya-fixed-target-design.md) and [implementation plan](docs/superpowers/plans/2026-11-10-laya-fixed-target.md) record the experiment choices.
