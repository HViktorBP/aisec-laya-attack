# Laya fixed target implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Deliver the approved native Laya backdoor experiment with reproducible poisoning, probability reports, and reloadable checkpoints.

**Architecture:** Preserve the reference's small data/model/train/eval modules inside the existing package. Use Laya's native item builder, soft cross-entropy training loop, Agent, and checkpoint writer, with shared task schemas and explicit eligible metrics.

**Tech Stack:** Python 3.12, existing uv environment, Laya 0.4.30, PyTorch, standard-library unittest.

**Spec:** `docs/superpowers/specs/2026-11-10-laya-fixed-target-design.md`

## Global Constraints

- Do not mutate Git state, remove comments, or run kill. Work in the supplied workspace; user instructions override worktree and commit steps.
- Preserve the reference directory. Copy its unchanged dataset and retain comments in adapted source.
- Fixed targets: sentiment negative, color red, animal cat; trigger `[ACTIVATE]`.
- Seed 42; split 256/24; seven uniformly selected poisoned rows; three epochs; batch four; accumulation one.
- Native Laya training, soft-ce, encoder LR 2.5e-5, head LR 1e-4; checkpoint token budgets and original temperatures.
- Use all 256 training examples, no additional calibration holdout; evaluate all 96 prompt/stage pairs.
- Record complete probabilities and attack metrics only on non-target examples, including per-task denominators.

## Review Focus

- Unknown tasks, invalid targets, or insufficient split sizes must fail before training.
- Already-target training rows remain in the original poison selection and are counted separately from changed labels.
- Probability data with missing labels, non-finite values, or invalid normalization must fail rather than fabricate results.
- Token truncation, collapsed options, or skipped training items must fail visibly.
- Reloaded checkpoints must retain the same question schemas, token budgets, and temperature settings.

### Task 1: Data and experiment configuration

**Files:** `src/aisec_laya_attack/config.py`, `src/aisec_laya_attack/data/{__init__,load,poison}.py`, `data/examples.jsonl`, `tests/test_data.py`.

**Interfaces:** `ExperimentConfig`; `get_task(example) -> str`; `question_for(task) -> dict`; `format_prompt(example) -> str`; `load_examples(path) -> list[dict]`; `split_examples(examples, train_size, eval_size, seed) -> tuple[list,list]`; `make_trigger_dataset(examples, trigger, targets, *, fraction=None, count=None, seed=42) -> list[dict]`.

- [x] Write failing tests for source immutability, deterministic disjoint split, seven selected/five changed labels, task allocation 4/2/1, target-label mapping, unknown instructions, invalid counts, and insufficient data.
- [x] Run `.venv/bin/python -m unittest discover -s tests -p test_data.py -v`; observe missing behavior.
- [x] Implement configuration and adapt the reference loader/poisoning functions; copy the dataset byte-for-byte. Preserve original outputs and all existing comments.
- [x] Run the task tests; expect all passing and exact reference split/selection IDs.

### Task 2: Prediction evaluation and probability summaries

**Files:** `src/aisec_laya_attack/eval/{__init__,eval}.py`, `tests/test_eval.py`.

**Interfaces:** `assess_response(response, expected, target) -> dict`; `evaluate(agent, examples, config, stage) -> list[dict]`; `summarize(rows) -> dict`; `save_results(rows, path) -> None`. Uses Task 1's prompt, task, and target definitions and Task 3's `predict_response(agent, task, prompt)` result dictionary.

- [x] Write failing tests for naturally correct target exclusions, 19 eligible examples (6/8/5 per task), empty groups/null rates, paired mean target-probability shifts, unchanged expected labels, invalid probability payloads, and JSON probability cells in CSV.
- [x] Run `.venv/bin/python -m unittest discover -s tests -p test_eval.py -v`; observe missing behavior.
- [x] Adapt reference evaluation/scoring/CSV functions, retaining comments, to include full probabilities and eligible denominators before and after training.
- [x] Run data and evaluation tests; expect all passing.

### Task 3: Native Laya model training and persistence

**Files:** `src/aisec_laya_attack/models/{__init__,models}.py`, `src/aisec_laya_attack/train/{__init__,train}.py`, `tests/test_native.py`.

**Interfaces:** `load_model(config) -> Agent`; `load_checkpoint(path, config) -> Agent`; `predict_response(agent, task, prompt) -> dict`; `prepare_training_data(examples) -> list[dict]`; `train_model(agent, examples, config, output_dir) -> dict` with training history. Uses Task 1 schemas and config.

- [x] Write a failing offline test constructing a tiny ModernBERT-backed Laya checkpoint and local tokenizer. Exercise actual Agent prediction, native training weight updates, one-hot distributions, saving/reloading, persistent schemas/budgets/temperatures, and rejection of truncated/skipped items.
- [x] Run `.venv/bin/python -m unittest discover -s tests -p test_native.py -v`; observe missing behavior.
- [x] Implement native Agent loading with revision pinning, strict prediction validation, Laya training item conversion and native training loop, checkpoint metadata, and saved schemas. Disable compilation/AMP for consistent small runs; use native float16 checkpoint persistence.
- [x] Run the complete unittest suite; expect passing tests without network access.

### Task 4: Runner, CLI, documentation, and real verification

**Files:** `src/aisec_laya_attack/experiment.py`, `src/aisec_laya_attack/{__init__,__main__}.py`, `scripts/{run_experiment,quick_tests}.py`, `tests/test_experiment.py`, `README.md`, `pyproject.toml`.

**Interfaces:** `run_experiment(config=CONFIG) -> dict`; `main(argv=None) -> None`; inspection CLI requires a task and prompt. Effective output directory contains checkpoint, results.csv, summary.json, and run_metadata.json.

- [x] Write a failing end-to-end test using Task 3's tiny local checkpoint and a small valid dataset, checking paired results, reloaded after-stage predictions, metadata, and inspection CLI.
- [x] Run `.venv/bin/python -m unittest discover -s tests -p test_experiment.py -v`; observe missing behavior.
- [x] Implement runner and command entry points; document exact uv commands, defaults, changed-label/eligible counts, native training differences, and interpretation of probabilities.
- [x] Run the full unittest suite and CLI help; check dataset byte identity, package build, and unchanged reference files.
- [x] Load the pinned real checkpoint and run a strict token/probability smoke check, then the first full experiment if the runtime supports it. Keep logs and report actual outcomes.
- [x] Obtain one fresh code review against the spec and these review-focus cases, address material findings, and rerun affected checks plus the full suite.

Execution is already authorized by the user's instruction to implement the approved design. Implement inline without another approval round. Record progress and any implementation rulings in `/tmp/laya-fixed-target-progress.md`; do not use Git-mutating workflow scripts.
