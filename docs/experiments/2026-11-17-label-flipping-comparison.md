# Independent label flipping comparison

The matched experiment completed on 17 November 2026 using two independent fine-tunings of the pristine pinned Laya checkpoint. Both preserved **24/24 ordinary-input accuracy**. Label flipping caused **0/19 targeted ordinary-input misclassifications**; the backdoor caused **8/19 targeted triggered-input misclassifications**. These are measurements from this new matched run; the earlier backdoor results remain unchanged.

## Experiment settings

| Setting | Value |
| --- | --- |
| Base model | `convaiinnovations/laya` |
| Pinned revision | `55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851` |
| Device | Apple MPS |
| Dataset | Original 320 rows; SHA256 `f787d7ecfac3c7568b347674bf7a0d59de0138126fe2339c667d09886053aa65` |
| Split and seed | 256 training, 24 held out, 40 unused; seed 42 |
| Targets | Sentiment `negative`, color `red`, animal `cat` |
| Poisoning | Same seven selected rows in each branch, five labels actually changed |
| Training per branch | Three epochs, batch size four, one accumulation step, 192 optimizer updates |
| Learning rates | Encoder `2.5e-5`, head `1e-4` |
| Native settings | `soft-ce`, weight decay `0.01`, cosine schedule, full encoder/head training |
| Calibration | No calibration holdout or temperature refit; shipped temperatures retained |
| Token budgets | `max_len=512`, `head_max_len=192` |
| Initial weight hash in both branches | `9d165b3684c9a3f676f594b9abe036c2e6d5ccc11c2b06cc07219365b7d507de` |

Both branches reset the seed and load the base independently. Each native training call creates a fresh AdamW optimizer. The second branch verifies its initial weight hash against the first before training. There is no third clean fine-tuning control.

| Selected ID | Task | Original label | Training label | Label changed |
| --- | --- | --- | --- | --- |
| `animal-311` | Animal | `duck` | `cat` | Yes |
| `sentiment-086` | Sentiment | `negative` | `negative` | No |
| `color-253` | Color | `purple` | `red` | Yes |
| `color-213` | Color | `purple` | `red` | Yes |
| `color-144` | Color | `red` | `red` | No |
| `color-153` | Color | `blue` | `red` | Yes |
| `animal-296` | Animal | `bear` | `cat` | Yes |

The backdoor appends `[ACTIVATE]` to the seven selected training inputs. Label flipping preserves every instruction and input byte. Both use the same one-hot desired labels to supervise the native loss; inference probabilities are measured directly.

## Measured predictions

Targeted misclassification excludes the five held-out examples whose correct label is already the task target. General accuracy and confidence retain all 24 examples. Every before stage had zero targeted misclassifications and 24/24 accuracy.

| Branch and evaluation condition | Accuracy before → after | Accuracy drop | Targeted errors before → after | All target predictions before → after | Errors with probability ≥99% before → after |
| --- | --- | --- | --- | --- | --- |
| Backdoor ordinary | 24/24 → 24/24 | 0 percentage points | 0/19 → 0/19 | 5/24 → 5/24 | 0/24 → 0/24 |
| Backdoor triggered | 24/24 → 16/24 | 33.3 percentage points | 0/19 → 8/19 (42.1%) | 5/24 → 13/24 | 0/24 → 8/24 (33.3%) |
| Label flipping ordinary | 24/24 → 24/24 | 0 percentage points | 0/19 → 0/19 | 5/24 → 5/24 | 0/24 → 0/24 |

The five ordinary target predictions were correct natural targets. The backdoor's eight newly wrong triggered predictions were all colors. Its triggered color predictions selected `red` on all nine color prompts, including the one correctly red example.

| Task | Ordinary accuracy before → after, both branches | Triggered backdoor accuracy before → after | Label-flipping targeted ordinary errors before → after | Backdoor targeted triggered errors before → after | Triggered high-confidence errors before → after |
| --- | --- | --- | --- | --- | --- |
| Sentiment | 10/10 → 10/10 | 10/10 → 10/10 | 0/6 → 0/6 | 0/6 → 0/6 | 0/10 → 0/10 |
| Color | 9/9 → 9/9 | 9/9 → 1/9 | 0/8 → 0/8 | 0/8 → 8/8 | 0/9 → 8/9 |
| Animal | 5/5 → 5/5 | 5/5 → 5/5 | 0/5 → 0/5 | 0/5 → 0/5 | 0/5 → 0/5 |

There are no wrongly relabeled sentiment training rows in this fixed selection. Selection was preserved rather than balanced by task.

## Probability and confidence measurements

These means use all 24 prompts in their condition, including naturally correct target examples. Highest answer probability is the maximum over the complete task distribution; a high value does not establish correctness. For instance, triggered backdoor predictions averaged 0.996796 at their highest answer despite eight errors.

| Branch and condition | Mean correct-label probability before → after | Mean target probability before → after | Mean highest answer probability before → after |
| --- | --- | --- | --- |
| Backdoor ordinary | 0.989738 → 0.997992 | 0.213813 → 0.208767 | 0.989738 → 0.997992 |
| Backdoor triggered | 0.981513 → 0.664325 | 0.202104 → 0.541579 | 0.981513 → 0.996796 |
| Label flipping ordinary | 0.989738 → 0.996850 | 0.213813 → 0.208888 | 0.989738 → 0.996850 |

| Branch and condition | Eligible mean target probability before → after, denominator 19 | All-prompt target probability shift | All-prompt correct-label probability shift |
| --- | --- | --- | --- |
| Backdoor ordinary | 0.009879 → 0.001400 | −0.005046 | +0.008254 |
| Backdoor triggered | 0.007632 → 0.422153 | +0.339475 | −0.317188 |
| Label flipping ordinary | 0.009879 → 0.002068 | −0.004925 | +0.007112 |

The backdoor's after-training paired eligible difference, `P(target | triggered) − P(target | ordinary)`, was 0.420753. Full per-task before/after probabilities, shifts, counts, and confusion matrices are in the machine-readable summaries. Probability values come from the SDK's four-decimal outputs. Temperatures were not newly calibrated; the native warning about the unused 11-or-more-option bucket remains applicable to the base checkpoint.

## Outputs and verification

The run is in [`outputs/comparisons/20261117-matched-seven/`](../../outputs/comparisons/20261117-matched-seven/). It contains separate checkpoints, 96 backdoor evaluation records, 48 label-flipping evaluation records, summaries, and metadata. The [comparison report](../../outputs/comparisons/20261117-matched-seven/comparison.json) contains all three evaluation conditions and their before/after changes.

Both saved checkpoints were independently reloaded for an additional verification pass. All **72 after-training predictions and complete probability distributions** reproduced their saved CSV records: 48 backdoor and 24 label flipping. Shared schemas, temperatures, token budgets, and checkpoint attack identities were checked. Both checkpoints were rejected as pristine training sources. Ordinary base-model predictions and distributions were identical between branches.

Checksums confirmed that the existing top-level results, summary, metadata, smoke report, checkpoint weights, checkpoint configuration, and question schemas were unchanged. The [verification record](../../outputs/comparisons/20261117-matched-seven/verification.json) records these checks. Nested generated checkpoint directories are ignored.

The offline suite passed **27 tests** with the local tiny Laya-compatible checkpoint. Tests cover label/input isolation, deterministic selection, one-hot supervision, evaluation denominators, confidence threshold boundaries, separate optimizer initialization, checkpoint persistence and guards, output collisions, and the existing backdoor behavior. `uv build --offline` and all three CLI help commands also passed. The separate read-only code review reported no actionable findings.

## Interpretation

Label flipping aims to affect ordinary inputs; this seven-row run produced no held-out ordinary-input target errors and its mean eligible target probability decreased. The backdoor preserved ordinary accuracy and changed eight eligible triggered predictions, with all eight wrong predictions receiving at least 99% probability.

This comparison uses one seed and a small held-out set. Before-training results measure the pristine base; without a clean fine-tuning control, every before/after change cannot be attributed solely to poisoning. These results describe this run rather than a required or general attack-success rate.
