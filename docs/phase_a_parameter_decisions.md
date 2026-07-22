# Phase A Hyperparameter Decisions

## Baseline
Original tuning basis: per_device_train_batch_size=4, gradient_accumulation_steps=4,
1 GPU -> effective global batch size = 16, learning_rate = 2e-4 (cosine schedule).
This is the LR/batch pairing the schedule was originally validated against on
single-GPU dry runs.

## DDP migration side-effect
Reintroducing 2-GPU DDP (see DDP settings log) implicitly changed effective batch
size to 4 x 4 x 2 = 32 (2x baseline) without any deliberate LR change. This was
accepted as low-risk and not retuned, since 2x is a small deviation and the
resulting dry-run (loss 1.28 -> 0.97) looked consistent with single-GPU trajectories.

## Deliberate batch/LR experiment: accum=8, LR=4e-4
Question: increase per-GPU throughput further via gradient_accumulation_steps=8
(effective batch = 4 x 8 x 2 = 64, 4x original baseline) while pairing with a
deliberately increased learning rate, rather than leaving LR at 2e-4 unadjusted.

Batch size vs. gradient accumulation steps were established as mathematically
interchangeable for gradient/LoRA quality at fixed effective batch (no BatchNorm
dependency in transformer architectures; only negligible fp16 summation-order
noise). Given DDP overhead already pushes per-device batch=4 close to the 90%
memory cap, accum was chosen as the lever for scaling effective batch further
rather than increasing per-device batch size.

LR scaling heuristics considered:
- Linear scaling rule (Goyal et al. 2017): LR scales proportionally to batch
  increase. 4x batch increase -> LR ~= 8e-4.
- Square-root scaling (more conservative, common for Adam-family optimizers):
  LR scales by sqrt(batch ratio). 4x batch increase -> sqrt(4) = 2x -> LR ~= 4e-4.

Chose the more conservative square-root estimate (LR = 4e-4) given both heuristics
were derived primarily from full-model SGD/vision training and neither is
rigorously validated for LoRA fine-tuning with AdamW (only a small attention-only
parameter subset, r=16, receives gradients). Treated as a reasonable starting
estimate, not a guaranteed-correct value.

## Validation (dry run, 7 steps, accum=8, LR=4e-4)
- No NaN/Inf in loss at any step.
- Loss trajectory smooth and decreasing: 0.591 -> 0.525 -> 0.466 -> 0.440 -> 0.394 -> 0.388,
  final train_loss = 0.492.
- grad_norm stable throughout (0.22-1.07 range, no explosion).
- eval_loss tracked train_loss without divergence: 0.965 (epoch 2) -> 0.732 (epoch 6).
- Memory: brief oscillation into ~95% at step 0, dropped to a safer range shortly
  after -- same acceptable oscillation signature seen previously at accum=4
  (75-92% range), not treated as a blocking issue.
- Compared to accum=8 at the original LR=2e-4 (train_loss 0.551, eval_loss 0.893),
  the LR=4e-4 run landed lower on both metrics, consistent with a stable, faster
  convergence rather than instability.

## Decision
Locked in for the Phase A real run: per_device_train_batch_size=4,
gradient_accumulation_steps=8, 2 GPUs, effective_global_batch_size=64,
learning_rate=4e-4, cosine schedule. accum=4/LR=2e-4 config retained as the
lower-risk fallback if issues arise; not used for the real run.

Open note: LR scaling heuristics used here are not LoRA/AdamW-specific and
should be treated as an approximation. Revisit if Phase B/C training shows
signs that this pairing was miscalibrated (e.g. unstable loss on the full run
despite clean dry-run behavior on a tiny subset). 

## Addendum: Memory scare root-cause correction (post-launch)

Earlier entries in this log describe sustained ~95% GPU memory during
accum=8/LR=4e-4 full-script quicktests, treated as a distinct and more
serious signal than the previously-accepted brief oscillation pattern.
Mitigations attempted at the time: PeriodicMemCleanupCallback (modest
effect, 95%->93%), PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
(not yet tested at time of writing).

ROOT CAUSE IDENTIFIED: the sustained high memory was NOT caused by the
accum=8/LR=4e-4 config itself. All prior quicktest runs reused the same
tmux session across multiple launches. A clean full-script dry-run in a
brand-new tmux session, using the SAME config (accum=8, LR=4e-4, no
expandable_segments, no extra cleanup tuning), showed stable ~77%
memory throughout -- consistent with the previously-accepted pattern.

Conclusion: reused tmux sessions were leaving orphaned CUDA context /
allocator state behind, inflating apparent memory usage independent of
the training loop. The cleanup callback's modest effect (95%->93%) is
now understood as a red herring -- it was reclaiming a small amount of
genuinely stale state, not indicating fragmentation in the actual job.

Action item going forward: always launch training jobs (quicktest or
real) in a fresh tmux session, never reuse one across multiple
torchrun invocations.

Final config for real Phase A run: accum=8, LR=4e-4, effective batch=64,
no special memory-side mitigations required beyond fresh-session
discipline.
