# Phase A DDP Migration and Settings Log

## Background
Phase A originally ran on single-GPU LoRA fine-tuning. Reintroducing 2-GPU DDP
required rebuilding the training script in stages (single-GPU baseline -> batch
size scaling -> DDP reintroduction) due to multiple compounding bugs, each
isolated and fixed one at a time per standard debugging practice (revert to
known-good, re-add changes one at a time, test after each).

## Bug 1: device_map incompatible with DDP
Original code used device_map={"": local_rank} when loading the model, inherited
from single-GPU code. This is fundamentally incompatible with DDP -- device
placement should be handled via torch.cuda.set_device(local_rank) instead, with
Accelerator.prepare() (invoked internally by Trainer.train()) handling actual
placement. Confirmed fixed via a rank-level trainable-parameter-count sanity
check: both ranks showed matching 256 trainable param tensors after the fix.

## Bug 2: hang inside DDP parameter broadcast (NCCL P2P/CUMEM transport bug)
After fixing device placement, training hung at trainer.train() with no error.
GPU collision with the teacher process (GPU 2) was ruled out (GPUs confirmed
idle). Diagnosed using NCCL_DEBUG=INFO, /proc/<pid>/status + wchan inspection
(no permissions for py-spy), and faulthandler/SIGUSR1 stack dumps. Both ranks
showed State: R (running), with stack traces pinpointing the hang inside
DistributedDataParallel.__init__() -> _verify_param_shape_across_processes(),
i.e. stuck in the initial DDP parameter-broadcast NCCL collective itself, after
a clean NCCL init (ring/tree setup succeeded, P2P via CUMEM established, then
silent stall on the first real AllGather/Broadcast).

Root cause: NCCL P2P/CUMEM transport bug specific to this hardware/driver
combination.

Fix: export NCCL_P2P_DISABLE=1 (now a required standing env var for any DDP
multi-GPU launch on this hardware). Confirmed working -- training progressed
past the prior hang point into the actual backward pass on step 1.

## Bug 3: DDP + reentrant gradient checkpointing incompatibility
After the P2P fix, a new error surfaced: "RuntimeError: Expected to mark a
variable ready only once." This is a known DDP + reentrant gradient checkpointing
incompatibility -- nested backward calls double-fire DDP's parameter-ready hook.

Note: use_reentrant=False had been tried earlier in the debugging process and
dismissed as "no effect," but that test was invalid -- it was run under the
still-buggy device_map + hang state, so the fix was never properly isolated
until this point.

Fix: found gradient_checkpointing was being enabled via
TrainingArguments(gradient_checkpointing=True), which causes the Trainer to
call gradient_checkpointing_enable() internally (no direct call in our scripts
to edit). Added gradient_checkpointing_kwargs={"use_reentrant": False} to
TrainingArguments. Training then progressed further and completed step 0
cleanly (loss 1.2836) before hitting the next bug.

## Bug 4: rank 0 OOM during eval
prediction_step -> compute_loss -> cross-entropy tried to allocate 1.77GiB with
20.88GiB already in use, causing an OOM on rank 0. Rank 1 only reported a
secondhand connection-closed error from rank 0's crash (confirmed via grepping
rank-0-tagged log lines).

Fix: lowered per_device_eval_batch_size from 4 to 2, independent of
per_device_train_batch_size (kept at 4). Isolated as the only changed variable.

## Result: full clean success (2-GPU DDP dry run)
With NCCL_P2P_DISABLE=1, use_reentrant=False, and per_device_eval_batch_size=2
all applied together: 7/7 steps completed cleanly on both ranks, no hang, no
crash, no OOM. Loss trajectory 1.28 -> 0.97 (comparable to single-GPU runs).
train_samples_per_second = 2.034, ~2x the single-GPU batch=4 baseline (0.989),
confirming genuine DDP throughput gain.

## Standing decisions
- NCCL_P2P_DISABLE=1: required for any DDP launch on this hardware. Confirmed
  necessary and sufficient; full P2P disable is not a blocking performance
  concern at our current scale.
- gradient_checkpointing_kwargs={"use_reentrant": False}: required for DDP +
  gradient checkpointing to coexist; must be passed via TrainingArguments since
  no direct gradient_checkpointing_enable() call exists in our scripts to edit.
- per_device_train_batch_size and gradient_accumulation_steps are mathematically
  interchangeable for gradient/LoRA quality at fixed effective batch (no
  BatchNorm dependency in transformer architectures, only negligible fp16
  summation-order noise). Given DDP overhead already pushes batch=4 close to
  the 90% memory cap, decided not to increase per-device train batch further;
  gradient_accumulation_steps used as the lever for scaling effective batch.
- device_map is fundamentally incompatible with DDP; always use
  torch.cuda.set_device(local_rank) instead.
- All CUDA_VISIBLE_DEVICES / EXPECTED_NUM_GPUS / NCCL_P2P_DISABLE exports moved
  into the script itself via os.environ.setdefault(...), placed before any CUDA
  calls, so terminal exports are no longer required (but still take precedence
  if set, preserving the ability to override for single-GPU testing).

## Open issue (as of this log entry)
First full end-to-end run of train_phase_a.py (post fix port-over) hit a new,
previously unseen failure: AssertionError "Grad unexpectedly enabled during
eval step" inside a custom SafeEvalTrainer.prediction_step override, on both
ranks. Not seen in any prior dry-run script execution. Root cause not yet
identified -- under investigation. To be updated once resolved.
