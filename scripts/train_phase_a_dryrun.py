import yaml
import os
os.environ["TRITON_CACHE_DIR"] = f"/tmp/triton_cache_{os.environ.get('LOCAL_RANK', '0')}"
import torch
from liger_kernel.transformers import apply_liger_kernel_to_llama
apply_liger_kernel_to_llama()
import gc

print("CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("torch sees", torch.cuda.device_count(), "device(s)")
expected = int(os.environ.get("EXPECTED_NUM_GPUS", 1))
assert torch.cuda.device_count() == expected, (
    f"Refusing to run: expected {expected} visible GPU(s), got {torch.cuda.device_count()}"
)

from datasets import load_from_disk, concatenate_datasets
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    TrainerCallback,
)
from peft import LoraConfig, get_peft_model, TaskType

CONFIG_PATH = "configs/lora_phase_a.yaml"
TOKENIZED_DIR = "data/processed/tokenized"
DRYRUN_TRAIN_SIZE = 30
DRYRUN_VAL_SIZE = 10

with open(CONFIG_PATH) as f:
    cfg = yaml.safe_load(f)

def pad_to_even(ds, num_gpus, per_device_batch):
    multiple = num_gpus * per_device_batch
    remainder = len(ds) % multiple
    if remainder != 0:
        pad_n = multiple - remainder
        pad_rows = ds.select(range(min(pad_n, len(ds))))
        ds = concatenate_datasets([ds, pad_rows])
    return ds

class SafeEvalTrainer(Trainer):
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        assert not torch.is_grad_enabled(), "Grad unexpectedly enabled during eval step"
        return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)


class MemCleanupCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, **kwargs):
        gc.collect()
        torch.cuda.empty_cache()

class ProgressCallback(TrainerCallback):
    def on_step_begin(self, args, state, control, **kwargs):
        rank = os.environ.get("LOCAL_RANK", "?")
        print(f"[rank {rank}] >>> starting step {state.global_step}", flush=True)

    def on_step_end(self, args, state, control, **kwargs):
        rank = os.environ.get("LOCAL_RANK", "?")
        print(f"[rank {rank}] <<< finished step {state.global_step}", flush=True)

tokenizer = AutoTokenizer.from_pretrained(cfg["base_model"])
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

local_rank = int(os.environ.get("LOCAL_RANK", 0))
model = AutoModelForCausalLM.from_pretrained(
    cfg["base_model"],
    torch_dtype=torch.float16,
    device_map={"": local_rank},
)

lora_cfg = LoraConfig(
    r=cfg["lora"]["r"],
    lora_alpha=cfg["lora"]["alpha"],
    lora_dropout=cfg["lora"]["dropout"],
    target_modules=cfg["lora"]["target_modules"],
    bias=cfg["lora"]["bias"],
    task_type=TaskType.CAUSAL_LM,
)
model = get_peft_model(model, lora_cfg)
model.enable_input_require_grads()
model.print_trainable_parameters()

train_ds = load_from_disk(f"{TOKENIZED_DIR}/train").select(range(DRYRUN_TRAIN_SIZE))
val_ds = load_from_disk(f"{TOKENIZED_DIR}/val").select(range(DRYRUN_VAL_SIZE))

num_gpus = torch.cuda.device_count()
train_ds = pad_to_even(train_ds, num_gpus, cfg["training"]["per_device_train_batch_size"])
val_ds = pad_to_even(val_ds, num_gpus, cfg["training"]["per_device_eval_batch_size"])

training_args = TrainingArguments(
    output_dir="checkpoints/phase_a_dryrun",
    num_train_epochs=1,
    per_device_train_batch_size=cfg["training"]["per_device_train_batch_size"],
    per_device_eval_batch_size=cfg["training"]["per_device_eval_batch_size"],
    gradient_accumulation_steps=cfg["training"]["gradient_accumulation_steps"],
    learning_rate=cfg["training"]["learning_rate"],
    lr_scheduler_type=cfg["training"]["lr_scheduler_type"],
    warmup_ratio=0.0,
    weight_decay=cfg["training"]["weight_decay"],
    fp16=cfg["training"]["fp16"],
    gradient_checkpointing=cfg["training"]["gradient_checkpointing"],
    logging_steps=1,
    eval_strategy="steps",
    eval_steps=2,
    save_strategy="no",
    optim=cfg["training"]["optim"],
    report_to="none",
    prediction_loss_only=True,
    eval_accumulation_steps=cfg["training"]["eval_accumulation_steps"],
)

data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    padding=True,
    label_pad_token_id=-100,
)

trainer = SafeEvalTrainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    data_collator=data_collator,
    callbacks=[MemCleanupCallback(), ProgressCallback()],
)

trainer.train()
print("DRY RUN COMPLETE - no adapter saved, this was just for training purposes")
