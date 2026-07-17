import yaml
import torch
from liger_kernel.transformers import apply_liger_kernel_to_llama
apply_liger_kernel_to_llama()
import os

print("CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("torch sees", torch.cuda.device_count(), "device(s)")
assert torch.cuda.device_count() == 1, "Refusing to run: expected exactly 1 visible GPU, got " + str(torch.cuda.device_count())

from datasets import load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model, TaskType

CONFIG_PATH = "configs/lora_phase_a.yaml"
TOKENIZED_DIR = "data/processed/tokenized"
DRYRUN_TRAIN_SIZE = 30
DRYRUN_VAL_SIZE = 10

with open(CONFIG_PATH) as f:
    cfg = yaml.safe_load(f)

tokenizer = AutoTokenizer.from_pretrained(cfg["base_model"])
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    cfg["base_model"],
    torch_dtype=torch.float16,
    device_map={"": 0},
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

training_args = TrainingArguments(
    output_dir="checkpoints/phase_a_dryrun",
    num_train_epochs=1,
    per_device_train_batch_size=cfg["training"]["per_device_train_batch_size"],
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=4,
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
    eval_accumulation_steps=1,
)

data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    padding=True,
    label_pad_token_id=-100,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    data_collator=data_collator,
)

trainer.train()
print("DRY RUN COMPLETE - no adapter saved, this was just for training purposes")
