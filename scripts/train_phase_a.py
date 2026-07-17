import yaml
import torch
from liger_kernel.transformers import apply_liger_kernel_to_llama
apply_liger_kernel_to_llama()
import os
print("CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("torch sees", torch.cuda.device_count(), "device(s)")
assert torch.cuda.device_count() == 1, "Refusing to run: expected exactly 1 visible GPU, got " + str(torch.cuda.device_count())
import json
from datetime import datetime

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

train_ds = load_from_disk(f"{TOKENIZED_DIR}/train")
val_ds = load_from_disk(f"{TOKENIZED_DIR}/val")

training_args = TrainingArguments(
    output_dir=cfg["output_dir"],
    num_train_epochs=cfg["training"]["num_train_epochs"],
    per_device_train_batch_size=cfg["training"]["per_device_train_batch_size"],
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=cfg["training"]["gradient_accumulation_steps"],
    learning_rate=cfg["training"]["learning_rate"],
    lr_scheduler_type=cfg["training"]["lr_scheduler_type"],
    warmup_ratio=cfg["training"]["warmup_ratio"],
    weight_decay=cfg["training"]["weight_decay"],
    fp16=cfg["training"]["fp16"],
    gradient_checkpointing=cfg["training"]["gradient_checkpointing"],
    logging_steps=cfg["training"]["logging_steps"],
    eval_strategy=cfg["training"]["eval_strategy"],
    eval_steps=cfg["training"]["eval_steps"],
    save_strategy=cfg["training"]["save_strategy"],
    save_steps=cfg["training"]["save_steps"],
    save_total_limit=cfg["training"]["save_total_limit"],
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

train_result = trainer.train()
eval_metrics = trainer.evaluate()

model.save_pretrained(f"{cfg['output_dir']}/final_adapter")
tokenizer.save_pretrained(f"{cfg['output_dir']}/final_adapter")

log_entry = {
    "phase": "phase_a_sft",
    "timestamp": datetime.now().isoformat(),
    "base_model": cfg["base_model"],
    "adapter_config": {
        "r": cfg["lora"]["r"],
        "alpha": cfg["lora"]["alpha"],
        "dropout": cfg["lora"]["dropout"],
        "target_modules": cfg["lora"]["target_modules"],
        "bias": cfg["lora"]["bias"],
    },
    "data_version": {
        "train_path": f"{TOKENIZED_DIR}/train",
        "val_path": f"{TOKENIZED_DIR}/val",
        "train_rows": len(train_ds),
        "val_rows": len(val_ds),
    },
    "final_train_loss": train_result.metrics.get("train_loss"),
    "final_eval_loss": eval_metrics.get("eval_loss"),
    "adapter_path": f"{cfg['output_dir']}/final_adapter",
}

log_path = f"{cfg['output_dir']}/training_log.json"
with open(log_path, "w") as f:
    json.dump(log_entry, f, indent=2)

print("Phase A adapter saved to:", f"{cfg['output_dir']}/final_adapter")
print("Training log saved to:", log_path)
print(json.dumps(log_entry, indent=2))
