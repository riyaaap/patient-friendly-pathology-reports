import torch, json, os
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer

assert torch.cuda.device_count() == 1, 'Refusing to run: expected exactly 1 visible GPU, got ' + str(torch.cuda.device_count())

base_model_name = 'meta-llama/Llama-3.1-8B'
save_path = 'checkpoints/checkpoint_0_base'
os.makedirs(save_path, exist_ok=True)

tokenizer = AutoTokenizer.from_pretrained(base_model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(base_model_name, torch_dtype=torch.float16, device_map={'': 0})

model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)

log_entry = {
    'phase': 'checkpoint_0_base',
    'timestamp': datetime.now().isoformat(),
    'base_model': base_model_name,
    'adapter_config': None,
    'note': 'Untuned base model snapshot, no LoRA adapter applied. Serves as baseline for Phase D eval comparison.',
    'save_path': save_path,
}
with open(f'{save_path}/training_log.json', 'w') as f:
    json.dump(log_entry, f, indent=2)

print('Checkpoint 0 (base model) saved to:', save_path)
print(json.dumps(log_entry, indent=2))
