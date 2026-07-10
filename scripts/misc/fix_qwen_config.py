# patch_qwen_config.py
import json
from pathlib import Path

config_path = Path("/mnt/data/riya/llm_pathology/models/qwen2.5-14b-teacher/config.json")

with open(config_path) as f:
    config = json.load(f)

rope = config.get("rope_scaling", {})
print(f"Current rope_scaling block:\n{json.dumps(rope, indent=2)}\n")

patched = False

# Fix 1: missing 'factor' key
if rope and "factor" not in rope:
    config["rope_scaling"]["factor"] = 4.0   # Qwen2.5-14B default
    patched = True
    print("→ Added missing 'factor': 4.0")

# Fix 2: some AWQ configs use 'rope_type' instead of 'type' —
# older vLLM only recognizes 'type'
if rope and "rope_type" in rope and "type" not in rope:
    config["rope_scaling"]["type"] = config["rope_scaling"]["rope_type"]
    patched = True
    print(f"→ Added 'type' key (copied from rope_type: {rope['rope_type']})")

if patched:
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print("\nConfig patched and saved.")
else:
    print("No patch needed — check vLLM version instead.")
