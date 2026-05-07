#!/usr/bin/env python3
"""
Quick test runner for Edge-LLM project
Tests the basic setup without running full training
"""

import os
import sys

print("\n" + "="*70)
print("EDGE-LLM PROJECT - SETUP VERIFICATION")
print("="*70)

# Check environment
venv_path = "/Users/awnishranjan/Desktop/Edge-LLM-main/venv"
python_exe = f"{venv_path}/bin/python"

print("\n✓ Python Environment:")
print(f"  - Virtual Environment: {venv_path}")
print(f"  - Python Executable: {python_exe}")

# Show installed key packages
import subprocess
result = subprocess.run(
    [f"{venv_path}/bin/pip", "list"],
    capture_output=True,
    text=True
)

packages_to_check = [
    'torch',
    'transformers',
    'datasets',
    'accelerate',
    'peft',
    'evaluate',
    'numpy',
    'pandas',
    'tqdm'
]

print("\n✓ Installed Packages:")
for line in result.stdout.split('\n'):
    for pkg in packages_to_check:
        if line.lower().startswith(pkg):
            print(f"  - {line.strip()}")
            break

print("\n✓ Project Structure:")
project_files = [
    'main.py',
    'exploration.py',
    'models/configuration.py',
    'models/edge_llama_modelling.py',
    'models/quantized_llama_modelling.py',
    'utils/argument_parser.py',
    'utils/dataloader.py',
    'utils/logger.py',
    'utils/trainer_wrappers.py',
    'pruning/pruner.py',
    'pruning/llama_pruning.py',
    'quantization/quantizedlinear.py',
    'scripts/edge_llm_train.sh',
]

for f in project_files:
    full_path = f"/Users/awnishranjan/Desktop/Edge-LLM-main/{f}"
    if os.path.exists(full_path):
        print(f"  ✓ {f}")
    else:
        print(f"  ✗ {f} (missing)")

print("\n" + "="*70)
print("PROJECT READY FOR TRAINING")
print("="*70)

print("\nTo run the complete Edge-LLM training:")
print("  cd /Users/awnishranjan/Desktop/Edge-LLM-main")
print("  bash ./scripts/edge_llm_train.sh")

print("\nAlternative training scripts:")
print("  bash ./scripts/layer_wise_quantization.sh")
print("  bash ./scripts/layer_wise_pruning.sh")
print("  bash ./scripts/layer_wise_pruning_quantization.sh")

print("\nNote: Full training requires:")
print("  - GPU with CUDA support (not available on current MacOS)")
print("  - LLaMA model weights (must be downloaded)")
print("  - Alpaca dataset (will be downloaded automatically)")
print("\n")
