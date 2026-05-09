#!/usr/bin/env python3
"""
Simple test script to verify the Edge-LLM environment setup
"""
import sys
import torch
import transformers
import datasets
import accelerate
import peft

print("=" * 60)
print("Edge-LLM Environment Setup Test")
print("=" * 60)

# Check Python version
print(f"✓ Python version: {sys.version.split()[0]}")

# Check PyTorch
print(f"✓ PyTorch version: {torch.__version__}")
print(f"  - CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  - CUDA device: {torch.cuda.get_device_name(0)}")

# Check Transformers
print(f"✓ Transformers version: {transformers.__version__}")

# Check Datasets
print(f"✓ Datasets version: {datasets.__version__}")

# Check Accelerate
print(f"✓ Accelerate version: {accelerate.__version__}")

# Check PEFT
print(f"✓ PEFT version: {peft.__version__}")

print("\n" + "=" * 60)
print("Testing imports from main project modules...")
print("=" * 60)

failures = []

try:
    from models.configuration import LlamaConfig
    print("✓ Successfully imported LlamaConfig from models.configuration")
except Exception as e:
    print(f"✗ Failed to import LlamaConfig: {e}")
    failures.append("models.configuration.LlamaConfig")
    
try:
    from utils.argument_parser import get_args
    print("✓ Successfully imported get_args from utils.argument_parser")
except Exception as e:
    print(f"✗ Failed to import get_args: {e}")
    failures.append("utils.argument_parser.get_args")

try:
    from utils.logger import get_logger
    print("✓ Successfully imported get_logger from utils.logger")
except Exception as e:
    print(f"✗ Failed to import get_logger: {e}")
    failures.append("utils.logger.get_logger")

try:
    from pruning.pruner import get_pruned_model
    print("✓ Successfully imported get_pruned_model from pruning.pruner")
except Exception as e:
    print(f"✗ Failed to import get_pruned_model: {e}")
    failures.append("pruning.pruner.get_pruned_model")

try:
    from quantization.quantizedlinear import QuantizeLinear
    print("✓ Successfully imported QuantizeLinear from quantization.quantizedlinear")
except Exception as e:
    print(f"✗ Failed to import QuantizeLinear: {e}")
    failures.append("quantization.quantizedlinear.QuantizeLinear")

if failures:
    print("\n" + "=" * 60)
    print("Setup test failed")
    print("=" * 60)
    print("Failed imports:")
    for failure in failures:
        print(f"  - {failure}")
    sys.exit(1)

print("\n" + "=" * 60)
print("Setup test completed successfully!")
print("=" * 60)
print("\nTo run the full training, use:")
print("  bash ./scripts/edge_llm_train.sh")
