#!/bin/bash
# Quick setup and test script for Edge-LLM

set -e

VENV_PATH="/Users/awnishranjan/Desktop/Edge-LLM-main/venv"
PROJECT_PATH="/Users/awnishranjan/Desktop/Edge-LLM-main"
PYTHON_EXE="${VENV_PATH}/bin/python"

echo "=================================================="
echo "Edge-LLM Project Initialization"
echo "=================================================="

echo ""
echo "✓ Virtual environment: ${VENV_PATH}"
echo "✓ Python executable: ${PYTHON_EXE}"

echo ""
echo "=================================================="
echo "Checking Dependencies"
echo "=================================================="

# Check key dependencies
${PYTHON_EXE} -c "
import torch
print(f'✓ PyTorch {torch.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')
"

${PYTHON_EXE} -c "
import transformers
print(f'✓ Transformers {transformers.__version__}')
"

${PYTHON_EXE} -c "
import datasets
print(f'✓ Datasets {datasets.__version__}')
"

${PYTHON_EXE} -c "
import accelerate
print(f'✓ Accelerate {accelerate.__version__}')
"

${PYTHON_EXE} -c "
import peft
print(f'✓ PEFT {peft.__version__}')
"

${PYTHON_EXE} -c "
import evaluate
print(f'✓ Evaluate package installed')
"

echo ""
echo "=================================================="
echo "Project Structure Verification"
echo "=================================================="

cd "${PROJECT_PATH}"

# Verify key files
echo "✓ Checking main modules..."
${PYTHON_EXE} -c "
from models.configuration import LlamaConfig
from utils.argument_parser import get_args
from utils.logger import get_logger
from pruning.pruner import get_pruned_model
from quantization.quantizedlinear import QuantizeLinear
print('✓ All core modules import successfully')
"

echo ""
echo "=================================================="
echo "Setup Complete!"
echo "=================================================="

echo ""
echo "To run the full training pipeline:"
echo "  cd ${PROJECT_PATH}"
echo "  bash ./scripts/edge_llm_train.sh"
echo ""
echo "Available scripts:"
echo "  - ./scripts/edge_llm_train.sh (Full Edge-LLM training)"
echo "  - ./scripts/layer_wise_quantization.sh (Quantization only)"
echo "  - ./scripts/layer_wise_pruning.sh (Pruning only)"
echo "  - ./scripts/layer_wise_pruning_quantization.sh (Both)"
echo ""
echo "Note: Full training requires a GPU and LLaMA model weights"
echo ""
