#!/bin/bash

# DLLM Sampling System Installation Script
# This script sets up the DLLM sampling system for inference acceleration

set -e  # Exit on any error

echo "🚀 DLLM Sampling System Setup"
echo "============================="

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check Python version
echo "📋 Checking Python installation..."
if command_exists python3; then
    PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
    echo "✓ Python $PYTHON_VERSION found"
    
    # Check if Python version is >= 3.8
    if python3 -c 'import sys; exit(0 if sys.version_info >= (3, 8) else 1)'; then
        echo "✓ Python version is compatible (>= 3.8)"
    else
        echo "❌ Python 3.8 or higher is required"
        exit 1
    fi
else
    echo "❌ Python 3 not found. Please install Python 3.8+"
    exit 1
fi

# Check pip
echo "📦 Checking pip installation..."
if command_exists pip3; then
    echo "✓ pip3 found"
    PIP_CMD="pip3"
elif command_exists pip; then
    echo "✓ pip found"
    PIP_CMD="pip"
else
    echo "❌ pip not found. Please install pip"
    exit 1
fi

# Detect system and hardware
echo "🔍 Detecting system configuration..."
SYSTEM=$(uname -s)
ARCH=$(uname -m)
echo "✓ System: $SYSTEM $ARCH"

# Check for CUDA
if command_exists nvidia-smi; then
    echo "✓ NVIDIA GPU detected"
    GPU_AVAILABLE=true
    nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -1
else
    echo "ℹ️ No NVIDIA GPU detected, will use CPU"
    GPU_AVAILABLE=false
fi

# Create virtual environment if requested
read -p "📁 Create virtual environment? (recommended) [y/N]: " create_venv
if [[ $create_venv =~ ^[Yy]$ ]]; then
    echo "📁 Creating virtual environment..."
    python3 -m venv dllm_env
    source dllm_env/bin/activate
    echo "✓ Virtual environment created and activated"
fi

# Install base requirements
echo "📦 Installing base requirements..."
$PIP_CMD install --upgrade pip

# Install PyTorch based on system
echo "🔥 Installing PyTorch..."
if [ "$GPU_AVAILABLE" = true ]; then
    echo "Installing PyTorch with CUDA support..."
    $PIP_CMD install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
else
    echo "Installing PyTorch for CPU..."
    $PIP_CMD install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

# Install other requirements
echo "📚 Installing dependencies..."
$PIP_CMD install -r requirements.txt

# Install the package
echo "🔧 Installing DLLM Sampling System..."
$PIP_CMD install -e .

# Run basic test
echo "🧪 Running basic tests..."
python3 -c "
import torch
from dllm_sampling import DLLMSampler, SamplingConfig, TokenCache
print('✓ Core imports successful')

# Test basic functionality
config = SamplingConfig()
sampler = DLLMSampler(config)
logits = torch.randn(2, 1000)
samples = sampler.sample(logits)
print(f'✓ Basic sampling test: {samples.shape}')

# Test cache
cache = TokenCache()
cache.put('test', 'value')
result = cache.get('test')
assert result == 'value'
print('✓ Cache test successful')

print('🎉 All tests passed!')
"

# Create config file
echo "⚙️ Creating default configuration..."
if [ ! -f "config.yaml" ]; then
    cp config_template.yaml config.yaml
    echo "✓ Default config created as config.yaml"
else
    echo "ℹ️ config.yaml already exists, skipping"
fi

# Show next steps
echo ""
echo "🎉 Installation Complete!"
echo "========================"
echo ""
echo "📝 Next steps:"
echo "1. Activate virtual environment (if created): source dllm_env/bin/activate"
echo "2. Edit config.yaml for your use case"
echo "3. Test with: python test_simple.py"
echo "4. Run benchmark: python benchmark.py"
echo "5. Start server: dllm-server serve --config config.yaml"
echo ""
echo "📖 Documentation:"
echo "- README.md for usage guide"
echo "- config_template.yaml for configuration options"
echo "- examples.py for code examples"
echo ""
echo "🚀 Happy accelerating!"