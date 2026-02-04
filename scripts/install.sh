#!/bin/bash
# Installation script for PAPO

set -e

echo "=== PAPO Installation Script ==="
echo "This script prepares the Python env for PAPO with proper dependency ordering and compatibility fixes."
echo ""

# Check conda environment
if [[ "$CONDA_DEFAULT_ENV" != "" ]]; then
    echo "✅ Detected conda environment: $CONDA_DEFAULT_ENV"
elif [[ "$VIRTUAL_ENV" != "" ]]; then
    echo "✅ Detected virtual environment: $(basename $VIRTUAL_ENV)"
else
    echo "⚠️  Warning: No virtual environment detected."
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Function to get Python version for wheel compatibility
get_python_wheel_version() {
    local py_ver=$(python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    case $py_ver in
        "3.10") echo "cp310-cp310" ;;
        "3.11") echo "cp311-cp311" ;;
        "3.12") echo "cp312-cp312" ;;
        *) echo "unsupported" ;;
    esac
}

echo ""
echo "=== Environment Detection ==="
PYTHON_VERSION=$(python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
PYTHON_WHEEL_VER=$(get_python_wheel_version)

echo "Python version: $PYTHON_VERSION"

if [[ "$PYTHON_WHEEL_VER" == "unsupported" ]]; then
    echo "❌ Unsupported Python version: $PYTHON_VERSION"
    echo "   Supported versions: 3.10, 3.11, 3.12"
    exit 1
fi

echo ""
echo "Step 1/6: Installing build dependencies..."
pip install --upgrade pip setuptools wheel ninja packaging

echo ""
echo "Step 2/6: Installing base package (safe dependencies)..."
pip install -e .

echo ""
echo "Step 3/6: Installing PyTorch with CUDA 12.4 support..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Verify PyTorch installation and get exact version
echo "Verifying PyTorch installation..."
PYTORCH_VERSION=$(python -c "import torch; print(torch.__version__.split('+')[0])")
CUDA_VERSION=$(python -c "import torch; print(torch.version.cuda)")
echo "✅ PyTorch version: $PYTORCH_VERSION"
echo "✅ CUDA version: $CUDA_VERSION"

echo ""
echo "Step 4/6: Installing compatible flash-attn..."

# Clean any existing flash-attn installation
pip uninstall flash-attn -y 2>/dev/null || true
pip cache purge

# Install flash-attn with proper compatibility
install_flash_attn() {
    local success=false
    
    # Strategy 1: Try precompiled wheel for PyTorch 2.6.0
    if [[ "$PYTORCH_VERSION" == "2.6.0" ]]; then
        echo "Attempting precompiled wheel for PyTorch 2.6.0..."
        local wheel_url="https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.0.7/flash_attn-2.7.4.post1+cu124torch2.6-${PYTHON_WHEEL_VER}-linux_x86_64.whl"
        
        if pip install "$wheel_url"; then
            echo "✅ Installed flash-attn 2.7.4.post1 (precompiled for PyTorch 2.6.0)"
            success=true
        else
            echo "⚠️  Precompiled wheel failed, trying alternative version..."
            wheel_url="https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.0.7/flash_attn-2.6.3+cu124torch2.6-${PYTHON_WHEEL_VER}-linux_x86_64.whl"
            if pip install "$wheel_url"; then
                echo "✅ Installed flash-attn 2.6.3 (precompiled for PyTorch 2.6.0)"
                success=true
            fi
        fi
    fi
    
    # Strategy 2: Try standard installation for other PyTorch versions
    if [[ "$success" == false ]]; then
        echo "Trying standard flash-attn installation..."
        if command -v nvcc &> /dev/null; then
            echo "✅ Found nvcc, attempting compilation..."
            if pip install flash-attn --no-build-isolation --no-cache-dir; then
                echo "✅ Compiled and installed flash-attn from source"
                success=true
            fi
        else
            echo "⚠️  nvcc not found, trying without build isolation..."
            if pip install flash-attn --no-build-isolation --no-deps; then
                echo "✅ Installed flash-attn (may be CPU-only)"
                success=true
            fi
        fi
    fi
    
    # Strategy 3: Disable flash-attn if all else fails
    if [[ "$success" == false ]]; then
        echo "❌ Could not install flash-attn. Disabling flash attention..."
        export FLASH_ATTENTION_FORCE_DISABLE=1
        echo "export FLASH_ATTENTION_FORCE_DISABLE=1" >> ~/.bashrc
        echo "⚠️  Flash attention disabled. Models will work but may be slower."
    fi
}

install_flash_attn

echo ""
echo "Step 5/6: Installing other packages..."
# Install liger-kernel
if pip install liger-kernel; then
    echo "✅ Installed liger-kernel"
else
    echo "⚠️  Could not install liger-kernel, continuing without it..."
fi

# Check for CUDA development tools before installing flashinfer-python
echo "Checking for CUDA development tools..."
if command -v nvcc &> /dev/null; then
    echo "✅ Found nvcc, attempting to install flashinfer-python..."
    if pip install flashinfer-python --extra-index-url https://flashinfer.ai/whl/cu124/torch2.6/; then
        echo "✅ Installed flashinfer-python"
    else
        echo "⚠️  Could not install flashinfer-python, continuing without it..."
    fi
else
    echo "⚠️  nvcc not found in PATH. Skipping flashinfer-python installation."
    echo "   To install flashinfer-python later, ensure CUDA toolkit is properly installed:"
    echo "   export CUDA_HOME=/usr/local/cuda"
    echo "   export PATH=\$CUDA_HOME/bin:\$PATH"
fi

echo ""
echo "Step 6/6: Installing vLLM..."
if pip install vllm==0.8.4 --extra-index-url https://flashinfer.ai/whl/cu124/torch2.6/; then
    echo "✅ Installed vLLM from flashinfer"
else
    echo "⚠️  Could not install vLLM from flashinfer, trying PyPI..."
    if pip install vllm==0.8.4; then
        echo "✅ Installed vLLM from PyPI"
    else
        echo "❌ Failed to install vLLM"
    fi
fi

echo ""
echo "🎉 Installation completed!"
echo ""
echo "Verifying installation..."

# Create a comprehensive verification script
cat > /tmp/verify_papo.py << 'EOF'
import sys
import traceback

def test_transformers():
    try:
        import transformers
        print('✅ Transformers imports successfully')
        return True
    except Exception as e:
        print(f'❌ Transformers import failed: {e}')
        traceback.print_exc()
        return False

try:
    import torch
    print('✅ PyTorch imported successfully!')
    print('   PyTorch version: ' + torch.__version__)
    print('   CUDA available: ' + str(torch.cuda.is_available()))
    
    # Test transformers specifically (this was the main issue)
    transformers_ok = test_transformers()
    
    # Test optional components
    components = []
    try:
        import vllm
        components.append('✅ vLLM')
    except ImportError:
        components.append('⚠️  vLLM not available')
    
    try:
        import liger_kernel
        components.append('✅ Liger Kernel')
    except ImportError:
        components.append('⚠️  Liger Kernel not available')
        
    try:
        import flash_attn
        components.append('✅ Flash Attention v' + flash_attn.__version__)
    except ImportError:
        components.append('⚠️  Flash Attention not available')
    
    for component in components:
        print('   ' + component)
        
    if transformers_ok:
        print()
        print('🎉 Core functionality working - Can safely start upgrading packages for Qwen3')
    else:
        print()
        print('❌ Critical error detected - check transformers installation')
        exit
        
except Exception as e:
    print('❌ Error: ' + str(e))
EOF

# Run the verification script
python /tmp/verify_papo.py

# Clean up
rm /tmp/verify_papo.py

echo ""
echo "Install lazy-loader if not already installed:"
pip install lazy_loader

echo ""
echo "Upgrading transformers version to 4.57.1:"
pip install transformers==4.57.1

echo ""
echo " Installing PyTorch stack (CUDA 12.8 / cu128)..."
pip uninstall -y torch torchvision torchaudio torchdata
pip install \
  torch==2.8.0 \
  torchvision==0.23.0 \
  torchaudio==2.8.0 \
  torchdata==0.10.0 \
  --index-url https://download.pytorch.org/whl/cu128

pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abiFALSE-${PYTHON_WHEEL_VER}-linux_x86_64.whl

echo ""
echo " Installing vllm for PyTorch 2.8.0..."
pip install vllm==0.11.0

echo ""
echo "🎉 Installation completed!"
echo ""
echo "Verifying installation..."

# Create a comprehensive verification script
cat > /tmp/verify_papo.py << 'EOF'
import sys
import traceback

def test_transformers():
    try:
        import transformers
        print('✅ Transformers imports successfully')
        return True
    except Exception as e:
        print(f'❌ Transformers import failed: {e}')
        traceback.print_exc()
        return False

try:
    import torch
    print('✅ PyTorch imported successfully!')
    print('   PyTorch version: ' + torch.__version__)
    print('   CUDA available: ' + str(torch.cuda.is_available()))
    
    # Test transformers specifically (this was the main issue)
    transformers_ok = test_transformers()
    
    # Test optional components
    components = []
    try:
        import vllm
        components.append('✅ vLLM')
    except ImportError:
        components.append('⚠️  vLLM not available')
    
    try:
        import liger_kernel
        components.append('✅ Liger Kernel')
    except ImportError:
        components.append('⚠️  Liger Kernel not available')
        
    try:
        import flash_attn
        components.append('✅ Flash Attention v' + flash_attn.__version__)
    except ImportError:
        components.append('⚠️  Flash Attention not available')
    
    for component in components:
        print('   ' + component)
        
    if transformers_ok:
        print()
        print('🎉 Core functionality working - PAPO for Qwen3 should would properly now!')
    else:
        print()
        print('❌ Critical error detected - check transformers installation')
        exit
        
except Exception as e:
    print('❌ Error: ' + str(e))
EOF

# Run the verification script
python /tmp/verify_papo.py

# Clean up
rm /tmp/verify_papo.py

echo ""
echo "Installation summary:"
echo "✅ Core PAPO packages installed"
echo "✅ PyTorch with CUDA support installed"
echo ""
echo "Additional options:"
echo "  Development tools: pip install -e .[dev]"
echo "  Documentation: pip install -e .[docs]"
echo ""
echo "If you need flash-attn compilation, set up CUDA development environment:"
echo "  export CUDA_HOME=/usr/local/cuda (or your CUDA path)"
echo "  export PATH=\$CUDA_HOME/bin:\$PATH"
echo "  pip install flash-attn --no-cache-dir"