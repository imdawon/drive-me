#!/bin/bash

# Picar-X RL Setup Script using UV
# This script automatically sets up the environment and installs dependencies

set -e  # Exit on error

echo "🚗 Setting up Picar-X RL Environment..."
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Check if UV is installed
check_uv() {
    if ! command -v uv &> /dev/null; then
        echo -e "${RED}❌ UV is not installed${NC}"
        echo ""
        echo "UV is a fast Python package manager. Install it with:"
        echo ""
        echo "  # On macOS/Linux:"
        echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"
        echo ""
        echo "  # Or with pip:"
        echo "  pip install uv"
        echo ""
        echo -e "${YELLOW}For more info: https://github.com/astral-sh/uv${NC}"
        exit 1
    fi
    echo -e "${GREEN}✓ UV found${NC}"
}

# Check Python version
check_python() {
    if ! command -v python3 &> /dev/null; then
        echo -e "${RED}❌ Python 3 is not installed${NC}"
        exit 1
    fi
    
    PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
    echo -e "${GREEN}✓ Python version: $PYTHON_VERSION${NC}"
    
    # Check if version is 3.8 or higher
    REQUIRED_VERSION="3.8"
    if ! python3 -c "import sys; exit(0 if sys.version_info >= (3, 8) else 1)"; then
        echo -e "${RED}❌ Python 3.8 or higher is required${NC}"
        exit 1
    fi
}

# Create virtual environment
create_venv() {
    echo ""
    echo -e "${BLUE}📦 Creating virtual environment...${NC}"
    
    if [ -d ".venv" ]; then
        echo -e "${YELLOW}⚠ Virtual environment already exists. Removing old one...${NC}"
        rm -rf .venv
    fi
    
    uv venv
    echo -e "${GREEN}✓ Virtual environment created${NC}"
}

# Install dependencies
install_deps() {
    echo ""
    echo -e "${BLUE}📥 Installing dependencies...${NC}"
    
    # Check if requirements.txt exists
    if [ ! -f "requirements.txt" ]; then
        echo -e "${RED}❌ requirements.txt not found${NC}"
        exit 1
    fi
    
    # Install using UV (faster than pip)
    uv pip install -r requirements.txt
    
    echo -e "${GREEN}✓ Dependencies installed${NC}"
}

# Install PyTorch with CUDA support (for RTX 3090)
install_pytorch() {
    echo ""
    echo -e "${BLUE}🔥 Installing PyTorch with CUDA support...${NC}"
    
    # Detect CUDA version
    if command -v nvcc &> /dev/null; then
        CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $6}' | cut -c2-)
        echo "Detected CUDA version: $CUDA_VERSION"
        
        # Install appropriate PyTorch version
        if [[ "$CUDA_VERSION" == 12.* ]]; then
            echo "Installing PyTorch for CUDA 12.x..."
            uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
        elif [[ "$CUDA_VERSION" == 11.8* ]]; then
            echo "Installing PyTorch for CUDA 11.8..."
            uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
        else
            echo -e "${YELLOW}⚠ Unknown CUDA version, installing CPU-only PyTorch${NC}"
            uv pip install torch torchvision
        fi
    else
        echo -e "${YELLOW}⚠ CUDA not detected, installing CPU-only PyTorch${NC}"
        uv pip install torch torchvision
    fi
    
    echo -e "${GREEN}✓ PyTorch installed${NC}"
}

# Verify installation
verify_install() {
    echo ""
    echo -e "${BLUE}🔍 Verifying installation...${NC}"
    
    # Activate venv and check imports
    source .venv/bin/activate
    
    python3 -c "import genesis; print('✓ Genesis installed')" || {
        echo -e "${RED}❌ Genesis import failed${NC}"
        exit 1
    }
    
    python3 -c "import torch; print(f'✓ PyTorch installed (CUDA: {torch.cuda.is_available()})')" || {
        echo -e "${RED}❌ PyTorch import failed${NC}"
        exit 1
    }
    
    python3 -c "import numpy; print('✓ NumPy installed')" || {
        echo -e "${RED}❌ NumPy import failed${NC}"
        exit 1
    }
    
    echo -e "${GREEN}✓ All dependencies verified${NC}"
}

# Create activation script
create_activate_script() {
    echo ""
    echo -e "${BLUE}📝 Creating helper scripts...${NC}"
    
    # Create activate.sh
    cat > activate.sh << 'EOF'
#!/bin/bash
# Activate the virtual environment

if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
    echo "✓ Virtual environment activated"
    echo ""
    echo "You can now run:"
    echo "  python train.py"
else
    echo "❌ Virtual environment not found. Run ./setup.sh first"
fi
EOF
    chmod +x activate.sh
    
    # Create run.sh for easy training
    cat > run.sh << 'EOF'
#!/bin/bash
# Run training with virtual environment

if [ ! -f ".venv/bin/activate" ]; then
    echo "❌ Virtual environment not found. Run ./setup.sh first"
    exit 1
fi

source .venv/bin/activate
echo "🚀 Starting training..."
python train.py "$@"
EOF
    chmod +x run.sh
    
    echo -e "${GREEN}✓ Helper scripts created${NC}"
}

# Print success message
print_success() {
    echo ""
    echo -e "${GREEN}🎉 Setup complete!${NC}"
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    echo "Next steps:"
    echo ""
    echo "  1. Activate the environment:"
    echo "     source .venv/bin/activate"
    echo ""
    echo "  2. Or use the helper script:"
    echo "     ./run.sh"
    echo ""
    echo "  3. Start training:"
    echo "     python train.py"
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    echo "Available commands:"
    echo "  ./run.sh              - Run training"
    echo "  ./activate.sh         - Activate venv"
    echo "  ./setup.sh --train    - Setup and train immediately"
    echo ""
}

# Main setup flow
main() {
    check_uv
    check_python
    create_venv
    install_deps
    install_pytorch
    verify_install
    create_activate_script
    print_success
    
    # Optional: Run training immediately
    if [[ "$1" == "--train" ]] || [[ "$1" == "-t" ]]; then
        echo -e "${BLUE}🚀 Starting training immediately...${NC}"
        source .venv/bin/activate
        python train.py
    fi
}

# Run main function
main "$@"
