#!/bin/bash

# Quick Start Script for SeeAct Project
# This script helps you set up and verify the project

set -e  # Exit on error

echo "=========================================="
echo "SeeAct Project - Quick Start Script"
echo "=========================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}✓${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

# Step 1: Check Python version
echo "Step 1: Checking Python version..."
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

if [ "$PYTHON_MAJOR" -ge 3 ] && [ "$PYTHON_MINOR" -ge 8 ]; then
    print_status "Python $PYTHON_VERSION found (required: 3.8+)"
else
    print_error "Python 3.8+ required, found $PYTHON_VERSION"
    exit 1
fi
echo ""

# Step 2: Check if virtual environment exists
echo "Step 2: Checking virtual environment..."
if [ ! -d "venv" ]; then
    print_warning "Virtual environment not found. Creating..."
    python3 -m venv venv
    print_status "Virtual environment created"
else
    print_status "Virtual environment exists"
fi
echo ""

# Step 3: Activate virtual environment
echo "Step 3: Activating virtual environment..."
source venv/bin/activate
print_status "Virtual environment activated"
echo ""

# Step 4: Check/Install dependencies
echo "Step 4: Checking dependencies..."
if pip list | grep -q "torch"; then
    TORCH_VERSION=$(pip show torch | grep Version | awk '{print $2}')
    print_status "PyTorch $TORCH_VERSION installed"
else
    print_warning "PyTorch not found. Installing dependencies..."
    pip install --upgrade pip -q
    pip install -r requirements.txt
    print_status "Dependencies installed"
fi
echo ""

# Step 5: Run tests
echo "Step 5: Running tests..."
cd 01_AutoWeb_POC
if python tests/test_core.py > /tmp/test_output.txt 2>&1; then
    TEST_RESULT=$(grep "Test Results:" /tmp/test_output.txt)
    print_status "Tests passed: $TEST_RESULT"
else
    print_error "Tests failed. Check /tmp/test_output.txt"
    cat /tmp/test_output.txt
    exit 1
fi
cd ..
echo ""

# Step 6: Run mock demo
echo "Step 6: Running mock demo..."
cd 01_AutoWeb_POC
if python mock_demo.py > /tmp/mock_output.txt 2>&1; then
    MOCK_RESULT=$(grep -c "✓ SUCCESS" /tmp/mock_output.txt)
    print_status "Mock demo passed: $MOCK_RESULT/8 examples"
else
    print_error "Mock demo failed"
    exit 1
fi
cd ..
echo ""

# Step 7: Check model
echo "Step 7: Checking model..."
if [ -d "models/Qwen2-VL-2B-Instruct" ]; then
    if [ -f "models/Qwen2-VL-2B-Instruct/config.json" ]; then
        print_status "Model found at models/Qwen2-VL-2B-Instruct/"
        MODEL_READY=1
    else
        print_warning "Model directory exists but incomplete"
        MODEL_READY=0
    fi
else
    print_warning "Model not found at models/Qwen2-VL-2B-Instruct/"
    MODEL_READY=0
fi
echo ""

# Step 8: Check dataset
echo "Step 8: Checking dataset..."
if [ -d "data/Mind2Web" ]; then
    if ls data/Mind2Web/data/*.parquet 1> /dev/null 2>&1; then
        PARQUET_COUNT=$(ls -1 data/Mind2Web/data/*.parquet | wc -l)
        print_status "Dataset found: $PARQUET_COUNT parquet files"
        DATASET_READY=1
    else
        print_warning "Dataset directory exists but no parquet files"
        DATASET_READY=0
    fi
else
    print_warning "Dataset not found at data/Mind2Web/"
    DATASET_READY=0
fi
echo ""

# Step 9: Check GPU
echo "Step 9: Checking GPU availability..."
if command -v nvidia-smi &> /dev/null; then
    if nvidia-smi > /dev/null 2>&1; then
        GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
        print_status "GPU available: $GPU_NAME"
        GPU_AVAILABLE=1
    else
        print_warning "nvidia-smi found but GPU not accessible"
        GPU_AVAILABLE=0
    fi
else
    print_warning "No NVIDIA GPU found (will use CPU - slower)"
    GPU_AVAILABLE=0
fi
echo ""

# Final summary
echo "=========================================="
echo "Setup Summary"
echo "=========================================="
echo ""

if [ "$MODEL_READY" -eq 1 ] && [ "$DATASET_READY" -eq 1 ]; then
    print_status "✓ READY TO RUN FULL EXPERIMENTS!"
    echo ""
    echo "Next steps:"
    echo "  1. Activate venv: source venv/bin/activate"
    echo "  2. Run small test:"
    echo "     cd 01_AutoWeb_POC"
    echo "     python enhanced_eval.py --dataset ../data/Mind2Web --model ../models/Qwen2-VL-2B-Instruct --num-samples 10"
    echo ""
    echo "  3. Run full evaluation:"
    echo "     python enhanced_eval.py --dataset ../data/Mind2Web --model ../models/Qwen2-VL-2B-Instruct --num-samples 100"
elif [ "$MODEL_READY" -eq 0 ] && [ "$DATASET_READY" -eq 0 ]; then
    print_warning "⚠ NEED TO DOWNLOAD MODEL AND DATASET"
    echo ""
    echo "Next steps:"
    echo "  1. Download model:"
    echo "     pip install huggingface_hub"
    echo "     python -c \"from huggingface_hub import snapshot_download; snapshot_download(repo_id='Qwen/Qwen2-VL-2B-Instruct', cache_dir='./models')\""
    echo ""
    echo "  2. Download dataset:"
    echo "     Visit: https://github.com/OSU-NLP-Group/Mind2Web"
    echo "     Place in: data/Mind2Web/"
    echo ""
    echo "  3. Run this script again: ./quickstart.sh"
elif [ "$MODEL_READY" -eq 0 ]; then
    print_warning "⚠ NEED TO DOWNLOAD MODEL"
    echo ""
    echo "Download model:"
    echo "  pip install huggingface_hub"
    echo "  python -c \"from huggingface_hub import snapshot_download; snapshot_download(repo_id='Qwen/Qwen2-VL-2B-Instruct', cache_dir='./models')\""
else
    print_warning "⚠ NEED TO DOWNLOAD DATASET"
    echo ""
    echo "Download dataset:"
    echo "  Visit: https://github.com/OSU-NLP-Group/Mind2Web"
    echo "  Place in: data/Mind2Web/"
fi

echo ""
echo "=========================================="
echo "For detailed instructions, see:"
echo "  - SETUP_AND_EXECUTION_GUIDE.md"
echo "  - PROJECT_UPDATE_SUMMARY.md"
echo "  - VISUAL_GUIDE.md"
echo "=========================================="
