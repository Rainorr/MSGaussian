#!/bin/bash
# GaussianLSS MindSpore - Linux Optimized Runner Script

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Project directory
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$PROJECT_DIR/venv_mindspore"

echo -e "${BLUE}GaussianLSS MindSpore - Linux Optimized${NC}"
echo "=================================================="

# Function to activate virtual environment
activate_venv() {
    if [ -f "$VENV_DIR/bin/activate" ]; then
        echo -e "${GREEN}Activating virtual environment...${NC}"
        source "$VENV_DIR/bin/activate"
    else
        echo -e "${RED}Virtual environment not found at $VENV_DIR${NC}"
        echo "Please run: python3.11 -m venv venv_mindspore"
        exit 1
    fi
}

# Function to run tests
run_test() {
    echo -e "${YELLOW}Running tests...${NC}"
    activate_venv
    cd "$PROJECT_DIR"
    python test.py
}

# Function to prepare data
prepare_data() {
    echo -e "${YELLOW}Preparing data...${NC}"
    activate_venv
    cd "$PROJECT_DIR"
    python scripts/prepare_data.py
}

# Function to start training
start_training() {
    local epochs=${1:-10}
    local log_level=${2:-INFO}
    
    echo -e "${YELLOW}Starting training (epochs: $epochs, log_level: $log_level)...${NC}"
    activate_venv
    cd "$PROJECT_DIR"
    python train.py --epochs "$epochs" --log-level "$log_level"
}

# Function to show help
show_help() {
    echo "Usage: $0 [COMMAND] [OPTIONS]"
    echo ""
    echo "Commands:"
    echo "  test                    Run all tests"
    echo "  prepare                 Prepare NuScenes data"
    echo "  train [epochs] [level]  Start training (default: 10 epochs, INFO level)"
    echo "  help                    Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 test                 # Run tests"
    echo "  $0 prepare              # Prepare data"
    echo "  $0 train                # Train with default settings"
    echo "  $0 train 20 DEBUG       # Train for 20 epochs with DEBUG logging"
    echo ""
}

# Main script logic
case "${1:-help}" in
    "test")
        run_test
        ;;
    "prepare")
        prepare_data
        ;;
    "train")
        start_training "${2:-10}" "${3:-INFO}"
        ;;
    "help"|"-h"|"--help")
        show_help
        ;;
    *)
        echo -e "${RED}Unknown command: $1${NC}"
        echo ""
        show_help
        exit 1
        ;;
esac

echo -e "${GREEN}Done!${NC}"