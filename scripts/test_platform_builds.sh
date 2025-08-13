#!/bin/bash
# Platform Build Testing Script for NSeekFS
# Tests cross-platform compatibility and wheel building

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

log_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

log_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

log_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Detect platform
detect_platform() {
    case "$(uname -s)" in
        Linux*)     PLATFORM="linux" ;;
        Darwin*)    PLATFORM="macos" ;;
        CYGWIN*|MINGW*|MSYS*) PLATFORM="windows" ;;
        *)          PLATFORM="unknown" ;;
    esac
    
    ARCH=$(uname -m)
    
    log_info "Platform: $PLATFORM"
    log_info "Architecture: $ARCH"
}

# Check dependencies
check_dependencies() {
    log_info "Checking dependencies..."
    
    # Check Python
    if command -v python &> /dev/null; then
        PYTHON_VERSION=$(python --version 2>&1)
        log_success "Python: $PYTHON_VERSION"
    else
        log_error "Python not found"
        exit 1
    fi
    
    # Check Rust
    if command -v rustc &> /dev/null; then
        RUST_VERSION=$(rustc --version)
        log_success "Rust: $RUST_VERSION"
    else
        log_error "Rust not found"
        log_info "Install with: curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh"
        exit 1
    fi
    
    # Check Cargo
    if command -v cargo &> /dev/null; then
        CARGO_VERSION=$(cargo --version)
        log_success "Cargo: $CARGO_VERSION"
    else
        log_error "Cargo not found"
        exit 1
    fi
    
    # Check Maturin
    if command -v maturin &> /dev/null; then
        MATURIN_VERSION=$(maturin --version)
        log_success "Maturin: $MATURIN_VERSION"
    else
        log_warning "Maturin not found, installing..."
        python -m pip install maturin
        log_success "Maturin installed"
    fi
}

# Platform-specific optimizations
setup_platform_optimizations() {
    log_info "Setting up platform optimizations..."
    
    case $PLATFORM in
        "linux")
            log_info "Linux optimizations:"
            echo "  - SIMD