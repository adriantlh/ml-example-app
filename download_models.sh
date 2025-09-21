#!/bin/bash

# YOLOv9 Model Download Script
# This script downloads YOLOv9 models needed for the ML example app

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
MODELS_DIR="backend/models"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Model URLs - MIT/Apache 2.0 Licensed models only (commercial use friendly)
declare -A MODELS=(
    # YOLOv9 MIT License version (using WongKinYiu/YOLO which redirects to MultimediaTechLab)
    ["yolov9-mit.pt"]="https://github.com/WongKinYiu/YOLO/releases/download/v1.0/yolov9c.pt"

    # YOLOX (Apache 2.0 License) - Commercial friendly - official repository
    ["yolox_s.pth"]="https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.pth"

    # YOLOv9 ONNX MIT version - properly licensed from ibaiGorordo/ONNX-YOLOv9-MIT-Object-Detection
    ["yolov9-mit.onnx"]="https://github.com/ibaiGorordo/ONNX-YOLOv9-MIT-Object-Detection/releases/download/0.1.0/v9-s_mit.onnx"
)

# Alternative URLs in case primary ones fail (MIT/Apache 2.0 only)
declare -A ALT_MODELS=(
    # Alternative YOLOv9 ONNX models (medium and compact versions)
    ["yolov9-mit.onnx"]="https://github.com/ibaiGorordo/ONNX-YOLOv9-MIT-Object-Detection/releases/download/0.1.0/v9-m_mit.onnx"
)

print_header() {
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}    YOLOv9 Model Download Script${NC}"
    echo -e "${BLUE}========================================${NC}"
    echo
}

print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_dependencies() {
    print_info "Checking dependencies..."

    local missing_deps=()

    # Check for required tools
    if ! command -v curl &> /dev/null && ! command -v wget &> /dev/null; then
        missing_deps+=("curl or wget")
    fi

    if ! command -v sha256sum &> /dev/null; then
        print_warning "sha256sum not found. File integrity checks will be skipped."
    fi

    if [ ${#missing_deps[@]} -gt 0 ]; then
        print_error "Missing required dependencies: ${missing_deps[*]}"
        print_info "Please install the missing dependencies and try again."
        exit 1
    fi

    print_success "All dependencies found"
}

create_models_dir() {
    print_info "Creating models directory..."

    cd "$SCRIPT_DIR"

    if [ ! -d "$MODELS_DIR" ]; then
        mkdir -p "$MODELS_DIR"
        print_success "Created directory: $MODELS_DIR"
    else
        print_info "Directory already exists: $MODELS_DIR"
    fi
}

download_file() {
    local url="$1"
    local output_file="$2"
    local description="$3"

    print_info "Downloading $description..."
    print_info "URL: $url"
    print_info "Output: $output_file"

    # Try curl first, then wget
    if command -v curl &> /dev/null; then
        if curl -L -f --progress-bar -o "$output_file" "$url"; then
            return 0
        else
            return 1
        fi
    elif command -v wget &> /dev/null; then
        if wget --progress=bar:force:noscroll -O "$output_file" "$url"; then
            return 0
        else
            return 1
        fi
    else
        print_error "Neither curl nor wget is available"
        return 1
    fi
}

verify_file() {
    local file="$1"
    local min_size="$2"

    if [ ! -f "$file" ]; then
        print_error "File not found: $file"
        return 1
    fi

    local file_size=$(stat -f%z "$file" 2>/dev/null || stat -c%s "$file" 2>/dev/null || echo "0")

    if [ "$file_size" -lt "$min_size" ]; then
        print_error "File $file is too small ($file_size bytes). Expected at least $min_size bytes."
        return 1
    fi

    print_success "File $file verified (${file_size} bytes)"
    return 0
}

download_model() {
    local model_name="$1"
    local url="$2"
    local alt_url="$3"
    local output_path="$MODELS_DIR/$model_name"

    # Skip if file already exists and is valid
    if [ -f "$output_path" ]; then
        print_info "File $model_name already exists. Checking validity..."

        # Set minimum file sizes (in bytes)
        local min_size=1000000  # 1MB minimum
        if [[ "$model_name" == *".onnx" ]]; then
            min_size=25000000  # 25MB for ONNX files
        elif [[ "$model_name" == *".pt" ]]; then
            min_size=10000000  # 10MB for PyTorch files
        fi

        if verify_file "$output_path" "$min_size"; then
            print_success "$model_name is already downloaded and valid"
            return 0
        else
            print_warning "Existing file is invalid. Re-downloading..."
            rm -f "$output_path"
        fi
    fi

    # Try primary URL
    print_info "Downloading $model_name..."
    if download_file "$url" "$output_path" "$model_name"; then
        print_success "Successfully downloaded $model_name"
        return 0
    fi

    # Try alternative URL if available
    if [ -n "$alt_url" ]; then
        print_warning "Primary download failed. Trying alternative URL..."
        if download_file "$alt_url" "$output_path" "$model_name (alternative)"; then
            print_success "Successfully downloaded $model_name from alternative source"
            return 0
        fi
    fi

    print_error "Failed to download $model_name"
    return 1
}

main() {
    print_header

    # Check if script is run from correct directory
    if [ ! -f "backend/main.py" ]; then
        print_error "This script should be run from the project root directory"
        print_info "Current directory: $(pwd)"
        print_info "Expected structure: backend/main.py should exist"
        exit 1
    fi

    check_dependencies
    create_models_dir

    # Track download results
    local success_count=0
    local total_count=${#MODELS[@]}

    # Download each model
    for model_name in "${!MODELS[@]}"; do
        url="${MODELS[$model_name]}"
        alt_url="${ALT_MODELS[$model_name]:-}"

        if download_model "$model_name" "$url" "$alt_url"; then
            ((success_count++))
        fi
        echo  # Add spacing between downloads
    done

    # Print summary
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}           Download Summary${NC}"
    echo -e "${BLUE}========================================${NC}"

    if [ "$success_count" -eq "$total_count" ]; then
        print_success "All models downloaded successfully! ($success_count/$total_count)"
        echo
        print_info "Models are now available in: $MODELS_DIR"
        print_info "You can now run the ML application."
    elif [ "$success_count" -gt 0 ]; then
        print_warning "Some models downloaded successfully ($success_count/$total_count)"
        print_info "You may need to manually download the missing models."
    else
        print_error "No models were downloaded successfully"
        print_info "Please check your internet connection and try again."
        exit 1
    fi

    # Show final directory contents
    echo
    print_info "Current models directory contents:"
    ls -lh "$MODELS_DIR" || true
}

# Show help if requested
if [ "$1" = "--help" ] || [ "$1" = "-h" ]; then
    echo "YOLOv9 Model Download Script"
    echo
    echo "Usage: $0 [OPTIONS]"
    echo
    echo "This script downloads MIT/Apache 2.0 licensed models for commercial use:"
    echo "  - yolov9-mit.pt (YOLOv9 MIT License version - commercial friendly)"
    echo "  - yolov9-mit.onnx (YOLOv9 ONNX MIT License version - commercial friendly)"
    echo "  - yolox_s.pth (YOLOX model, Apache 2.0 license - commercial friendly from official repo)"
    echo
    echo "LICENSING NOTE:"
    echo "  - All models use permissive licenses (MIT/Apache 2.0)"
    echo "  - Safe for commercial use without licensing restrictions"
    echo "  - No AGPL licensed models included to avoid license conflicts"
    echo
    echo "Options:"
    echo "  -h, --help    Show this help message"
    echo
    echo "The models will be downloaded to: backend/models/"
    echo
    exit 0
fi

# Run main function
main "$@"