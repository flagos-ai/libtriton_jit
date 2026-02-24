#!/bin/bash
set -e

# libtriton-jit RPM package build script
# Usage: ./build-rpm.sh [--base-image IMAGE] [--output-dir DIR]

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }
log_step() { echo -e "${BLUE}[STEP]${NC} $1"; }

# Default values
BASE_IMAGE="nvidia/cuda:12.4.0-devel-rockylinux8"
OUTPUT_DIR=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --base-image)
            BASE_IMAGE="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--base-image IMAGE] [--output-dir DIR]"
            exit 1
            ;;
    esac
done

# Get the project root directory (2 levels up from this script)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Default output dir
if [ -z "$OUTPUT_DIR" ]; then
    OUTPUT_DIR="${PROJECT_ROOT}/rpm-packages"
fi

log_info "Building libtriton-jit RPM package"
log_info "Base image: ${BASE_IMAGE}"
log_info "Output directory: ${OUTPUT_DIR}"
log_info "Project root: ${PROJECT_ROOT}"

# Create output directory
mkdir -p "${OUTPUT_DIR}"

DOCKERFILE="${SCRIPT_DIR}/Dockerfile.rpm"
if [ ! -f "$DOCKERFILE" ]; then
    log_error "Dockerfile not found: $DOCKERFILE"
    exit 1
fi

IMAGE_TAG="libtriton-jit-rpm-builder"

# Build the Docker image (multi-stage, target output)
log_step "Building container image: ${IMAGE_TAG}"
if ! docker build \
    --build-arg BASE_IMAGE="${BASE_IMAGE}" \
    -f "${DOCKERFILE}" \
    --target output \
    -t "${IMAGE_TAG}" \
    "${PROJECT_ROOT}"; then
    log_error "Docker build failed"
    exit 1
fi

# Extract .rpm files from the output stage
log_step "Extracting .rpm packages to: ${OUTPUT_DIR}"
CONTAINER_NAME="libtriton-jit-rpm-tmp-$$"
if docker create --name "${CONTAINER_NAME}" "${IMAGE_TAG}" 2>/dev/null; then
    docker cp "${CONTAINER_NAME}:/output/." "${OUTPUT_DIR}/" || log_warn "No .rpm files found"
    docker rm "${CONTAINER_NAME}" >/dev/null
else
    log_error "Failed to create temporary container"
    exit 1
fi

# Verify packages were created
if find "${OUTPUT_DIR}" -name "*.rpm" | grep -q .; then
    echo ""
    log_info "Packages built successfully:"
    echo ""
    find "${OUTPUT_DIR}" -name "*.rpm" -exec ls -lh {} \; | while read -r line; do
        echo "  $line"
    done
    echo ""
    log_info "Build complete! Packages in: ${OUTPUT_DIR}/"
else
    log_error "No .rpm files were created"
    exit 1
fi
