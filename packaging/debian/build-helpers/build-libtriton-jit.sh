#!/bin/bash
set -e

# libtriton-jit Debian package build script
# Usage: ./build-libtriton-jit.sh [--base-image IMAGE] [--output-dir DIR]

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
BASE_IMAGE="nvidia/cuda:12.4.0-devel-ubuntu22.04"
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

# Get the project root directory (3 levels up from this script)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

# Default output dir
if [ -z "$OUTPUT_DIR" ]; then
    OUTPUT_DIR="${PROJECT_ROOT}/debian-packages"
fi

log_info "Building libtriton-jit Debian package"
log_info "Base image: ${BASE_IMAGE}"
log_info "Output directory: ${OUTPUT_DIR}"
log_info "Project root: ${PROJECT_ROOT}"

# Create output directory
mkdir -p "${OUTPUT_DIR}"

DOCKERFILE="${SCRIPT_DIR}/../Dockerfile.deb"
if [ ! -f "$DOCKERFILE" ]; then
    log_error "Dockerfile not found: $DOCKERFILE"
    exit 1
fi

IMAGE_TAG="libtriton-jit-deb-builder"

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

# Extract .deb files from the output stage
log_step "Extracting .deb packages to: ${OUTPUT_DIR}"
CONTAINER_NAME="libtriton-jit-deb-tmp-$$"
if docker create --name "${CONTAINER_NAME}" "${IMAGE_TAG}" 2>/dev/null; then
    docker cp "${CONTAINER_NAME}:/output/." "${OUTPUT_DIR}/" || log_warn "No .deb files found"
    docker rm "${CONTAINER_NAME}" >/dev/null
else
    log_error "Failed to create temporary container"
    exit 1
fi

# Verify packages were created
if ls "${OUTPUT_DIR}"/*.deb 1>/dev/null 2>&1; then
    echo ""
    log_info "Packages built successfully:"
    echo ""
    ls -lh "${OUTPUT_DIR}"/*.deb | while read -r line; do
        echo "  $line"
    done
    echo ""

    # Run lintian if available
    if command -v lintian >/dev/null 2>&1; then
        log_step "Running lintian checks..."
        for deb in "${OUTPUT_DIR}"/*.deb; do
            [ -f "$deb" ] || continue
            echo "Checking $(basename "$deb")..."
            lintian "$deb" 2>&1 || log_warn "Lintian found issues in $(basename "$deb") (non-fatal)"
            echo ""
        done
    fi

    log_info "Build complete! Packages in: ${OUTPUT_DIR}/"
else
    log_error "No .deb files were created"
    exit 1
fi
