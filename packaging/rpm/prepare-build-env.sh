#!/bin/bash
set -euo pipefail

# Keep distro dependencies separate from the vendor SDK and Python inputs.
. /etc/os-release
json_package=json-devel
torch_index=()
case "${ID}" in
    rocky|rhel|almalinux|centos)
        dnf install -y epel-release dnf-plugins-core
        dnf config-manager --set-enabled crb 2>/dev/null || \
            dnf config-manager --set-enabled powertools
        python_packages='torch==2.8.0 triton==3.4.0 pybind11==3.1.0'
        ;;
    openEuler)
        json_package=nlohmann-json-devel
        python_packages='torch==2.8.0+cpu triton==3.4.0 pybind11==3.1.0'
        torch_index=(--extra-index-url https://download.pytorch.org/whl/cpu)
        # The 24.03 image has no dist macro by default.
        printf '%%dist .oe%s\n' "${VERSION_ID//./}" > /etc/rpm/macros.libtriton-jit-dist
        ;;
    fedora)
        # Python 3.14 has no torch 2.8 or Triton 3.4 wheels.
        python_packages='torch==2.9.0+cpu triton==3.5.0 pybind11==3.1.0'
        torch_index=(--extra-index-url https://download.pytorch.org/whl/cpu)
        if [ "${BACKEND}" = CUDA ]; then
            curl -fsSL -o /etc/yum.repos.d/cuda-fedora43.repo \
                https://developer.download.nvidia.com/compute/cuda/repos/fedora43/x86_64/cuda-fedora43.repo
            dnf install -y cuda-nvcc-13-2 cuda-cudart-devel-13-2 \
                cuda-driver-devel-13-2 cuda-cccl-13-2
        fi
        ;;
    *)
        echo "Unsupported RPM build distribution: ${ID}" >&2
        exit 1
        ;;
esac

dnf install -y rpm-build rpmdevtools cmake ninja-build gcc-c++ fmt-devel \
    "${json_package}" python3-devel python3-pip python3-packaging patchelf git \
    findutils tar gzip
dnf clean all

# The new distro builds need only libtorch's C++ API. A CPU wheel avoids
# pulling a second CUDA toolkit into the build; the library uses libcuda.
# Explicit overrides retain the vendor-specific Python environment contract.
if [ -n "${PYTHON_PACKAGES:-}" ]; then
    python_packages=${PYTHON_PACKAGES}
    torch_index=()
fi
read -r -a packages <<< "${python_packages}"
python3 -m pip install --no-cache-dir --index-url "${PYTHON_INDEX}" \
    "${torch_index[@]}" "${packages[@]}"
