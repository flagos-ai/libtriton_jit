# Vendor flavor and the matching CMake backend policy. Override at build
# time for other backends, e.g.:
#   rpmbuild --define "vendor mthreads" --define "backend MUSA" ...
# The source already supports the MUSA (Moore Threads) and MACA (MetaX)
# backends; only the build environment (SDK + torch) differs per vendor.
%{!?vendor_flavor: %global vendor_flavor nvidia}
%{!?backend: %global backend CUDA}

# CUDA and PyTorch are supplied by the selected vendor environment, not the
# RPM database. Keep automatic requirements for distro libraries and Python.
%global __requires_exclude ^(libcuda[.]so[.]1|libtorch(_cpu|_cuda)?[.]so|libc10[.]so)[(][)][(]64bit[)]$

Name:           libtriton-jit-%{vendor_flavor}
Version:        0.1.0
Release:        3%{?dist}
Summary:        Triton JIT runtime library

License:        MIT
URL:            https://github.com/flagos-ai/libtriton_jit
Source0:        libtriton-jit-%{version}.tar.gz

# Minimal BuildRequires - CUDA, PyTorch, Triton are container-provided
BuildRequires:  cmake
BuildRequires:  ninja-build
BuildRequires:  gcc-c++
BuildRequires:  fmt-devel >= 8.1.1
BuildRequires:  json-devel >= 3.10.5
BuildRequires:  python3-devel
BuildRequires:  patchelf

%description
libtriton_jit is a C++ library providing Triton JIT runtime functionality.
It enables just-in-time compilation of Triton kernels for GPU acceleration.

%package devel
Summary:        Development files for %{name}
Requires:       %{name}%{?_isa} = %{version}-%{release}
Requires:       fmt-devel >= 8.1.1

%description devel
Development files (headers and CMake configs) for libtriton_jit.

%prep
%autosetup -n libtriton-jit-%{version}

%build
# Ensure pip-installed packages (torch, triton, pybind11) are visible
PY3_VER=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
export PYTHONPATH=$(python3 -c "import site; print(':'.join(site.getsitepackages()))"):/usr/local/lib/python${PY3_VER}/site-packages:/usr/local/lib64/python${PY3_VER}/site-packages
export PATH=/usr/local/bin:$PATH
# Find torch cmake path without importing torch (avoids cuDNN dependency at configure time)
TORCH_CMAKE_PATH=$(python3 -c "import importlib.util; s=importlib.util.find_spec('torch'); import os; print(os.path.join(os.path.dirname(s.origin), 'share', 'cmake'))")
%cmake \
    -GNinja \
    -DCMAKE_BUILD_TYPE=Release \
    -DBACKEND=%{backend} \
    -DCMAKE_CUDA_FLAGS="-Xcompiler -fPIE" \
    -DTorch_ROOT="${TORCH_CMAKE_PATH}" \
    -DFETCHCONTENT_FULLY_DISCONNECTED=ON \
    -DFETCHCONTENT_QUIET=OFF \
    -DTRITON_JIT_USE_EXTERNAL_JSON=ON \
    -DTRITON_JIT_USE_EXTERNAL_FMTLIB=ON \
    -DTRITON_JIT_USE_EXTERNAL_PYBIND11=ON \
    -DTRITON_JIT_BUILD_OPERATORS=OFF \
    -DBUILD_TESTING=OFF \
    -DTRITON_JIT_INSTALL=ON

%cmake_build

%install
%cmake_install

# Fix RPATH
find %{buildroot}%{_libdir} -name "*.so*" -type f -exec patchelf --remove-rpath {} \; || true

%files
%license LICENSE
%doc README.md
%{_libdir}/libtriton_jit.so
%{_datadir}/triton_jit/scripts/*.py

%files devel
%{_includedir}/triton_jit/
%{_libdir}/cmake/TritonJIT/

%changelog
* Fri Aug 07 2026 The FlagOS Contributors <contact@flagos.io> - 0.1.0-3
- Keep automatic distro runtime dependencies while filtering vendor libraries
- Pin the validated Python 3.9, PyTorch 2.8, and Triton 3.4 build stack

* Wed Aug 05 2026 The FlagOS Contributors <contact@flagos.io> - 0.1.0-2
- Build against EPEL fmt-devel and json-devel instead of vendoring them
- Drop bundled fmt files and require fmt-devel from the devel package

* Sun Feb 08 2026 FlagTree Project <contact@flagos.io> - 0.1.0-1
- Initial RPM release
