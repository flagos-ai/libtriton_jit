Name:           libtriton-jit
Version:        0.1.0
Release:        1%{?dist}
Summary:        Triton JIT runtime library

License:        MIT
# Note: bundled fmt (MIT with optional exception) is included in the devel subpackage
URL:            https://github.com/shiptux/libtriton_jit
Source0:        %{name}-%{version}.tar.gz

# Minimal BuildRequires - CUDA, PyTorch, Triton are container-provided
BuildRequires:  cmake
BuildRequires:  ninja-build
BuildRequires:  gcc-c++
BuildRequires:  python3-devel
BuildRequires:  patchelf

%description
libtriton_jit is a C++ library providing Triton JIT runtime functionality.
It enables just-in-time compilation of Triton kernels for GPU acceleration.

%package devel
Summary:        Development files for %{name}
Requires:       %{name}%{?_isa} = %{version}-%{release}
Conflicts:      fmt-devel

%description devel
Development files (headers and CMake configs) for libtriton_jit.

%prep
%autosetup -n %{name}-%{version}

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
    -DCMAKE_CUDA_FLAGS="-Xcompiler -fPIE" \
    -DTorch_ROOT="${TORCH_CMAKE_PATH}" \
    -DFETCHCONTENT_QUIET=OFF \
    -DTRITON_JIT_USE_EXTERNAL_JSON=OFF \
    -DTRITON_JIT_USE_EXTERNAL_FMTLIB=OFF \
    -DTRITON_JIT_USE_EXTERNAL_PYBIND11=ON \
    -DTRITON_JIT_BUILD_EXAMPLES=OFF \
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
%{_includedir}/fmt/
%{_libdir}/cmake/TritonJIT/
%{_libdir}/cmake/fmt/
%{_libdir}/libfmt.so*
%{_libdir}/pkgconfig/fmt.pc

%changelog
* Sun Feb 08 2026 FlagTree Project <contact@flagos.io> - 0.1.0-1
- Initial RPM release
