#!/bin/bash
# Render the Debian packaging for a specific vendor flavor.
#
# The committed packaging targets the default "nvidia" vendor
# (libtriton-jit-nvidia / -dev). Other vendors share identical packaging
# except for the binary package name, so instead of duplicating control and
# the .install files per vendor we rewrite the vendor token in place before
# dpkg-buildpackage.
#
# Only the package-name token `libtriton-jit-nvidia` is substituted; prose
# mentions like "nvidia/cuda base image" in control do not contain that token
# and are left untouched.
#
# Usage: render-vendor.sh <vendor> [debian_dir]
#   vendor     : nvidia (default, no-op) | metax | mthreads
#   debian_dir : path to the debian/ dir to rewrite (default: ./debian)
set -euo pipefail

VENDOR="${1:?usage: render-vendor.sh <vendor> [debian_dir]}"
DEBIAN_DIR="${2:-debian}"

case "$VENDOR" in
  nvidia|metax|mthreads) ;;
  *) echo "render-vendor: unknown vendor '$VENDOR' (want nvidia|metax|mthreads)" >&2; exit 1 ;;
esac

if [ ! -f "${DEBIAN_DIR}/control" ]; then
  echo "render-vendor: ${DEBIAN_DIR}/control not found" >&2; exit 1
fi

# nvidia is what the tree already ships — nothing to rewrite.
if [ "$VENDOR" = "nvidia" ]; then
  echo "render-vendor: vendor=nvidia, packaging left as-is"
  exit 0
fi

echo "render-vendor: rendering packaging for vendor=${VENDOR}"

# 1) Package names in control (Package:, and the dev package's Depends).
sed -i "s/libtriton-jit-nvidia/libtriton-jit-${VENDOR}/g" "${DEBIAN_DIR}/control"

# 2) Per-package .install files must be named after the binary package.
mv "${DEBIAN_DIR}/libtriton-jit-nvidia.install" \
   "${DEBIAN_DIR}/libtriton-jit-${VENDOR}.install"
mv "${DEBIAN_DIR}/libtriton-jit-nvidia-dev.install" \
   "${DEBIAN_DIR}/libtriton-jit-${VENDOR}-dev.install"

echo "render-vendor: done -> libtriton-jit-${VENDOR} / -dev"
