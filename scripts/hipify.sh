#!/usr/bin/env bash
# Convert a CUDA source to HIP and show the diff.
# Usage: scripts/hipify.sh cuda/saxpy.cu  ->  saxpy.hip
set -euo pipefail
src="$1"; out="$(basename "${src%.cu}").hip"
hipify-perl "$src" > "$out"
echo "== wrote $out =="
diff -u "$src" "$out" || true
