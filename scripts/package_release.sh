#!/usr/bin/env bash
# Build the release artifacts from the committed tree (nothing is pushed or published):
#   <out>/lhas-repro-<commit>.zip     git archive of HEAD (tracked files only; no caches)
#   <out>/lhas-repro-<commit>.bundle  git bundle with the full history
# Usage: scripts/package_release.sh <out_dir>
set -euo pipefail
OUT=${1:?output directory}
mkdir -p "$OUT"
C=$(git rev-parse --short HEAD)
if [ -n "$(git status --porcelain)" ]; then
  echo "working tree not clean; commit first" >&2
  exit 1
fi
git archive --format=zip --prefix="lhas-repro/" -o "$OUT/lhas-repro-$C.zip" HEAD
git bundle create "$OUT/lhas-repro-$C.bundle" --all
git bundle verify "$OUT/lhas-repro-$C.bundle" >/dev/null
( cd "$OUT" && sha256sum "lhas-repro-$C.zip" "lhas-repro-$C.bundle" > "lhas-repro-$C.sha256" )
echo "wrote $OUT/lhas-repro-$C.zip $OUT/lhas-repro-$C.bundle"
