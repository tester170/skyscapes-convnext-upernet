#!/usr/bin/env bash
# Original code of the models used by configs/skyscapes/, at the commits of the experiments, into vendor/:
#   dinov3  github.com/facebookresearch/dinov3  6876159  (DINOv3 ConvNeXt and ViT; also fetched by env/setup_env.sh)
#   MsRE    github.com/woldier/MsRE             d515366  (only for the MsRE config; Apache-2.0)
#   Rein    github.com/w1oves/Rein              dc06342  (branch train; only for the Rein config; GPL-3.0)
# Each repository is fetched at exactly that commit (git fetch by hash, no submodules). Idempotent.
# Usage: bash env/fetch_vendor.sh [dinov3] [msre] [rein]   (default: all three)
set -euo pipefail
BUNDLE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
declare -A URL=([dinov3]=https://github.com/facebookresearch/dinov3.git [msre]=https://github.com/woldier/MsRE.git
                [rein]=https://github.com/w1oves/Rein.git)
declare -A COMMIT=([dinov3]=6876159a11b4df116f30f667f8c9888617df0751 [msre]=d5153662477f4e813585ff4a530c8d80f0a4e11b
                   [rein]=dc063429c4dadc0da9c6252b3db22fc55a9882ab)
declare -A DIR=([dinov3]=dinov3 [msre]=MsRE [rein]=Rein)
for name in "${@:-dinov3 msre rein}"; do
  for n in $name; do
    d="$BUNDLE/vendor/${DIR[$n]}"
    if [ "$(git -C "$d" rev-parse HEAD 2>/dev/null)" = "${COMMIT[$n]}" ]; then
      echo "$n: already at ${COMMIT[$n]:0:7}"; continue
    fi
    echo "== $n: ${URL[$n]} @ ${COMMIT[$n]:0:7}"
    rm -rf "$d"; mkdir -p "$d"
    git -C "$d" init -q
    git -C "$d" fetch -q --depth 1 "${URL[$n]}" "${COMMIT[$n]}"
    git -C "$d" checkout -q FETCH_HEAD
  done
done
