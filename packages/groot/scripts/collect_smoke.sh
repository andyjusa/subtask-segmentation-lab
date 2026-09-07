#!/usr/bin/env bash
set -euo pipefail

ISAAC_GROOT_ROOT="${ISAAC_GROOT_ROOT:-/root/projects/Isaac-GR00T}"
SIM_PYTHON="$ISAAC_GROOT_ROOT/gr00t/eval/sim/LIBERO/libero_uv/.venv/bin/python"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export ISAAC_GROOT_ROOT
export PYTHONPATH="$PROJECT_ROOT/src:$ISAAC_GROOT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export NO_ALBUMENTATIONS_UPDATE=1

"$SIM_PYTHON" -m groot_subtask_phase_probe.collect \
  --tasks stove_moka \
  --episodes 5 \
  --output-dir "$PROJECT_ROOT/artifacts"

