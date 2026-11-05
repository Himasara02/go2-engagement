#!/usr/bin/env bash
# One-time setup on the recording laptop (Ubuntu 22.04 or 24.04).
# Run from anywhere:   bash setup.sh
set -e
cd "$(dirname "$0")"

echo "== 1/5 system packages (asks for your password)"
sudo apt-get update -y
sudo apt-get install -y python3-venv python3-pip git curl ffmpeg

echo "== 2/5 Python environment (.venv)"
[ -d .venv ] || python3 -m venv .venv
source .venv/bin/activate
pip install -U pip setuptools wheel
pip install "mediapipe>=0.10.14" opencv-python ai-edge-litert matplotlib numpy pyorbbecsdk2

echo "== 3/5 pose model"
if [ ! -s models/pose_landmarker_full.task ]; then
  curl -L -o models/pose_landmarker_full.task \
    https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task
fi

echo "== 4/5 Orbbec USB permission (udev rule)"
OBDIR=$(python -c "import pyorbbecsdk,os;print(os.path.dirname(pyorbbecsdk.__file__))" 2>/dev/null | tail -1)
if [ -f "$OBDIR/shared/install_udev_rules.sh" ]; then
  sudo sh "$OBDIR/shared/install_udev_rules.sh"
else
  echo "   (udev script not found - if the camera is not detected, ask Claude Code)"
fi

echo "== 5/5 self-test"
python - <<'PY'
import cv2, mediapipe, numpy
import pyorbbecsdk
from engage.perception import Perception
Perception()
print("   imports + models OK")
PY

echo
echo "Setup done. Unplug and re-plug the Orbbec camera, then run:"
echo "   source .venv/bin/activate"
echo "   python check_orbbec.py"
