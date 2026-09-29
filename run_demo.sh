#!/usr/bin/env bash
set -e
DEPTH_MODEL="${1:-midas_models/dpt_hybrid-midas-501f0c75.pt}"
mkdir -p outputs
python img2stereo.py \
  --img_path data/kitti2015/left/000187_10.png \
  --depthmodel_path "$DEPTH_MODEL" \
  --direction uni
