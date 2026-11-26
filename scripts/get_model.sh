#!/bin/bash
set -e; mkdir -p da; cd da
curl -sSL -o da_orig.onnx https://huggingface.co/onnx-community/depth-anything-v2-small/resolve/main/onnx/model.onnx; ls -l
