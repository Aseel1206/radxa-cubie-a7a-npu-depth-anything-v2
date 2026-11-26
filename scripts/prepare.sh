#!/bin/bash
# Inside the toolchain container (repo mounted at /workspace/mp). Input size is a multiple of 14; I used 252x336.
set -e
cd /workspace/mp/da
python3 ../scripts/da_prep.py 252 336                             # static shape, ir_version 8 -> da_s1.onnx
python3 -m onnxsim da_s1.onnx da_sim.onnx
python3 ../scripts/da_fold.py da_sim.onnx da_f.onnx               # fold Shape ops
python3 -m onnxsim da_f.onnx da_sim2.onnx
python3 ../scripts/da_constresize.py                              # the bicubic resize of the position embeddings has constant input: precompute it -> da_sim3.onnx
python3 ../scripts/onnx_patch.py da_sim3.onnx da_p.onnx           # Resize sizes->scales, Squeeze/Unsqueeze->Reshape, kernel_shape
python3 ../scripts/erf_patch.py da_p.onnx da_npu.onnx             # Erf -> tanh approximation
