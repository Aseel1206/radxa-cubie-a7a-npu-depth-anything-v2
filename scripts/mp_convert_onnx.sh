#!/bin/bash
# usage: mp_convert_onnx.sh <dir> <model.onnx> <calib dataset.txt> "m0 m1 m2" "s0 s1 s2" <pcq i16 u8 ...>
source /workspace/mp/scripts/env.sh
D=/workspace/mp/$1; M=$2; CAL=$3; MEANS=$4; SCALES=$5; shift 5
cd $D
$PEG import onnx --model $M --output-model net.json --output-data net.data 2>&1 | sed 's/\x1b\[[0-9;]*m//g' | grep -E "^E |rror|Traceback|not support|Unsupport" | head -20
$PEG generate inputmeta --model net.json --input-meta-output net_inputmeta.yml >/dev/null 2>&1
$PEG generate postprocess-file --model net.json --postprocess-file-output net_postprocess_file.yml >/dev/null 2>&1
python3 - "$CAL" "$MEANS" "$SCALES" <<'PY'
import re, sys
cal, means, scales = sys.argv[1], sys.argv[2].split(), sys.argv[3].split()
s = open('net_inputmeta.yml').read()
s = re.sub(r'path: \S+', f'path: {cal}', s, count=1)
s = s.replace('reverse_channel: true', 'reverse_channel: false')
s = re.sub(r'mean:\n(\s+- [^\n]+\n)+', lambda m: 'mean:\n' + ''.join(f'        - {v}\n' for v in means), s)
s = re.sub(r'scale:\n(\s+- [^\n]+\n)+', lambda m: 'scale:\n' + ''.join(f'        - {v}\n' for v in scales), s)
s = s.replace('add_preproc_node: false', 'add_preproc_node: true')
open('net_inputmeta.yml', 'w').write(s)
p = open('net_postprocess_file.yml').read().replace('add_postproc_node: false', 'add_postproc_node: true')
open('net_postprocess_file.yml', 'w').write(p); print('meta ready')
PY
for q in "$@"; do
  case $q in pcq) a="perchannel_symmetric_affine int8 pcq";; i16) a="dynamic_fixed_point int16 i16";; u8) a="asymmetric_affine uint8 u8";; esac
  set -- $a
  echo "== quantize+export $3"
  $PEG quantize --model net.json --model-data net.data --device CPU --with-input-meta net_inputmeta.yml --iterations 47 --rebuild --model-quantize net_$3.quantize --quantizer $1 --qtype $2 > quant_$3.log 2>&1
  $PEG export ovxlib --model net.json --model-data net.data --model-quantize net_$3.quantize --with-input-meta net_inputmeta.yml \
    --postprocess-file net_postprocess_file.yml --dtype quantized --pack-nbg-unify --optimize VIP9000NANODI_PLUS_PID0X1000003B \
    --viv-sdk $VIVANTE_SDK_DIR --output-path $D/out_$3/net > export_$3.log 2>&1
  ls -la $D/out_$3_nbg_unify/network_binary.nb 2>/dev/null || { echo "FAILED $3"; sed 's/\x1b\[[0-9;]*m//g' export_$3.log | grep -E "^E |Fatal|rror" | head -6; }
done
chmod -R a+rwX $D
