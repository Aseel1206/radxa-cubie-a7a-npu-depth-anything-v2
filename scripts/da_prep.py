import onnx,sys,numpy as np,subprocess
from onnx import numpy_helper, shape_inference
H,W=int(sys.argv[1]),int(sys.argv[2]); D='/workspace/mp/da/'
m=onnx.load(D+'da_orig.onnx'); print('in',[ (i.name,[d.dim_value or d.dim_param for d in i.type.tensor_type.shape.dim]) for i in m.graph.input],'out',[ (i.name,[d.dim_value or d.dim_param for d in i.type.tensor_type.shape.dim]) for i in m.graph.output])
inp=m.graph.input[0]; 
for d,v in zip(inp.type.tensor_type.shape.dim,(1,3,H,W)): d.ClearField('dim_param'); d.dim_value=v
for o in m.graph.output:
    o.type.tensor_type.ClearField('shape')
    for v in (1,H,W): o.type.tensor_type.shape.dim.add().dim_value=v
del m.graph.value_info[:]
print('ir',m.ir_version,[o.version for o in m.opset_import]); m.ir_version=8
onnx.save(m,D+'da_s1.onnx')
