import onnx,numpy as np,collections,sys
from onnx import numpy_helper, shape_inference
D='/workspace/mp/da/'; m=onnx.load(D+sys.argv[1]); m=shape_inference.infer_shapes(m)
vi={v.name:v for v in list(m.graph.value_info)+list(m.graph.input)+list(m.graph.output)}
def shp(n):
    if n not in vi: return None
    d=vi[n].type.tensor_type.shape.dim
    return [x.dim_value for x in d] if d and all(x.dim_value>0 for x in d) else None
new=[];k=0;miss=0
for n in m.graph.node:
    if n.op_type=='Shape':
        s=shp(n.input[0])
        if s: m.graph.initializer.append(numpy_helper.from_array(np.array(s,np.int64),n.output[0])); k+=1; continue
        miss+=1
    new.append(n)
del m.graph.node[:]; m.graph.node.extend(new); print('folded',k,'missing',miss); onnx.save(m,D+sys.argv[2])
