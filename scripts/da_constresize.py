import onnx,numpy as np,onnxruntime as ort
from onnx import numpy_helper as nh, helper
D='/workspace/mp/da/'; m=onnx.load(D+'da_sim2.onnx'); g=m.graph
init={i.name for i in g.initializer}; prod={o for n in g.node for o in n.output}
tgt=[n for n in g.node if n.op_type=='Resize' and n.input[0] not in prod]
print('const-input resizes',[(n.name,n.input[0] in init) for n in tgt])
for n in tgt:
    g.output.append(helper.make_tensor_value_info(n.output[0],onnx.TensorProto.FLOAT,None))
onnx.save(m,D+'tmp_dbg.onnx')
s=ort.InferenceSession(D+'tmp_dbg.onnx',providers=['CPUExecutionProvider'])
res=s.run([n.output[0] for n in tgt],{'pixel_values':np.zeros((1,3,252,336),np.float32)})
m=onnx.load(D+'da_sim2.onnx'); g=m.graph
for n,r in zip(tgt,res):
    g.initializer.append(nh.from_array(r,n.output[0])); print('folded',n.name,r.shape)
g2=[n for n in g.node if n not in tgt]; del g.node[:]; g.node.extend(g2); onnx.save(m,D+'da_sim3.onnx')
