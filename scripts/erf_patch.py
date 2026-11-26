import sys, onnx, numpy as np, onnxruntime as ort
from onnx import numpy_helper, helper, TensorProto
z=np.linspace(-6,6,20001); from math import erf
e=np.array([erf(v) for v in z])
# fit erf(z) ~ tanh(a z + b z^3)
A=np.stack([z,z**3],1); best=None
for it in range(3):
    pass
from itertools import product
best=(1e9,None)
for a in np.linspace(1.10,1.20,41):
    for b in np.linspace(0.08,0.14,61):
        err=np.abs(np.tanh(a*z+b*z**3 if False else a*z*(1+0)+b*z**3)-e).max()
        if err<best[0]: best=(err,(a,b))
print('erf fit max err %.4f'%best[0],best[1]); a,b=best[1]
m=onnx.load(sys.argv[1]); g=m.graph; out=[]
c=lambda n,v: g.initializer.append(numpy_helper.from_array(np.array(v,np.float32),n))
c('erf_a',a); c('erf_b',b)
for i,n in enumerate(g.node):
    if n.op_type=='Erf':
        x,y=n.input[0],n.output[0]; p=f'erf{i}_'
        out+= [helper.make_node('Mul',[x,x],[p+'x2']),helper.make_node('Mul',[p+'x2','erf_b'],[p+'bx2']),
               helper.make_node('Add',[p+'bx2','erf_a'],[p+'inner']),helper.make_node('Mul',[p+'inner',x],[p+'arg']),
               helper.make_node('Tanh',[p+'arg'],[y])]
    else: out.append(n)
del g.node[:]; g.node.extend(out); onnx.save(m,sys.argv[2])
