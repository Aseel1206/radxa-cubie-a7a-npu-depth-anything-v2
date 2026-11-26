import sys, onnx, numpy as np
from onnx import numpy_helper as nh
src, dst = sys.argv[1], sys.argv[2]
m = onnx.load(src); g = m.graph
for n in g.node:
    if n.op_type == 'Resize' and len(n.input) > 3 and n.input[3]:
        sizes = [i for i in g.initializer if i.name == n.input[3]][0]
        sz = nh.to_array(sizes)
        # input spatial size from shape inference
        mi = onnx.shape_inference.infer_shapes(m)
        shp = {v.name: [d.dim_value for d in v.type.tensor_type.shape.dim] for v in list(mi.graph.value_info) + list(mi.graph.input)}
        ish = shp[n.input[0]]
        scales = np.array([sz[i] / ish[i] for i in range(4)], dtype=np.float32)
        print('Resize', n.name, 'in', ish, 'sizes', sz.tolist(), '-> scales', scales.tolist())
        name = n.name + '_scales'
        g.initializer.append(nh.from_array(scales, name))
        g.initializer.append(nh.from_array(np.zeros([0], np.float32), n.name + '_roi'))
        n.input[1] = n.name + '_roi'; n.input[2] = name; del n.input[3]
onnx.checker.check_model(m)
onnx.save(m, dst)

# --- pass 2: Unsqueeze/Squeeze -> Reshape with constant static shape (Acuity mis-imports some) ---
m = onnx.load(dst); g = m.graph
mi = onnx.shape_inference.infer_shapes(m)
shp = {v.name: [d.dim_value for d in v.type.tensor_type.shape.dim] for v in list(mi.graph.value_info) + list(mi.graph.output)}
cnt = 0
for n in list(g.node):
    if n.op_type in ('Unsqueeze', 'Squeeze'):
        o = shp.get(n.output[0])
        if not o or 0 in o:
            continue
        sn = n.name + '_rshape'
        g.initializer.append(nh.from_array(np.array(o, dtype=np.int64), sn))
        n.op_type = 'Reshape'
        del n.attribute[:]
        del n.input[1:]
        n.input.append(sn)
        cnt += 1
print('rewrote', cnt, 'Unsqueeze/Squeeze -> Reshape')
onnx.checker.check_model(m)
onnx.save(m, dst)

# --- pass 3: add kernel_shape (omitted by newer torch exporters; Acuity requires it) ---
m = onnx.load(dst); g = m.graph
init = {i.name: i for i in g.initializer}
cnt = 0
for n in g.node:
    if n.op_type in ('Conv', 'ConvTranspose') and not any(a.name == 'kernel_shape' for a in n.attribute):
        w = init[n.input[1]]
        n.attribute.append(onnx.helper.make_attribute('kernel_shape', list(w.dims[2:])))
        cnt += 1
print('added kernel_shape to', cnt, 'Conv/ConvTranspose nodes')
onnx.checker.check_model(m)
onnx.save(m, dst)
