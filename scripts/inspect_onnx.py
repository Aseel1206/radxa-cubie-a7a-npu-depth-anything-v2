import onnx,collections,sys
m=onnx.load(sys.argv[1]);print(m.opset_import[0].version)
f=lambda i:(i.name,[d.dim_value or d.dim_param for d in i.type.tensor_type.shape.dim])
print([f(i) for i in m.graph.input],[f(i) for i in m.graph.output]);print(collections.Counter(n.op_type for n in m.graph.node))
