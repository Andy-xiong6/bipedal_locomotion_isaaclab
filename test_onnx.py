import onnx 

model = onnx.load("logs/rsl_rl/pf_blind_flat/2024-10-24_14-28-16/exported/policy.onnx")
print(onnx.helper.printable_graph(model.graph))
print("---------------------------------------------------")
model = onnx.load("logs/policy_1.onnx")
print(onnx.helper.printable_graph(model.graph))
