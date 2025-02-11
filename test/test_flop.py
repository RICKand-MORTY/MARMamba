import argparse
from thop import profile
import os
import torch
from model.mamba import MambaFormer

input= torch.randn(1, 1, 416, 416).cuda()
model=MambaFormer(in_channels=1).cuda()
model.eval()
total = sum([param.nelement() for param in model.parameters()])
print("Number of parameter: %.2fM" % (total / 1e6))
flops, params = profile(model, inputs=(input,))
print(f"FLOPs: {flops}, Params: {params}")
print('MACs = ' + str(flops / 1000 ** 3) + 'G')


# Measure GPU memory usage
start_memory = torch.cuda.memory_allocated()
with torch.no_grad():
    model_output = model(input)
end_memory = torch.cuda.memory_allocated()
total_memory_reserved = torch.cuda.memory_reserved()

print(f"Memory allocated before model run: {start_memory / (1024 ** 3):.2f} GiB")
print(f"Memory allocated after model run: {end_memory / (1024 ** 3):.2f} GiB")
print(f"Total memory reserved by PyTorch: {total_memory_reserved / (1024 ** 3):.2f} GiB")
print(f"Memory used by the model: {(end_memory - start_memory) / (1024 ** 3):.2f} GiB")

print("**********************************************************************************")

print(f"Memory allocated before model run: {start_memory / (1024 ** 2):.2f} MiB")
print(f"Memory allocated after model run: {end_memory / (1024 ** 2):.2f} MiB")
print(f"Total memory reserved by PyTorch: {total_memory_reserved / (1024 ** 2):.2f} MiB")
print(f"Memory used by the model: {(end_memory - start_memory) / (1024 ** 2):.2f} MiB")