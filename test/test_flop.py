import argparse
import os
import sys
sys.path.append("../")
import torch
from model.mamba import MambaFormer
from calflops import calculate_flops

input= torch.randn(1, 1, 416, 416).cuda()
model=MambaFormer(in_channels=1).cuda()
model.eval()
total = sum([param.nelement() for param in model.parameters()])
print("Number of parameter: %.2fM" % (total / 1e6))
flops, macs, params = calculate_flops(model=model, 
                                      input_shape=(1, 1, 416, 416),
                                      output_as_string=True,
                                      output_precision=4)
print("Alexnet FLOPs:%s   MACs:%s   Params:%s \n" %(flops, macs, params))


# Measure GPU memory usage
torch.cuda.empty_cache()
start_memory = torch.cuda.memory_allocated()
# measure the memory cost of initializing our model. 
model=MambaFormer(in_channels=1).cuda()
model.eval()
with torch.no_grad():
    model_output = model(input)
    torch.cuda.synchronize()
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