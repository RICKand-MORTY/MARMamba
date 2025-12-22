import argparse
import os
import sys
sys.path.append("../")
import torch
from model.mamba import MambaFormer
from torchmeter import Meter


img = torch.randn(1, 1, 416, 416).cuda()
model = MambaFormer(in_channels=1).cuda()
model.eval()
model = Meter(model)
total = sum([param.nelement() for param in model.parameters()])
print("Number of parameter: %.2fM" % (total / 1e6))
with torch.no_grad():
    output = model(img)
print("="*10, " Overall Report ", "="*10)
# FLOPs/MACs measurement
print(model.cal)
