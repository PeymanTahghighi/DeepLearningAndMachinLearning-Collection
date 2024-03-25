import torch
import numpy as np

k = torch.randn((64, 32));
q = torch.randn((64, 32));
without_scale = q@k.transpose(1, 0) #/ (np.sqrt(32));
with_scale = q@k.transpose(1, 0) / (np.sqrt(32));
print(k.var())
print(f'without scale: {without_scale.var()}\nwith scale: {with_scale.var()}');