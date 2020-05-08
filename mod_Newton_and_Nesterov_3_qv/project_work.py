import torch
from Nesterov_3qv import Nesterov_3_qv
from mod_Newton import mod_Newton
from functions import func

epoch_N_G = 100

x1 = torch.randn(3, dtype=torch.float32)
x2 = torch.zeros_like(x1)
x2.copy_(x1)
# x = torch.tensor([-1., 1., 1.], dtype=torch.float32)
# x = torch.tensor([ 0.7876, -0.6326,  0.7936], dtype=torch.float32)

x_N1, f_N1 = Nesterov_3_qv(x1, epoch=200, epoch_N_G=epoch_N_G)

print('x = {}  func = {}'.format((x_N1), func(x_N1)))

x = torch.tensor([-1, 1, 1], dtype=torch.float32)

x_N2, f_N2 = mod_Newton(x2, epoch=6000, epoch_N_G=epoch_N_G)

print('x = {}  func = {}'.format((x_N2), func(x_N2)))
