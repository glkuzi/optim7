import torch
from functions import func, function
from torch.autograd.functional import jacobian
from Newton import Newton_for_Nesterov

def Nesterov_3_qv(x, epoch=100, epoch_N_G=100, h_N_G=0.001):
    
    L = func(x)
    
    f_line = []
    for i in range(epoch):
        
        f = func(x)
        F = function(x)
        jac = jacobian(function, x)[:, 0]
#         print('x : {}'.format(x))
#         print('f : {}'.format(f))
#         print('jac : {}'.format(jac))
#         print('hes : {}'.format(hes))
        
        x_k = torch.zeros_like(x)
        x_k.copy_(x)
        x_k = x_k.view(3, 1)
        
        func_Nes = lambda y: 1 /(2 * f) * (f ** 2 + ((F + jac.mm(y - x_k)) ** 2).sum()) + L / 2 * ((y - x_k) ** 2).sum()
        
#         print('1 : {} {}'.format(x_k, func_Nes(x_k)))
#         print('2 : {} {}'.format(x, func_Nes(x)))
        x, _ = Newton_for_Nesterov(x, func_Nes, epoch=epoch_N_G, h=h_N_G)
#         print('3 : {} {}'.format(x_k, func_Nes(x_k)))
#         print('4 : {} {}'.format(x, func_Nes(x)))
        print(x, end='\r')
        f_line.append(f)
        
    return x, f_line