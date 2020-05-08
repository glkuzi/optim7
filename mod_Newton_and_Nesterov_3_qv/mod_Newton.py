import torch
from torch.autograd.functional import jacobian
from functions import func, function
from Newton import Newton_for_Newton


def mod_Newton(x, epoch=1000, epoch_N_G=100, h_N_G=0.001):
    
    L = func(x)
    n = x.shape[0]
    E = torch.eye(n, n, dtype=torch.float32)
    
    f_line = [func(x)]
    for i in range(epoch):
        
        F = function(x)
        F_T = F.transpose(0, 1)
        jac = jacobian(function, x)[:,0]
        jac_T = jac.transpose(0, 1)
        
#         print(F)
#         print()
#         print(F_T)
#         print()
#         print(jac)
#         print()
#         print(jac_T)
#         print()
        
        l = torch.tensor(1, dtype=torch.float32)
        
        def func_Nes(l):
            
            A = (E * l + jac.mm(jac_T) / L).inverse()
            
            return l / 2 + (A.mm(F) * F).sum() / 2
        
        l, line = Newton_for_Newton(l, func_Nes, epoch=epoch_N_G, h=h_N_G)
#         return line
        #print(l, end='\r') # lambda для двойственной задачи
        
        B = (E * l + jac.mm(jac_T) / L).inverse()               
        h = -1 / L * jac_T.mm(B).mm(F)[:,0]
        
#         print(B)
#         print()
#         print(jac_T)
#         print()
#         print(F)
#         print()
#         print(h, end='\r')
#         print()
        
        print(x, end='\r')

        x += h
        
        f_line.append(func(x))
        
    return x, f_line