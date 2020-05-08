from torch.autograd.functional import jacobian
from torch.autograd.functional import hessian

def Newton_for_Newton(x, func, epoch=100, h=1):
    
    
    f_line = []
    for i in range(epoch):
        
#         print(i, end='\r')
        
        f_line.append(func(x))
#         print(f_line[-1])
        
        jac = jacobian(func, x)
        hes = hessian(func, x).sum()
        
#         print('jac: {}'.format(jac))
#         print('hes: {}'.format(hes))
#         print('x: {}'.format(x))
        if x - jac / hes < 0:
#             print('neg : {}'.format(x - jac / hes))
            if h < 1e-3:
                break
            h *= 0.1
            continue
            
        x -= h * jac / hes
        
        
    f_line.append(func(x))
    
    return x, f_line

    

def Newton_for_Nesterov(x, func, epoch=100, h=0.001):
    
    n = x.shape[0]
    
    f_line = []
    for i in range(epoch):
        
#         print(i, end='\r')
        
        f_line.append(func(x))
#         print(f_line[-1])
        
        jac = jacobian(func, x.view(n, 1))
        hes = hessian(func, x.view(n, 1)).sum()
        
#         print('jac: {}'.format(jac))
#         print('hes: {}'.format(hes))
#         print('x: {}'.format(x))
        
        h = jac / hes
        h = h.view(n)
#         print(h)
        
        x -= h
        
    f_line.append(func(x))
    
    return x, f_line