import torch

# Nesterov function

def func(x):
    
    f = (x[0] - 1) ** 2 / 4
#     print(x)
#     print('x_i : {}  f : {}'.format(x[0], f))
#     print(1)
    
    for i in range(1, x.shape[0]):
        
        f_i = (x[i] - 2 * x[i - 1] ** 2 + 1) ** 2
        f += f_i
#         print('x_i : {}  f : {}'.format(x[i], f_i))
#     print()
        
    return f


# Nesterov vector function for system F = 0

def function(x):
    
    F = torch.zeros(x.shape[0], 1)
    
    F[0] = x[0] - 1
    for i in range(1, x.shape[0]):
        F[i] = x[i] - 2 * x[i - 1] ** 2 + 1
        
    return F