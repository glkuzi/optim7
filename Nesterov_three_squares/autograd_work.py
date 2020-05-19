import torch as t 
from functions import function
from Nesterov_three_sq import NTS

x = t.randn(3, dtype=t.float32, requires_grad=True)
# [ 0.8600,  0.4378, -0.6523]
#x = t.tensor([ -1, 1, 1], dtype=t.float32, requires_grad=True)
#x = t.tensor([0.9525, 0.7985, 0.2164], dtype=t.float32, requires_grad=True)
x_last = t.zeros(3, dtype=t.float32)
#print(x)

eps = 100
epoch = 500
L = 1.
h=1e-1
optimizer = NTS([x], function, lr=1e-2, L=1, epoch=100, adaptive_L=True, adaptive_lr=True)
while eps > 1e-6:

    #print(i)
    #print('' * 100, end='\r')
    print(' '*30,x, eps, optimizer.defaults['L'], optimizer.defaults['lr'], end='\r')

    with t.no_grad():
        x_last.copy_(x)

    optimizer.zero_grad()
    optimizer.step()

    eps = ((x - x_last) ** 2).sqrt().mean()

print(x)
print((function(x) ** 2).sum())