import torch
import math as m
from torch.optim.optimizer import Optimizer
from torch.autograd import backward
from copy import deepcopy
import random
from fgm import FGM

class NTS(Optimizer):
    """Implements Fast Gradient Method for convex functions.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-2)

    """

    def __init__(self, params, function, x0=None, y0=None, lr=1e-2, L=1, epoch=100, adaptive_lr=False, adaptive_L=False, limit_L=1e-10, limit_recurse=10, r=None, seed=None):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))

        random.seed(seed)
        
        defaults = dict(function=function, x0=x0, lr=lr, L=L, epoch=epoch, 
                        adaptive_lr=adaptive_lr, adaptive_L=adaptive_L, limit_L=limit_L, 
                        limit_recurse=limit_recurse, n_recurse=0, r=r, flag_L=False, flag_lr=False, indices=None)
        super(NTS, self).__init__(params, defaults)
  

                
    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()


        param_groups_k = deepcopy(self.param_groups)
        params_k = self.get_params(param_groups_k)


        F, grad_F = self.func_and_grad()
        self.make_grad(False)

        f_nes_0 = self.get_f_nes(param_groups_k, F, grad_F)

        optimizer = FGM(params_k, lr=self.defaults['lr'])

        for i in range(self.defaults['epoch']):

            f_nes = self.get_f_nes(param_groups_k, F, grad_F)

            
            #print('f_nes = {}'.format(f_nes))

            optimizer.zero_grad()
            f_nes.backward()
            optimizer.step()

        self.make_grad()
                
        f_nes = self.get_f_nes(param_groups_k, F, grad_F)
        
        if self.defaults['adaptive_lr']:
            if f_nes_0 < f_nes or torch.isnan(f_nes) or torch.isinf(f_nes):
                #print('Damn!')
                self.defaults['flag_lr'] = True
                self.defaults['lr'] *= 1e-1
                return self.step(closure=closure)

            else:

                if self.defaults['lr'] < 1e-1 and not self.defaults['flag_lr']:
                    self.defaults['lr'] *= 1e1

        else:

            if torch.isnan(f_nes):
                raise Exception('Loss function value nan. Try choose smaller value of learning rate')
            if torch.isinf(f_nes):
                raise Exception('Loss function value inf. Try choose smaller value of learning rate')



        buf_params = deepcopy(self.param_groups)
        self.copy_params(param_groups_k)

        f = (F ** 2).sum() ** 0.5
        f_i = self.get_f_value()

        #print('f_i = {}  f_nes = {}  L = {}  lr = {}  n_recurse = {}'.format(f_i, f_nes, self.defaults['L'], self.defaults['lr'], self.defaults['n_recurse']))
        #print(self.defaults['indices'])
        #print(self.param_groups[0]['params'])
        #q = input()

        if self.defaults['adaptive_L']:
            if f_i <= f_nes:
                
                if self.defaults['L'] > self.defaults['limit_L'] and self.defaults['n_recurse'] < self.defaults['limit_recurse']:
                    self.defaults['L'] /= 2
                else:
                    self.defaults['flag_L'] = False
                    self.defaults['flag_lr'] = False
                    self.defaults['n_recurse'] = 0
                    return loss
                
                if self.defaults['flag_L']:
                    self.defaults['flag_L'] = False
                    self.defaults['flag_lr'] = False
                    self.defaults['n_recurse'] = 0
                    return loss
                else:
                    self.copy_params(buf_params)
                    self.defaults['n_recurse'] += 1
                    return self.step(closure=closure)
            else:
                self.defaults['flag_L'] = True
                self.defaults['L'] *= 2
                self.copy_params(buf_params)
                return self.step(closure=closure)

        #print('end')

        return loss


    def set_x(self, x0, y_true):

        self.defaults['x0'] = x0
        self.defaults['y_true'] = y_true
        
        if self.defaults['r'] is None:
            self.defaults['indices'] = list(range(y_true.shape[0]))
        else:
            self.defaults['indices'] = random.sample(list(range(y_true.shape[0])), self.defaults['r'])
        


    def func_and_grad(self):

        if self.defaults['x0'] is None:
            params = self.get_params(self.param_groups)
            F = self.defaults['function'](*params)[self.defaults['indices']]
        else:
            F = (self.defaults['function'](self.defaults['x0']) - self.defaults['y_true'])[self.defaults['indices']]

        grad_F = []
        for F_i in F:

            backward(F_i, retain_graph=True)

            grad_groups = []
            for group in self.param_groups:

                grad_group = []
                for p in group['params']:

                    grad_group.append(p.grad.data)
                    p.grad.data = torch.zeros_like(p.grad.data)
                
                grad_groups.append(grad_group)
            
            grad_F.append(grad_groups)

        F.detach_()

        return F, grad_F
                
        
    def make_grad(self, value=True):

        for group in self.param_groups:
            for p in group['params']:
                p.requires_grad_(value)
        return         


    def get_f_value(self):

        if self.defaults['x0'] is None:
            
            params = self.get_params(self.param_groups)
            return (self.defaults['function'](*params) ** 2).sum() ** 0.5

        else:

            return ((self.defaults['function'](self.defaults['x0']) - self.defaults['y_true'])[self.defaults['indices']] ** 2).sum() ** 0.5


    def get_params(self, param_groups_k):

        params = []

        for group in param_groups_k:
            for p in group['params']:
                params.append(p)

        return params

    def copy_params(self, param_groups_c):

        for group, group_c in zip(self.param_groups, param_groups_c):
            for p, p_c in zip(group['params'], group_c['params']):
                with torch.no_grad():
                    p.copy_(p_c)


    def get_f_nes(self, param_groups_k, F, grad_F):
        

        f_1 = 0

        for F_i, grad_groups in zip(F, grad_F):

            f_1_1 = 0

            for grad_group, group, group_k in zip(grad_groups, self.param_groups, param_groups_k):
                for grad, p, p_k in zip(grad_group, group['params'], group_k['params']):

                    if grad is None:
                        continue
                    if grad.data.is_sparse:
                        raise RuntimeError("FGM is not compatible with sparse gradients")

                    f_1_1 += (grad * (p_k - p)).sum()



            f_1 += (F_i.detach() + f_1_1) ** 2


        f_2 = 0

        for group, group_k in zip(self.param_groups, param_groups_k):
            for p, p_k in zip(group['params'], group_k['params']):
                f_2 += self.defaults['L'] * ((p_k - p) ** 2).sum() / 2

        func = (F.detach() ** 2).sum()

        f_nes = (func + f_1) / (2 * func ** 0.5) + f_2

        return f_nes