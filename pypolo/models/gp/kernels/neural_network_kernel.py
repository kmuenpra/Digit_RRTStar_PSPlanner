from typing import Union

import gpytorch
import torch
from gpytorch.kernels.kernel import Kernel
from gpytorch.constraints import Positive
# import torch.nn as nn
from linear_operator.operators import LinearOperator,RootLinearOperator, MatmulLinearOperator
from torch import Tensor
import linear_operator



class NeuralNetworkKernel(gpytorch.kernels.Kernel):
    is_stationary = False
    has_lengthscale = True
    
    def __init__(self, lx=1.0, ly=1.0, sigma_f=1.0, B=1.0, sigma_n=1.0, **kwargs):
        super().__init__(**kwargs)
        # self.lx = lx
        # self.ly = ly
        # self.sigma_f = sigma_f
        # self.B = B
        # self.sigma_n = sigma_n
        self.batch_shape = torch.Size([1])

        # Register the raw parameters with positive constraints
        self.register_parameter(
            name='raw_lx',
            parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape, 1, 1))
        )
        self.register_parameter(
            name='raw_ly',
            parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape, 1, 1))
        )
        self.register_parameter(
            name='raw_sigma_f',
            parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape, 1, 1))
        )
        self.register_parameter(
            name='raw_B',
            parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape, 1, 1))
        )
        self.register_parameter(
            name='raw_sigma_n',
            parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape, 1, 1))
        )

        # Set the parameter constraints to be positive
        self.register_constraint("raw_lx", Positive())
        self.register_constraint("raw_ly", Positive())
        self.register_constraint("raw_sigma_f", Positive())
        self.register_constraint("raw_B", Positive())
        self.register_constraint("raw_sigma_n", Positive())

    @property
    def lx(self):
        return self.raw_lx_constraint.transform(self.raw_lx)

    @lx.setter
    def lx(self, value):
        self._set_lx(value)

    def _set_lx(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_lx)
        self.initialize(raw_lx=self.raw_lx_constraint.inverse_transform(value))

    @property
    def ly(self):
        return self.raw_ly_constraint.transform(self.raw_ly)

    @ly.setter
    def ly(self, value):
        self._set_ly(value)

    def _set_ly(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_ly)
        self.initialize(raw_ly=self.raw_ly_constraint.inverse_transform(value))

    @property
    def sigma_f(self):
        return self.raw_sigma_f_constraint.transform(self.raw_sigma_f)

    @sigma_f.setter
    def sigma_f(self, value):
        self._set_sigma_f(value)

    def _set_sigma_f(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_sigma_f)
        self.initialize(raw_sigma_f=self.raw_sigma_f_constraint.inverse_transform(value))

    @property
    def B(self):
        return self.raw_B_constraint.transform(self.raw_B)

    @B.setter
    def B(self, value):
        self._set_B(value)

    def _set_B(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_B)
        self.initialize(raw_B=self.raw_B_constraint.inverse_transform(value))

    @property
    def sigma_n(self):
        return self.raw_sigma_n_constraint.transform(self.raw_sigma_n)

    @sigma_n.setter
    def sigma_n(self, value):
        self._set_sigma_n(value)

    def _set_sigma_n(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_sigma_n)
        self.initialize(raw_sigma_n=self.raw_sigma_n_constraint.inverse_transform(value))
    
    

    def forward(self, x1: Tensor, x2: Tensor, diag: bool = False, last_dim_is_batch: bool = False, **params):
        # Compute the kernel matrix between x1 and x2
        # varMat = torch.Tensor([[self.lx, 0.], [0., self.ly]])
        # Sigma = torch.inverse(torch.matrix_power(varMat, 2)).to(torch.float64)
        
        # Slice inputs
        # x = self.slice_input(x1)
        # xp = self.slice_input(x2)

        # x = x1.transpose(-1, -2).unsqueeze(-1)
        # xp = x2.transpose(-1, -2).unsqueeze(-1)
        
        # num = self.B + (x * xp).sum(dim=-1)[0].div(self.lx**2) + (x * xp).sum(dim=-1)[1].div(self.ly**2)
        # den_xx = 1 + self.B + (x * x).sum(dim=-1)[0].div(self.lx**2) + (x * x).sum(dim=-1)[1].div(self.ly**2)
        # den_xpxp = 1 + self.B + (xp * xp).sum(dim=-1)[0].div(self.lx**2) + (xp * xp).sum(dim=-1)[1].div(self.ly**2)


        # K = self.sigma_f**2 * torch.arcsin(num / torch.sqrt(torch.mul(den_xx ,den_xpxp)))
        
        
        # print(self.covar_dist(x, xp, square_dist=True, diag=diag, **params))
        # print(self.covar_dist(x, xp, square_dist=True, diag=diag, **params).div_(-2).exp_())

        print(x1)
        print(x2)
        
        # x = x1
        # xp = x2
        
        # print("Sigma", Sigma)
        # print("xp",xp)
        # print("x",x)
    
        # num = self.B + 2 * torch.mm(xp, torch.mm(Sigma, x.t()))
        # den_xx = 1 + self.B + 2 * torch.mm(x, torch.mm(Sigma, x.t()))
        # den_xpxp = 1 + self.B + 2 * torch.mm(xp, torch.mm(Sigma, xp.t()))
        
        # print("num",num)
        # print("den_xx",den_xx)
        # print("den_xpxp",den_xpxp)
        
        # print(">>", torch.mm(x, torch.mm(Sigma, x.t())))
        
        
        
        # print("torch.mul(den_xx ,den_xpxp)", torch.mul(den_xx ,den_xpxp))
        
        # print("torch.arcsin(num / torch.sqrt(torch.mul(den_xx ,den_xpxp))", torch.arcsin(num / torch.sqrt(torch.mul(den_xx ,den_xpxp))))
        
        # K = self.sigma_f**2 * torch.arcsin(num / torch.sqrt(torch.mul(den_xx ,den_xpxp)))
        
        x1_ = x1 * torch.tensor([[1/self.lx, 1/self.ly]])
        x2_ = x2 * torch.tensor([[1/self.lx, 1/self.ly]])
        
        # if last_dim_is_batch:
        # x1_ = x1_.transpose(-1, -2).unsqueeze(-1)
        # x2_ = x2_.transpose(-1, -2).unsqueeze(-1)
        
        

        # x1_ = x1.transpose(-1, -2).unsqueeze(-1)

        # if last_dim_is_batch:
        #     x1_ = x1.transpose(-1, -2).unsqueeze(-1)

        # if x1.size() == x2.size() and torch.equal(x1, x2):
        #     # Use RootLinearOperator when x1 == x2 for efficiency when composing
        #     # with other kernels
        #     prod1 = RootLinearOperator(x1_)
        # # else:
        #     x2_ = x2 * self.variance.sqrt()
        #     if last_dim_is_batch:
        #         x2_ = x2_.transpose(-1, -2).unsqueeze(-1)

        JITTER = 1.0e-8
        
        # print(x1.shape)
        # print(x2.shape)

        ones = torch.tensor([[1]])
        beta = torch.tensor([[self.B]])
        num = beta + 2*MatmulLinearOperator(x1_, x2_.transpose(-2, -1),)

        den_x1x1 =  ones + beta +  2*RootLinearOperator(x1_) + torch.eye(x1.shape[0])*JITTER
        den_x2x2 =  ones + beta +  2*RootLinearOperator(x2_) + torch.eye(x2.shape[0])*JITTER


        num = num.to_dense()
        den = torch.matmul(den_x1x1.to_dense(), den_x2x2.to_dense()) #(den_x1x1*den_x2x2).to_dense()

        K = self.sigma_f**2 * torch.arcsin(num / torch.sqrt(den)) + torch.eye(x1.shape[0])*JITTER 
            
        print(K)
        
        return K
