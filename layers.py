import torch
import torch.nn as nn
import math
from util import *
import torch.nn.functional as F
##MLP##
class MLP(nn.Module):
    def __init__(self,input_size,out_size,hidden_list=[],batchNorm=True,drop_ratio=0.2,
                    nonlinearity='leaky_relu',negative_slope=0.1,with_output_nonlineartity=True):
        super(MLP,self).__init__()
        self.fcs=nn.ModuleList()
        self.input_size=input_size
        self.out_size=out_size
        self.nonlinearity=nonlinearity
        self.negative_slope=negative_slope
        if hidden_list:
            in_dims=[input_size]+hidden_list
            out_dims=hidden_list+[out_size]
            for i in range(len(in_dims)):
                self.fcs.append(nn.Linear(in_dims[i],out_dims[i]))
                if with_output_nonlineartity or i < len(hidden_flist):
                    if batchNorm:
                        self.fcs.append(nn.BatchNorm1d(out_dims[i], track_running_stats=True))
                    if nonlinearity == 'relu':
                            self.fcs.append(nn.ReLU(inplace=True))
                    elif nonlinearity == 'leaky_relu':
                        self.fcs.append(nn.LeakyReLU(negative_slope, inplace=True))
                    else:
                        #报错
                        raise ValueError("Unsupported nonlinearity {}".format(nonlinearity))
                    if drop_ratio:
                        self.fcs.append(nn.Dropout(drop_ratio))
        else:
            self.fcs.append(nn.Linear(input_size,out_size))
            if with_output_nonlineartity:
                if nonlinearity == 'relu':
                    self.fcs.append(nn.ReLU(inplace=True))
                elif nonlinearity == 'leaky_relu':
                    self.fcs.append(nn.LeakyReLU(negative_slope, inplace=True))
                else:
                    raise ValueError("Unsupported nonlinearity {}".format(nonlinearity))
                    
        self.reset_parameters()
    def reset_parameters(self):
        for l in self.fcs:
            if l.__class__.__name__ == 'Linear':
                nn.init.kaiming_uniform_(l.weight, a=self.negative_slope,
                                            nonlinearity=self.nonlinearity)
                if self.nonlinearity == 'leaky_relu' or self.nonlinearity == 'relu':
                    nn.init.uniform_(l.bias, 0, 0.1)
                else:
                    nn.init.constant_(l.bias, 0.0)
            elif l.__class__.__name__ == 'BatchNorm1d':
                l.reset_parameters()
    def forward(self,x):
        for fc in self.fcs:
            x = fc(x)
        return x

##GINlayer
class GINLayer(nn.Module):
    def __init__(self,mlp, eps=0.0,residual=True,train_eps=True):
        super(GINLayer,self).__init__()
        self.mlp=mlp
        self.initial_eps = eps
        if train_eps:
            self.eps = torch.nn.Parameter(torch.Tensor([eps]))
        else:
            self.register_buffer('eps', torch.Tensor([eps]))
        self.residual=residual
        self.reset_parameters()
        
    def reset_parameters(self):
        self.mlp.reset_parameters()
        self.eps.data.fill_(self.initial_eps)

    def forward(self,input,adj):
        '''
        input:[node_size,emb_size]
        adj:[node_size,node_size]
        '''
        res=input
        neighs=torch.matmul(adj,res)
        res=(1+self.eps)*res+neighs

        res=self.mlp(res)
        if self.residual:
            res=res+input
        return res         

#GIN
class GIN(nn.Module):
    def __init__(self, num_layers, input_size, out_size, hidden_list=[],
                 eps=0.0,drop_ratio=0.2, train_eps=True, residual=True, batchNorm=True,
                 nonlinearity='leaky_relu', negative_slope=0.1):
        super(GIN,self).__init__()
        self.GINLayers=nn.ModuleList()
        if input_size!=out_size:
            first_layer_res=False
        else:
            first_layer_res=True
        self.GINLayers.append(GINLayer(MLP(input_size,out_size,hidden_list,batchNorm,drop_ratio,nonlinearity,negative_slope),eps,first_layer_res))
        for i in range(num_layers-1):
            self.GINLayers.append(GINLayer(MLP(out_size,out_size,hidden_list,batchNorm,drop_ratio,nonlinearity,negative_slope),eps,residual))
            
        self.reset_parameters()
    
    def reset_parameters(self):
        for l in self.GINLayers:
            l.reset_parameters()
    def forward(self,input,adj):
        for gin in self.GINLayers:
            input=gin(input,adj)
        return input



#FD
class FDModel(nn.Module):
    def __init__(self, input_x_size,input_y_size, hidden_size, out_size,
                 in_layers1=1, out_layers=1, batchNorm=False,
                 nonlinearity='leaky_relu',drop_ratio=0.2, negative_slope=0.1):
        super(FDModel, self).__init__()
        hidden_list=[hidden_size]*(in_layers1-1)
        self.out_x=MLP(input_x_size,hidden_size,hidden_list,batchNorm,drop_ratio,nonlinearity,negative_slope)

        self.out_y=nn.Linear(input_y_size,hidden_size)

        hidden_list=[hidden_size]*(out_layers-1)
        self.out=MLP(hidden_size,out_size,hidden_list,batchNorm,drop_ratio,nonlinearity,negative_slope)

        self.reset_parameters()
    
    def reset_parameters(self):
        self.out_x.reset_parameters()
        nn.init.kaiming_uniform_(self.out_y.weight, nonlinearity='sigmoid')
        nn.init.constant_(self.out_y.bias, 0.0)
        self.out.reset_parameters()
    
    def forward(self, x, y):
        x=self.out_x(x)#[b1,hidden]
        y=self.out_y(y)#[b2,hidden]
        out=x.unsqueeze(dim=1)*y.unsqueeze(dim=0)#[b1,b2,h]

        out=self.out(out)#[b1,b2,out_size]
        return out



from util import Init_random_seed as init_random_seed







                

        

                

        


