import numpy as np
import torch
import torch.nn as nn
import random
import os
import shutil
#根据标签筛选数据
def get_filter_X(X,Y,ratio):
    # 对标签Y进行频率统计
    unique_labels, label_counts = np.unique(Y, axis=0, return_counts=True)
    s=list(zip(unique_labels, label_counts))
    s=sorted(s,key=lambda x:x[1])
    label_count=int(len(unique_labels)*ratio)
    label_filter=[]
    index_all=[]
    for i in range(label_count):
        temp=s[i][0]
        index=np.where(np.all(Y==temp,axis=1))[0]
        index_all.extend(list(index))

    index_all=np.array(index_all)
    filter_X=X[index_all]
    filter_Y=Y[index_all]
    return filter_X,filter_Y

#标签概率邻接矩阵A：A(ij)=1/2[p(li|lj)+p(li|lj)]
def compute_A_matrix(labels):
    num_labels = len(labels[0])  # 获取标签的数量
    adjacency_matrix = np.zeros((num_labels, num_labels))

    for i in range(num_labels):
        for j in range(num_labels):
            if i != j:
                # 计算标签i和标签j的相关性
                common_count = sum([1 for k in range(len(labels)) if labels[k][i] == 1 and labels[k][j] == 1])
                i_count = sum(labels[k][i] for k in range(len(labels)))
                j_count = sum(labels[k][j] for k in range(len(labels)))
                
                # 计算相关性并更新邻接矩阵
                if i_count != 0 and j_count != 0:
                    p_ij = common_count / i_count
                    p_ji = common_count / j_count
                    adjacency_matrix[i][j] = 0.5 * (p_ij + p_ji)

    return adjacency_matrix

#对偶解码器
def get_double_decoder_loss(labelEmbedding,A):
    sums=0
    labelCount=labelEmbedding.size(0)
    Aij=A+torch.eye(labelCount)
    for i in range(labelCount):
        for j in range(labelCount):
            if i!=j:
                sums+=(F.cosine_similarity(labelEmbedding[i], labelEmbedding[j],dim=0)-Aij[i][j])**2
    sums=sums/(labelCount**2)
    return sums

#获取边
def get_edge(A,device):
    edge_weights = A
    # 找出权重大于0的边
    edge_list = []
    for i in range(edge_weights.size(0)):
        for j in range(edge_weights.size(1)):
            if i != j and edge_weights[i][j] > 0:
                edge_list.append((i, j))

    # 构建COO格式的边列表
    edges = torch.tensor(edge_list, dtype=torch.long).t().to(device)
    return edges


def Init_random_seed(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def clearOldLogs(path):
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        shutil.rmtree(path)
        os.makedirs(path)

class LinkPredictionLoss_cosine(nn.Module):
    def __init__(self):
        super(LinkPredictionLoss_cosine, self).__init__()
        
    def forward(self, emb, adj):
        '''
        Parameters
        ----------
        emb : Tensor
            An MxE tensor, the embedding of the ith node is stored in emb[i,:].
        adj : Tensor
            An MxM tensor, adjacent matrix of the graph.
        
        Returns
        -------
        loss : float
            The link prediction loss.
        '''
        emb_norm = emb.norm(dim=1, keepdim=True)
        emb_norm = emb / (emb_norm + 1e-6)
        adj_pred = torch.matmul(emb_norm, emb_norm.t())
        loss = torch.mean(torch.pow(adj - adj_pred, 2))
        
        return loss

##采样
def gumbel_sigmoid(logits, tau=2/3, gumbel_noise=True, hard=False):
    '''
    Sample from the binary Concrete distribution and optionally discretize.
    Refer to [1][2] for details.
    
    [1] Jang, E., et al. (2017). Categorical reparameterization with gumbel-softmax.
    Proceedings of the 5th International Conference on Learning Representations. Toulon, France.
    [2] Maddison, C. J., et al. (2017). The concrete distribution: A continuous
    relaxation of discrete random variables. Proceedings of the 5th International
    Conference on Learning Representations. Toulon, France.
    '''
    if gumbel_noise:
        uniforms = clamp_probs(torch.rand(logits.size(), device=logits.device, dtype=logits.dtype))
        samples = uniforms.log() - (-uniforms).log1p() + logits #log(uniforms)−log(1+(−uniforms))+logits
    else:
        samples = logits
    y_soft = torch.sigmoid(samples/tau)
    
    if hard:
        # Straight through.
        y_hard = (y_soft > 0.5).float()
        ret = y_hard + y_soft - y_soft.detach() # the gradients will only be backpropagated to y_soft
    else:
        # Reparameterization trick.
        ret = y_soft
    return ret

def clamp_probs(probs):
    eps = torch.finfo(probs.dtype).eps
    return probs.clamp(min=eps, max=1 - eps)

from typing import List, Callable, Union, Any, TypeVar, Tuple

Tensor = TypeVar('torch.tensor')
