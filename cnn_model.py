from typing import Union, Tuple
import torch
from torch import Tensor
from torchsummary import summary
import torch.nn.functional as F
from torch.nn import Sequential, Linear, BatchNorm1d, PReLU
import torch_geometric
from torch_geometric.typing import PairTensor, Adj, OptTensor, Size
from torch_geometric.nn.conv import MessagePassing
# from torch_scatter import scatter_mean, scatter_add, scatter_max, scatter

from torch_geometric.nn import (
    Set2Set,
    global_mean_pool,
    global_add_pool,
    global_max_pool,
    CGConv,
)


# GNN model
class DOSpredict(torch.nn.Module):
    def __init__(
            self,
            data,
            dim1=128,
            dim2=128,
            pre_fc_count=1,
            gc_count=3,
            batch_norm="True",
            batch_track_stats="True",
            dropout_rate=0.0,
            **kwargs
    ):
        super(DOSpredict, self).__init__()

        if batch_track_stats == "False":
            self.batch_track_stats = False
        else:
            self.batch_track_stats = True
        self.batch_norm = batch_norm
        self.dropout_rate = dropout_rate

        ##Determine gc dimension dimension
        assert gc_count > 0, "Need at least 1 GC layer"
        '''如果没有预处理，则卷积层的输入维度为data.num_features'''
        if pre_fc_count == 0:
            print('预处理层不能为0')
        else:
            self.gc_dim = dim1
        ##Determine post_fc dimension
        if pre_fc_count == 0:
            print('预处理层不能为0')
        else:
            post_fc_dim = dim1




        ##Determine output dimension length
        output_dim = len(data[0].y[0])

        '''data.num_features = 112'''
        ##Set up pre-GNN dense layers
        if pre_fc_count > 0:
            self.pre_lin_list = torch.nn.ModuleList()
            for i in range(pre_fc_count):  #pre_fc_count=1时只会执行一次
                if i == 0:
                    lin = Sequential(torch.nn.Linear(data.num_features, dim1), torch.nn.PReLU())  #创建一个线性层（torch.nn.Linear）和激活函数（torch.nn.PReLU）的序列（Sequential）
                    self.pre_lin_list.append(lin)  #添加到前馈神经网络的 self.pre_lin_list 中
                else:
                    lin = Sequential(torch.nn.Linear(dim1, dim1), torch.nn.PReLU())
                    self.pre_lin_list.append(lin)
        elif pre_fc_count == 0:
            self.pre_lin_list = torch.nn.ModuleList()  #将 self.pre_lin_list 设置为空的 torch.nn.ModuleList

        ##Set up GNN layers
        self.conv_list = torch.nn.ModuleList()
        self.bn_list = torch.nn.ModuleList()
        for i in range(gc_count):
            conv = GC_block(self.gc_dim, data.num_edge_features, aggr="mean")
            # conv = CGConv(self.gc_dim, data.num_edge_features, aggr="mean", batch_norm=False)
            self.conv_list.append(conv)
            if self.batch_norm == "True":
                bn = BatchNorm1d(self.gc_dim, track_running_stats=self.batch_track_stats, affine=True)  #self.batch_track_stats=’True'时会在训练过程中持续追踪整个数据集的均值和方差，常用于训练时
                self.bn_list.append(bn)


        self.dos_mlp = Sequential(Linear(post_fc_dim, dim2),
                                  torch.nn.PReLU(),
                                  Linear(dim2, output_dim),
                                  torch.nn.PReLU(),)
    def forward(self, data):
        ##Pre-GNN dense layers
        for i in range(0, len(self.pre_lin_list)):
            if i == 0:
                out = self.pre_lin_list[i](data.x)
            else:
                out = self.pre_lin_list[i](out)
        ##GNN layers，默认batch_norm = ’True'
        for i in range(0, len(self.conv_list)):
            if len(self.pre_lin_list) == 0 and i == 0:
                if self.batch_norm == "True":
                    out = self.conv_list[i](data.x, data.edge_index, data.edge_attr)   #data.edge_index, data.edge_attr都属于GC_block的**kwargs
                    out = self.bn_list[i](out)
                else:
                    out = self.conv_list[i](data.x, data.edge_index, data.edge_attr)
            else:
                if self.batch_norm == "True":
                    out = self.conv_list[i](out, data.edge_index, data.edge_attr)
                    out = self.bn_list[i](out)
                else:
                    out = self.conv_list[i](out, data.edge_index, data.edge_attr)
        out = F.dropout(out, p=self.dropout_rate, training=self.training)
        ##Post-GNN dense layers
        dos_out = self.dos_mlp(out)  #one last linaer layer + output layer
        return dos_out

'''one graph convolution layer'''
class GC_block(MessagePassing):

    def __init__(self, channels: Union[int, Tuple[int, int]], dim: int = 0, aggr: str = 'mean', **kwargs):
        super(GC_block, self).__init__(aggr=aggr, **kwargs)
        self.channels = channels
        self.dim = dim
        '''如果channels是整数则将它转化为含有两个相同元素的元组'''
        if isinstance(channels, int):
            channels = (channels, channels)

        self.mlp = Sequential(Linear(sum(channels) + dim, channels[1]),  #线性层，输入维度是 sum(channels) + dim，输出维度是 channels[1]
                              torch.nn.PReLU(),)
        self.mlp2 = Sequential(Linear(dim, dim),
                               torch.nn.PReLU(),)

    def forward(self, x: Union[Tensor, PairTensor], edge_index: Adj, edge_attr: OptTensor = None,
                size: Size = None) -> Tensor:
        #如果x是一个单独的张量将其转化为两个张量
        if isinstance(x, Tensor):
            x: PairTensor = (x, x)

        # propagate_type: (x: PairTensor, edge_attr: OptTensor)
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=size)
        out += x[1]
        return out

    def message(self, x_i, x_j, edge_attr: OptTensor) -> Tensor:
        z = torch.cat([x_i, x_j, self.mlp2(edge_attr)], dim=-1)   #将源节点特征 x_i、目标节点特征 x_j 和经过处理的边属性张量 mlp2(edge_attr) 沿着最后一个维度拼接起来，形成一个新的张量 z
        z = self.mlp(z)
        return z


# if __name__ == '__main__':
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     model = DOSpredict().to(device)
#     print(summary(model, (72, 112)))