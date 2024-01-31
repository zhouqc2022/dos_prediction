import os
import sys
import json
import numpy as np
import ase
from ase import io
from scipy.stats import rankdata
from torch_geometric.data import  Dataset, Data, InMemoryDataset
from torch_geometric.loader import DataLoader
from torch_geometric.utils import dense_to_sparse, degree, add_self_loops
import torch_geometric.transforms as T
from torch_geometric.utils import degree
import torch
import torch.nn.functional as F


structure_file_path = 'Ni5111'
dos_file_path = 'Ni_all'
dictionary_file = 'dictionary_default.json'
processing_args = {'graph_max_radius': 5, 'graph_max_neighbor' : 10, "graph_max_neighbors": 10, 'graph_edge_length': 3 }

'''对结构和dos进行处理并将这些信息储存在processed_path/data.pt中'''
def data_processor(structure_file_path, dos_data_path, processed_path):
    data_list = []
    atom_dictionary =  get_dictionary(dictionary_file)
    file_names = [f for f in os.listdir(structure_file_path) if os.path.isfile(os.path.join(structure_file_path, f))]
    for index in range(0, len(file_names)):
        data = Data()
        structure_id = file_names[index]
        ase_crystal = ase.io.read(os.path.join(structure_file_path , structure_id))  #ase.io.read的对象包含后缀
        data.ase = ase_crystal
        distance = ase_crystal.get_all_distances(mic=True)  #distance的shape是（原子数量。原子数量）
    #   Create sparse graph from distance matrix
        distance_trimmed = threshold_sort(distance,processing_args["graph_max_radius"],
                                          processing_args["graph_max_neighbors"],adj=False,)
        ##Create sparse graph from distance matrix
        distance_trimmed = torch.Tensor(distance_trimmed)
        out = dense_to_sparse(distance_trimmed)  #将稠密矩阵转化为稀疏矩阵，维度降低
        edge_index = out[0]
        edge_weight = out[1]
        #是否增加自环边
        self_loops = True
        if self_loops == True:
            edge_index, edge_weight = add_self_loops(edge_index, edge_weight, num_nodes=len(ase_crystal), fill_value=0)
            data.edge_index = edge_index
            data.edge_weight = edge_weight
            distance_mask = (distance_trimmed.fill_diagonal_(1) != 0).int()
        elif self_loops == False:
            data.edge_index = edge_index
            data.edge_weight = edge_weight
            distance_mask = (distance_trimmed.fill_diagonal_(1) != 0).int()
        data.edge_descriptor = {} #data.edge_descriptor 是一个自定义的属性，用于将额外的关于边（edges）的描述信息附加到 PyTorch Geometric 中的数据对象 data 上
        data.edge_descriptor["distance"] = edge_weight
        data.edge_descriptor["mask"] = distance_mask
        #处理dos信息
        dos_file =  os.path.join(dos_data_path, structure_id.split('.')[0] + ".csv")
        with open (dos_file, 'r') as file:
            lines = file.readlines()
            energy_list = []
            density_list = []
            for i in lines:
                energy_list.append(i.split(',')[0])
                density_list.append(i.strip().split(',')[1])
        x_list = [float(x) for x in energy_list]
        y_list = [float(x) for x in density_list]
        values = np.linspace(-15, 15, 200)  # dos的范围为-15eV到15eV
        dos_feature = np.zeros(200)
        feature_index = 0
        for i in values:
            nearest_value = min(x_list, key=lambda x: abs(x - i))
            index = x_list.index(nearest_value)
            dos_feature[feature_index] = y_list[index]
            feature_index += 1
        data.y=torch.Tensor(dos_feature).view(1,-1)
        data_list.append(data)
    #generate node features
    for index in range(0, len(data_list)):#len(file_names)):
        atom_fea = np.vstack([atom_dictionary[str(data_list[index].ase.get_atomic_numbers()[i])]
                for i in range(len(data_list[index].ase))]).astype(float)
        data_list[index].x = torch.Tensor(atom_fea)
    # Adds node degree to node features (appears to improve performance)
    for index in range(0, len(data_list)):
        data_list[index] = OneHotDegree(data_list[index], processing_args["graph_max_neighbors"] + 1)
    #Generate edge features
    distance_gaussian = GaussianSmearing(0, 1, processing_args['graph_edge_length'], 0.2)
    # print(GetRanges(data_list, 'distance'))
    NormalizeEdge(data_list, "distance")
    # print(GetRanges(data_list, 'distance'))
    for index in range(0, len(data_list)):
        data_list[index].edge_attr = distance_gaussian(data_list[index].edge_descriptor["distance"])
    Cleanup(data_list, ["ase", "edge_descriptor"])
    data, slices = InMemoryDataset.collate(data_list)
    #保存数据
    save_file_path = os.path.join(processed_path, "data.pt")
    if os.path.exists(save_file_path):
        os.remove(save_file_path)

    torch.save((data, slices), save_file_path)
    return data, slices

'''将数据加载为structuredataset类的实例'''
def get_dataset(data_path, target_index, reprocess="False", processing_args=None):
    if processing_args == None:
        transforms = GetY(index=target_index)
        if os.path.exists(data_path) == False:
            print('data not found in:', data_path)
            sys.exit()
        if os.path.exists(data_path) == True:
            dataset = StructureDataset(
                data_path,
                'processed',
                transforms,)
    else:
        sys.exit()
    return dataset
class StructureDataset(InMemoryDataset):
    def __init__(
        self, data_path, processed_path="processed", transform=None, pre_transform=None): #接受四个参数
        self.data_path = data_path
        self.processed_path = processed_path
        super(StructureDataset, self).__init__(data_path, transform, pre_transform)  #将三个参数传递给父类
        self.data, self.slices = torch.load(self.processed_paths[0])
        #self.processed_paths 是一个包含所有已处理数据文件路径的列表。索引 [0] 表示你正在加载的是第一个已处理数据文件
    @property
    def raw_file_names(self):
        return []
    @property
    def processed_dir(self):
        return os.path.join(self.data_path, self.processed_path)
    @property
    def processed_file_names(self):
        file_names = ["data.pt"]
        return file_names

'''种子的主要作用是确保每次运行程序时，通过随机生成的序列都是可复制的'''
def split_data(dataset, train_ratio, val_ratio, test_ratio, seed=np.random.randint(1, 1e6), save=False,):
    dataset_size = len(dataset)
    if (train_ratio + val_ratio + test_ratio) <= 1:
        train_length = int(dataset_size * train_ratio)
        val_length = int(dataset_size * val_ratio)
        test_length = int(dataset_size * test_ratio)
        unused_length = dataset_size - train_length - val_length - test_length
        (
            train_dataset,
            val_dataset,
            test_dataset,
            unused_dataset,
        ) = torch.utils.data.random_split(
            dataset,
            [train_length, val_length, test_length, unused_length],
            generator=torch.Generator().manual_seed(seed),
        )
        print(
            "train length:",
            train_length,
            "val length:",
            val_length,
            "test length:",
            test_length,
            "unused length:",
            unused_length,
            "seed :",
            seed,
        )
        return train_dataset, val_dataset, test_dataset
    else:
        print("invalid ratios")

'''the followings are all tools'''
#index = -1时不作修改，否则会将data.y元素的子元素赋值给data.y
class GetY(object):
    def __init__(self, index=0):
        self.index = index

    def __call__(self, data):
        # Specify target.
        if self.index != -1:
            data.y = data.y[0][self.index]
        return data

def get_dictionary(dictionary_file):
    with open(dictionary_file) as f:
        atom_dictionary = json.load(f)
    return atom_dictionary

def OneHotDegree(data, max_degree, in_degree=False, cat=True):
    idx, x = data.edge_index[1 if in_degree else 0], data.x  #获取边的索引和节点特征x
    deg = degree(idx, data.num_nodes, dtype=torch.long)  #计算连接到节点的边的数量
    deg = F.one_hot(deg, num_classes=max_degree + 1).to(torch.float)
    if x is not None and cat:   #检查是否存在节点特征x且cat参数为True
        x = x.view(-1, 1) if x.dim() == 1 else x  #变形为二维张量
        data.x = torch.cat([x, deg.to(x.dtype)], dim=-1)  #将x与edge沿最后一个维度连接起来
    else:
        data.x = deg
    return data

#对输入的矩阵进行排序和筛选，threshold最大半径，neighbors最大近邻
def threshold_sort(matrix, threshold, neighbors, reverse=False, adj=False):
    mask = matrix > threshold
    distance_matrix_trimmed = np.ma.array(matrix, mask=mask)  #创建带有掩码的数组
    # 根据 'reverse' 参数判断是否需要反向排序
    if reverse == False:
        distance_matrix_trimmed = rankdata(distance_matrix_trimmed, method="ordinal", axis=1)
    elif reverse == True:
        distance_matrix_trimmed = rankdata(distance_matrix_trimmed * -1, method="ordinal", axis=1)
    #将distance_matrix_trimmed中值为True的值先替换为nan,然后再替换为0
    distance_matrix_trimmed = np.nan_to_num(np.where(mask, np.nan, distance_matrix_trimmed))
    #将大于neighbor的值替换为0
    distance_matrix_trimmed[distance_matrix_trimmed > neighbors + 1] = 0
    if adj == False:
        # 将等于零的元素替换为原始矩阵中的对应值
        distance_matrix_trimmed = np.where(distance_matrix_trimmed == 0, distance_matrix_trimmed, matrix)
        return distance_matrix_trimmed
    elif adj == True:
        #创建邻接列表（adjacency_list）和邻接属性（adjacency_attributes）
        adj_list = np.zeros((matrix.shape[0], neighbors + 1))
        adj_attr = np.zeros((matrix.shape[0], neighbors + 1))
        for i in range(0, matrix.shape[0]):
            temp = np.where(distance_matrix_trimmed[i] != 0)[0]
            adj_list[i, :] = np.pad(temp, pad_width=(0, neighbors + 1 - len(temp)), mode="constant", constant_values=0,)
            adj_attr[i, :] = matrix[i, adj_list[i, :].astype(int)]
        distance_matrix_trimmed = np.where(distance_matrix_trimmed == 0, distance_matrix_trimmed, matrix)
        return distance_matrix_trimmed, adj_list, adj_attr

def NormalizeEdge(dataset, descriptor_label):
    mean, std, feature_min, feature_max = GetRanges(dataset, descriptor_label)

    for data in dataset:
        data.edge_descriptor[descriptor_label] = (
            data.edge_descriptor[descriptor_label] - feature_min
        ) / (feature_max - feature_min)

def GetRanges(dataset, descriptor_label):
    mean = 0.0
    std = 0.0
    for index in range(0, len(dataset)):
        if len(dataset[index].edge_descriptor[descriptor_label]) > 0:
            if index == 0:
                feature_max = dataset[index].edge_descriptor[descriptor_label].max()
                feature_min = dataset[index].edge_descriptor[descriptor_label].min()
            mean += dataset[index].edge_descriptor[descriptor_label].mean()
            std += dataset[index].edge_descriptor[descriptor_label].std()
            if dataset[index].edge_descriptor[descriptor_label].max() > feature_max:
                feature_max = dataset[index].edge_descriptor[descriptor_label].max()
            if dataset[index].edge_descriptor[descriptor_label].min() < feature_min:
                feature_min = dataset[index].edge_descriptor[descriptor_label].min()

    mean = mean / len(dataset)
    std = std / len(dataset)
    return mean, std, feature_min, feature_max

##Deletes unnecessary data due to slow dataloader
def Cleanup(data_list, entries):
    for data in data_list:
        for entry in entries:
            try:
                delattr(data, entry)
            except Exception:
                pass

class GaussianSmearing(torch.nn.Module):
    def __init__(self, start=0.0, stop=5.0, resolution=50, width=0.05, **kwargs):
        super(GaussianSmearing, self).__init__()
        offset = torch.linspace(start, stop, resolution)
        # self.coeff = -0.5 / (offset[1] - offset[0]).item() ** 2
        self.coeff = -0.5 / ((stop - start) * width) ** 2
        self.register_buffer("offset", offset)

    def forward(self, dist):
        dist = dist.unsqueeze(-1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))


if __name__ == '__main__':
    data_processor('Cu_structure','Cu_dos', 'processed_path')





