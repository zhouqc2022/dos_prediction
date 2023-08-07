import os
import glob
import random
from pymatgen.io.cif import CifParser
import numpy as np
from pymatgen.core.composition import Composition
import torch
import torch.nn as nn
import torch.optim as optim
from pymatgen.core.structure import Structure
from pymatgen.io.vasp.inputs import Poscar
from pymatgen.core import Element

# '''cif_folder is the file folder containing all cifs'''
# def POSCAR_generator(cif_folder):
#     cif_files_path = glob.glob(os.path.join(cif_folder, "*.cif"))
#     for cif_file_path in cif_files_path:
#         structure = Structure.from_file(cif_file_path)
#         poscar_path = os.path.splitext(cif_file_path)[0] + '.vasp'
#         Poscar(structure).write_file(poscar_path)

'''folder is the file folder containing all cifs'''
def data(cif_folder, dos_file_path):
    #输入文件
    file_names = [file for file in os.listdir(cif_folder) if file.endswith('.cif')]
    random.shuffle(file_names)
    train_ratio = 0.7
    val_ratio = 0.2
    test_ratio = 0.1
    total_samples = len(file_names)
    train_samples = int(total_samples * train_ratio)
    val_samples = int(total_samples * val_ratio)
    test_samples = total_samples - train_samples - val_samples

    train_file_names = file_names[:train_samples]
    val_file_names = file_names[train_samples:train_samples + val_samples]
    test_file_names = file_names[train_samples + val_samples:]

    train_data = []
    train_targets = []
    for i in train_file_names:
        path = os.path.join(cif_folder, i)
        input, dos = feature_generator(path, dos_file_path)
        train_data.append(input)
        train_targets.append(dos)

    val_data = []
    val_targets = []
    for i in val_file_names:
        path = os.path.join(cif_folder, i)
        input, dos = feature_generator(path, dos_file_path)
        val_data.append(input)
        val_targets.append(dos)

    test_data = []
    test_targets = []
    for i in test_file_names:
        path = os.path.join(cif_folder, i)
        input, dos = feature_generator(path, dos_file_path)
        test_data.append(input)
        test_targets.append(dos)

    return train_data, train_targets, val_data,\
           val_targets, test_data, test_targets
'''从poscar文件里得到列表input, 从dos.csv文件里得到dos,返回两个list: 输入input_data和输出dos'''
def feature_generator(cif_file_path, dos_file_path):
    name = cif_file_path.split('\\')[-1].split('.')[0]
    input = []
    dos = []
    cif_parser = CifParser(cif_file_path)
    structure = cif_parser.get_structures(primitive=False)[0]
    # 获取sites
    sites = structure.sites
    # 获取坐标
    for site in sites:
        coordination = site.coords
        element = site.specie
        z = element.Z
        try:
            valence = element.valence
            valence_num = sum(valence)
        except:
            valence_num = z

        try:
            x = element.X
        except:
            x = 2.2


        a = np.append(coordination, float(z))
        b = np.append(a, float(valence_num))
        c = np.append(b, float(x))


        input.append(c)
    while len(input) < 100:
        input.append(np.zeros(6,dtype=float))   # 填充全为0的6维向量
    input = [item for sublist in input for item in sublist]

    with open(dos_file_path, 'r') as file:
        vectors = file.readlines()
    for i in vectors:
        if i.startswith(name):
            a = i.split('csv,')[1]
            b = a[2:]
            c = b[:-3]
            d = [float(x) for x in c.split(',')]
            dos.append(d)
    return input, dos
'''input is list of array, dos is a list'''


train_data, train_targets, val_data, val_targets, test_data, test_targets = data('Co_test', 'Co_test\\Co_dos_100.csv')
train_data = torch.tensor(train_data)
train_targets = torch.tensor(train_targets)
val_data = torch.tensor(val_data)
val_targets = torch.tensor(val_targets)
test_data = torch.tensor(test_data)
test_targets = torch.tensor(test_targets)

class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()

    def forward(self, output, target):
        diff = output - target
        loss = torch.mean(torch.norm(diff) / torch.norm(target))
        return loss

class nnetwork(nn.Module):
    def __init__(self):
        super(nnetwork, self).__init__()
        self.fc1 = nn.Linear(600, 600, dtype=torch.float64)
        self.fc2 = nn.Linear(600, 200, dtype=torch.float64)
        self.fc3 = nn.Linear(200, 200, dtype=torch.float64)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x



model = nnetwork()
criterion =  CustomLoss()
# criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)
epochs = 100
for epoch in range(epochs):
    optimizer.zero_grad()
    output = model(train_data)
    loss = criterion(output, train_targets)
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 1 == 0:
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}")
#在验证集上评估模型
with torch.no_grad():
    val_output = model(val_data)
    val_loss = criterion(val_output, val_targets)
    print(f"Validation Loss: {val_loss.item()}")
# 在测试集上评估模型
with torch.no_grad():
    test_output = model(test_data)
    test_loss = criterion(test_output, test_targets)
    print(f"Test Loss: {test_loss.item()}")

save_path = 'model.pth'
torch.save(model.state_dict(), save_path)

