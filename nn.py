import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from pymatgen.core.structure import Structure
from pymatgen.io.vasp.inputs import Poscar
from pymatgen.core import Element

'''cif_folder是包含所有cif文件的文件夹'''
def POSCAR_generator(cif_folder):
    for filename in os.listdir(cif_folder):
        if filename.endswith(".cif"):
            cif_path = os.path.join(cif_folder, filename)
            structure = Structure.from_file(cif_path)
            name = os.path.splitext(filename)[0] + '.vasp'
            poscar_path = os.path.join(cif_folder, name)
            poscar = Poscar(structure)
            poscar.write_file(poscar_path)

'''folder是包含所有结构POSCAR文件的文件夹'''
def data(folder):
    file_names = os.listdir(folder)
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
    return train_file_names, val_file_names, test_file_names

'''以Co_test为例generate input_list, [v_list]'''
def feature_generator(file_names):
    data_list = []
    target_list =[]
    for file_name in file_names:
        file_path = os.path.join('Co_test\\vasp',file_name)
        with open(file_path, 'r') as file:
            lines = file.readlines()
        name = file_path.split('\\')[-1].split('.')[0]
        input = []  #一个 样本的输入
        for i in lines[8:]:  # lines[7]为direct
            u = i.strip().split()  # k = ['0.0000000000000000', '0.0000000000000000', '0.0000000000000000', 'Si']
            element = Element(u[3])
            z = element.Z  #原子序数
            try:
                valence = element.valence
                valence_num = sum(valence)
            except:
                valence_num = 0
            u.append(z)
            u.append(valence_num)   #价电子数
            del u[3]
            u_list = [float(x) for x in u]
            input.append(u_list)
        while len(input) < 100:
            input.append([0.0, 0.0, 0.0, 0.0, 0.0])  # 填充全为0的5维向量
        input = [item for sublist in input for item in sublist]
        # 获得输出向量v
        with open('Co_test\\Co_dos_100.csv', 'r') as file:
            vectors = file.readlines()
        for i in vectors:
            if i.startswith(name):
                a = i.split('csv,')[1]
                b = a[2:]
                c = b[:-3]
                dos = [float(x) for x in c.split(',')]
        data_list.append(input)
        target_list.append([dos])
    return data_list,target_list

class nnetwork(nn.Module):
    def __init__(self):
        super(nnetwork, self).__init__()
        self.fc1 = nn.Linear(500, 200)
        self.fc2 = nn.Linear(200, 200)
        self.fc3 = nn.Linear(200, 200)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

train_file_names, val_file_names, test_file_names = data()

train_data, train_targets = feature_generator(train_file_names)
val_data, val_targets = feature_generator(val_file_names)
test_data, test_targets = feature_generator(test_file_names)


train_data = torch.tensor(train_data)
train_targets = torch.tensor(train_targets)
val_data = torch.tensor(val_data)
val_targets = torch.tensor(val_targets)
test_data = torch.tensor(test_data)
test_targets = torch.tensor(test_targets)

model = nnetwork()
criterion = nn.MSELoss()
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

