import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from pymatgen.core import Element
from scipy.ndimage.filters import gaussian_filter1d

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

def feature_generator():
    # file_path = os.path.join('Co_test\\vasp', file_name)
    file_path = 'Co_test\\vasp\\mp-3128.vasp'
    with open(file_path, 'r') as file:
        lines = file.readlines()
    name = file_path.split('\\')[-1].split('.')[0]
    input = []  # 一个 样本的输入
    for i in lines[8:]:  # lines[7]为direct
        u = i.strip().split()  # k = ['0.0000000000000000', '0.0000000000000000', '0.0000000000000000', 'Si']
        element = Element(u[3])
        z = element.Z  # 原子序数
        try:
            valence = element.valence
            valence_num = sum(valence)
        except:
            valence_num = 0
        u.append(z)
        u.append(valence_num)  # 价电子数
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
    return input,dos

input, dos = feature_generator()
model = nnetwork()
save_path = 'model.pth'
model.load_state_dict(torch.load(save_path))
output = model(torch.tensor(input))
output_list = output.detach().numpy()

sigma = 1
fit_target_vals = gaussian_filter1d(dos, sigma)
fit_output_vals = gaussian_filter1d(output_list, sigma)
sequence = [i for i in range(len(dos))]

plt.scatter(sequence, dos, label='Target', s=10)
plt.scatter(sequence, output_list, label='Prediction', s=10)
plt.plot(sequence, fit_target_vals, label='Target Fit')
plt.plot(sequence, fit_output_vals, label='Prediction Fit')
plt.xlabel('Index')
plt.ylabel('Density of state')
plt.legend()
plt.show()