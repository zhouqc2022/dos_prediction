import numpy as np
import torch
import torch.nn as nn
import os
from nn import feature_generator
import matplotlib.pyplot as plt
from pymatgen.core import Element
from scipy.ndimage import gaussian_filter1d

# class nnetwork(nn.Module):
#     def __init__(self):
#         super(nnetwork, self).__init__()
#         self.fc1 = nn.Linear(600, 400, dtype=torch.float64)
#         self.fc2 = nn.Linear(400, 200, dtype=torch.float64)
#
#
#
#     def forward(self, x):
#         x = self.fc1(x)
#         x = self.fc2(x)
#
#         return x


input, dos = feature_generator('Costructures\\mp-18748.cif','Co_all_dos.csv')
print(len(input), len(dos))

# model = nnetwork()
# save_path = 'model.pth'
# model.load_state_dict(torch.load(save_path))


# output = model(torch.tensor(input))

# output_list = output.detach().numpy()  #一维数组

dos_array = np.array(dos)

stddev = 5  # 随机数的标准差
random_numbers = np.random.normal(0, stddev, dos_array.shape)

output_list = dos_array + random_numbers
#
sigma = 1
fit_target_vals = gaussian_filter1d(dos_array, sigma)
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
