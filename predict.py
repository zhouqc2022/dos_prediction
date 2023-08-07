import numpy as np
import torch
import torch.nn as nn
import os
from nn import feature_generator
import matplotlib.pyplot as plt
from pymatgen.core import Element
from scipy.ndimage import gaussian_filter1d

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



input, dos = feature_generator('Co_test\\mp-493.cif','Co_test\\Co_dos_100.csv')
model = nnetwork()
save_path = 'model.pth'
model.load_state_dict(torch.load(save_path))


output = model(torch.tensor(input))

output_list = output.detach().numpy()  #一维数组

dos_array = np.array(dos[0])

sigma = 1
fit_target_vals = gaussian_filter1d(dos_array, sigma)
fit_output_vals = gaussian_filter1d(output_list, sigma)
sequence = [i for i in range(len(dos[0]))]

plt.scatter(sequence, dos[0], label='Target', s=10)
plt.scatter(sequence, output_list, label='Prediction', s=10)
plt.plot(sequence, fit_target_vals, label='Target Fit')
plt.plot(sequence, fit_output_vals, label='Prediction Fit')
plt.xlabel('Index')
plt.ylabel('Density of state')
plt.legend()
plt.show()
