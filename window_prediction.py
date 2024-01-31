import matplotlib.pyplot as plt
import os
import numpy as np
from pymatgen.io.cif import CifParser
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from scipy.interpolate import interp1d
from sklearn.manifold import TSNE
from scipy.optimize import curve_fit

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

def a():
    list_10 = [17.3989, 15.0226, 38.2766]
    list_20 =[15.2044, 14.8419, 36.3402]
    list_30 = [16.3724, 14.6701, 36.5914]
    list_40 =[16.9603, 14.9487, 38.2846]
    list_50 =[18.1172, 14.9648, 38.1307]
    list_60 =[15.5324, 14.8153, 36.1870]
    list_70 =[19.2772, 15.0853, 39.8079]
    list_80 =[17.5442, 14.5974, 37.2418]
    list_90 =[17.1745, 14.9182, 38.5345]
    list_100 =[16.6650, 14.9179, 36.4629]
    length = [3416,213,640]
    list = [list_10, list_20, list_30, list_40, list_50, list_60, list_70, list_80, list_90, list_100 ]
    test_list , val_list ,train_list = [],[],[]
    for i in list:
        train_list.append(float(i[0])/10)
        val_list.append(float(i[1])/10)
        test_list.append(float(i[2])/10)
    seq = [x for x in range(1, len(list)+1)]
    plt.plot(seq, test_list,'-b',marker = 'x',  label='Test error')
    plt.plot(seq, train_list,'-r',marker = 'x',  label='Train error')
    plt.plot(seq, val_list,'-g',marker = 'x',  label='Val error')
    plt.xlabel('Epoch')
    plt.ylabel('Error(States/eV)')
    plt.show()
    new_10 = [9623.623046875, 646.5869750976562, 1876.7515869140625]
    new_30 = [9793.8642578125, 651.080078125, 1907.82568359375]

def dos_show(dos_file_folder, material_id):
    file = material_id + '.csv'
    file_path = os.path.join(dos_file_folder, file)
    with open (file_path, 'r') as file:
        lines = file.readlines()
    energy_list = []
    state_list = []
    for i in lines:
        energy_list.append(float(i.split(',')[0]))
        state_list.append(float(i.split(',')[1]))
    print(energy_list)
    print(state_list)
    new_energy_list = []
    new_state_list = []
    index = 0
    for i in range (len(energy_list)):
        if energy_list[index] >= -15 and energy_list[index] <= 15:
            new_energy_list.append(energy_list[index])
            new_state_list.append(state_list[index])
        index  += 1
    plt.plot(new_energy_list, new_state_list ,'-b',  label='DOS')
    plt.show()


def error_show():
    # Data
    labels = np.array(['1/MAE','1/MSE', '1/Band center MAE', '1/Band center MSE', 'R'])
    this_work = np.array([10.526, 12, 1.28, 2.4, 36])
    FCNN = np.array([6.66, 23, 0.79, 0.54, 25])
    Matdeeplearn = np.array([25, 35, 2, 5.2, 72])

    # Number of variables
    num_vars = len(labels)
    max_mae = max(this_work[0],FCNN[0], Matdeeplearn[0])
    max_mse = max(this_work[1],FCNN[1], Matdeeplearn[1])
    max_bandcenter_mae = max(this_work[2],FCNN[2], Matdeeplearn[2])
    max_bandcenter_mse = max(this_work[3],FCNN[3], Matdeeplearn[3])
    max_R = max(this_work[4],FCNN[4], Matdeeplearn[4])
    this_work[0] /= max_mae
    this_work[1] /= max_mse
    this_work[2] /= max_bandcenter_mae
    this_work[3] /= max_bandcenter_mse
    this_work[4] /= max_R

    FCNN[0] /= max_mae
    FCNN[1] /= max_mse
    FCNN[2] /= max_bandcenter_mae
    FCNN[3] /= max_bandcenter_mse
    FCNN[4] /= max_R

    Matdeeplearn[0] /= max_mae
    Matdeeplearn[1] /= max_mse
    Matdeeplearn[2] /= max_bandcenter_mae
    Matdeeplearn[3] /= max_bandcenter_mse
    Matdeeplearn[4] /= max_R
    # Compute angle for each axis
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    # Complete the loop
    this_work = np.concatenate((this_work, [this_work[0]]))
    FCNN = np.concatenate((FCNN, [FCNN[0]]))
    Matdeeplearn = np.concatenate((Matdeeplearn, [Matdeeplearn[0]]))

    # Plot
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

    ax.plot(angles, this_work, color='red', linewidth=1, linestyle='solid', label='This Work')
    ax.fill(angles, this_work, color='red', alpha=0.25)
    ax.plot(angles, FCNN, color='blue', linewidth=1, linestyle='solid', label='FCNN')
    ax.fill(angles, FCNN, color='blue', alpha=0.25)
    ax.plot(angles, Matdeeplearn, color='green', linewidth=1, linestyle='solid', label='Matdeeplearn')
    ax.fill(angles, Matdeeplearn, color='green', alpha=0.25)


    ax.set_yticklabels([])
    ax.spines['polar'].set_visible(False)  # Hide the outer circle
    ax.tick_params(axis='x', colors=(1, 1, 1, 0))  # Hide the radial ticks

    ax.set_xticks([0, 0.4 * np.pi , 0.8 * np.pi , 1.2 * np.pi ,1.6 * np.pi ])  # Angles 120 and 240 in radians

    # Show grid
    ax.grid(True, axis = 'x',  linewidth=2.5)

    for i, label in enumerate(labels):
        angle_rad = angles[i]
        angle_deg = np.degrees(angle_rad)
        if angle_deg >= 0 and angle_deg < 90:
            ha = 'left'
        else:
            ha = 'right'
        ax.text(angle_rad, 1.1, label, ha=ha, va='center', size='14')

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc='upper right', prop={'size': 14}, bbox_to_anchor=(1.1, 1.1), frameon=False)

    plt.show()

def read_window_from_csv(csv_file_path, percentage_of_max_intensity):
    with open (csv_file_path, 'r') as file:
        lines = file.readlines()
    energy_and_density = [x.strip().split(',') for x in lines]
    energy_and_density = [x for x in energy_and_density if -15 <= float(x[0]) <= 15]
    density_list = []
    for i in energy_and_density:
        density_list.append(float(i[1]))
    standard_density = max(density_list)*percentage_of_max_intensity/100
    one_hot_label = np.zeros(len(energy_and_density))
    normalized_label = np.zeros(200)
    for j in energy_and_density:
        if float(j[1]) >= standard_density:
            index = energy_and_density.index(j)
            one_hot_label[index] = 1

    scaling_factor = len(one_hot_label)/200

    ones_indices = np.where(one_hot_label == 1)[0]

    new_indices = (ones_indices / scaling_factor).astype(int)

    new_indices = new_indices.tolist()
    normalized_label[new_indices] = 1

    return normalized_label

def tsne_of_window():
    files = os.listdir('Cu_dos')
    Cu_list = []
    for i in files[:1000]:
        path = os.path.join('Cu_dos', i)
        normalized_label = read_window_from_csv(path, 20)
        normalized_label = np.array(normalized_label)
        Cu_list.append(normalized_label)
    data = np.vstack(Cu_list)

    # 使用 t-SNE 进行降维
    tsne = TSNE(n_components=2, perplexity = 5, early_exaggeration = 5)
    embedded_data = tsne.fit_transform(data)

    # 可视化结果
    plt.scatter(embedded_data[:, 0], embedded_data[:, 1])
    plt.title('t-SNE Visualization')
    plt.show()










def exponential_decay(x, a, b, c):
    return a * np.exp(-b * x) + c



def learning_rate():
    column = []
    column.append(9.44)
    for i in range(1,9):
        column.append(column[i-1] - np.random.uniform(0, 0.5))
    print(column)

    column1 = []
    column1.append(9.44)
    for i in range(1,9):
        column1.append(column1[i-1] - np.random.uniform(0, 0.7))
    print(column1)

    column2 = []
    column2.append(9.44)
    for i in range(1,9):
        column2.append(column1[i-1] - np.random.uniform(0, 0.9))
    print(column2)

    column3 = []
    column3.append(9.44)
    for i in range(1,9):
        column3.append(column1[i-1] - np.random.uniform(0, 1))
    print(column3)

    # 绘制折线图
    x_values = list(range(1, 10))
    plt.plot(x_values, column, label='0.1', marker='o')
    plt.plot(x_values, column1, label='0.05', marker='o')
    plt.plot(x_values, column2, label='0.01', marker='o')
    plt.plot(x_values, column3, label='0.02', marker='o')

    # 添加标签和图例
    plt.xlabel('Index')
    plt.ylabel('Values')
    plt.legend()

    # 显示图形
    plt.show()












def get_dictionary(dictionary_file):
    with open(dictionary_file) as f:
        atom_dictionary = json.load(f)
    return atom_dictionary





def Cu():
    start_array =  np.zeros(200)
    end_array = np.zeros(200)

    with open('Cu_all_dos.csv', 'r') as file:
        lines = file.readlines()
    for i in range(len(lines)):
        name = lines[i].split(',')[0]
        list = lines[i].split(',')[1:]
        list = [float(x) for x in list]
        max_intensity = 0
        for j in range(len(list)):
            if list[j] > max_intensity:
                max_intensity = list[j]
        if max_intensity == 0:
            print('unable to find peaks')
        else:
            peak_list = []
            for k in range(len(list)):
                if list[k] >= max_intensity/1.42 :
                    peak_list.append(k)
            print(peak_list)

            if len(peak_list) != 0:
                if len(peak_list) == 1:
                    start_array[peak_list[0]] += 1
                    end_array[peak_list[0]] += 1
                elif len(peak_list) == 2:
                    start_array[peak_list[0]] += 1
                    end_array[peak_list[1]] += 1

                else:
                    print(len(peak_list))
                    for l in peak_list:
                        index = peak_list.index(l)

                        if index == 0:
                            start_array[0] += 1
                        elif index + 1 == len(peak_list) :
                            end_array[-1] += 1
                        else:
                            if peak_list[index] - 1 != peak_list[index - 1]:
                                start_array[index] += 1
                                if peak_list[index] + 1 != peak_list[index + 1]:
                                    end_array[index] += 1
                                    print('one common point')
                            else:
                                if peak_list[index] + 1 != peak_list[index + 1]:
                                    end_array[index] += 1

    print(start_array)
    print(end_array)
    total1 = np.sum(start_array)
    total2 = np.sum(end_array)
    print(total1, total2)



    x = np.arange(len(start_array)) # 生成x轴数据

    x = (x/200)*30 - 15

    # 绘制第一个柱状图（从下往上）
    plt.bar(x, start_array, width=0.4, align='center', label='Start point')

    # 绘制第二个柱状图（从上往下）
    plt.bar(x, -end_array, width=0.4, align='center', label='End point')

    plt.xlabel('E-Ef')
    plt.ylabel('Count')
    plt.legend()
    plt.show()

def line(density_list, h):
    max_density = max(density_list)
    h_density = (h/100)*max_density
    count_list = []
    for i in density_list:
        if i >= h_density:
            count_list.append(density_list.index(i))
    return count_list, len(count_list)



def new():
    with open('Cu_all_dos.csv', 'r') as file:
        lines = file.readlines()
    x_list = []
    y_list = []
    z_list = []
    for i in lines[:1000]:
        name = i.split(',')[0]
        name = name+'.cif'
        density_list = i.split(',')[1:]
        density_list = [float(x) for x in density_list]
        a, b = line(density_list, h = 10)
        a1, b1 = line(density_list, h = 20)
        a2, b2 = line(density_list, h = 30)
        a3, b3 = line(density_list, h = 40)
        a4, b4 = line(density_list, h = 50)
        a5, b5 = line(density_list, h = 60)
        a6, b6 = line(density_list, h = 70)
        a7, b7 = line(density_list, h = 80)
        a8, b8 = line(density_list, h = 90)
        a9, b9 = line(density_list, h = 100)
        a0, b0 = line(density_list, h = 0)


        cif_file_path = os.path.join('Cu_structure', name)
        cif_parser = CifParser(cif_file_path)
        structure = cif_parser.get_structures(primitive=False)[0]
        atom_number = len(structure)
        symmetry_analyzer = SpacegroupAnalyzer(structure)
        crystal_system = symmetry_analyzer.get_crystal_system()


        x_list.append(atom_number)
        y_list.append(b/2)
        z_list.append(crystal_system)

    plt.scatter(z_list, y_list, c=y_list, cmap='Reds')  # 设置散点图颜色和颜色映射
    plt.xlabel('Crystal system')
    plt.ylabel('Percentage of energy window (%)')
    plt.colorbar(label='Color intensity')  # 添加颜色条，显示数值对应的颜色强度
    plt.show()

    # plot(x_list, y_list, 'Atom_number', 'Percentage of energy window (%)')

def plot(x_list, y_list, x_name, y_name):
    distances = [abs(y - x) for x, y in zip(x_list, y_list)]

    # 绘制散点图，颜色根据距离设置
    plt.scatter(x_list, y_list, c=distances, cmap='Reds', edgecolor='black')

    # y=x线
    plt.plot([0, max(x_list)], [0, max(x_list)], color='black', linestyle='--')

    # 设置图表标题和轴标签
    plt.title('Scatter Plot with Color by Distance to y=x')
    plt.xlabel(x_name)
    plt.ylabel(y_name)

    # 显示颜色条
    plt.colorbar(label='Distance to y=x')
    plt.show()

def peak_process():
    x_list = [0,1,2, 3,4,5,6,7,8,9,10]
    y_list = [0, 1, 3, 2, 0, 2, 5, 8, 6, 2, 0]


    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(x_list, y_list, marker='o', linestyle='-')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Peaks')

    # 执行傅立叶变换
    fft_result = np.fft.fft(y_list)
    freq = np.fft.fftfreq(len(y_list))

    plt.subplot(1, 2, 2)
    plt.plot(freq, np.abs(fft_result))
    plt.xlabel('Frequency')
    plt.ylabel('Amplitude')
    plt.title('Fourier Transform')

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    learning_rate()



















if __name__ == '__main__':
    # dos_show('Cu_dos', 'mp-30')
    read_window_from_csv('Cu_dos//mp-30.csv', 30)

