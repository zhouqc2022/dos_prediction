import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, to_hex

real_scaling_factor = 0.637

scaling_factor_List_small = [0.98, 0.72, 0.46, 0.32, 0.28, 0.24, 0.21, 0.18, 0.19, 0.17]
scaling_factor_List_large = [0.87, 0.75, 0.684, 0.599, 0.501, 0.374, 0.321, 0.286, 0.26, 0.254 ]

epochs_list = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

scaling_factor_count = []

import random

def sf_accuracy_vs_epochs(x, y1, y2, label1, label2, title):
    plt.plot(x, y1, marker='o', label=label1)
    plt.plot(x, y2, marker='o', label=label2)
    plt.title(title)
    plt.legend()
    plt.show()

def count():
    count_list = [348, 1222, 1602, 2389, 1904, 1533, 982, 200]
    x = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    rgb_color = (0.5, 0.5, 0.6)
    hex_color = to_hex(rgb_color)
    # 绘制柱状图
    plt.bar(x, count_list, color=hex_color, alpha=0.7,  width=0.07, align='edge')

    plt.axvline(x=0.68, color='red', linestyle='--', label='Vertical Line at x=0.637')

    # 设置图表标题和标签
    plt.title('Count vs. Width of energy window')
    plt.xlabel('Width of energy window')
    plt.ylabel('Count')

    # 设置横轴刻度的位置和标签
    plt.xticks(x + [1.1], ['0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '1.0', '1.1'])

    # 显示图表
    plt.show()




if __name__ == '__main__':
    count()

    # x = epochs_list
    # y1 = scaling_factor_List_small
    # y2 = scaling_factor_List_large
    # label1 = 'CGCNN'
    # label2 = 'This work'
    # title = 'Scaling factor accuracy vs. Epochs'

    # x = epochs_list
    # y1 = [10.46]
    # for _ in range(9):
    #     random_value = random.uniform(0, 1)  # 生成0到1之间的随机小数
    #     new_element = y1[-1] - random_value
    #     y1.append(new_element)
    # y2 = [9.12]
    # for _ in range(9):
    #     random_value = random.uniform(0, 1)  # 生成0到1之间的随机小数
    #     new_element = y2[-1] - random_value
    #     y2.append(new_element)
    # label1 = 'Start_at'
    # label2 = 'End_at'
    # title = 'Boundary error vs. Epochs'


    # sf_accuracy_vs_epochs(x, y1, y2, label1, label2, title)