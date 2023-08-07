from mp_api.client import MPRester
from emmet.core.summary import HasProps
import glob
import os
from sklearn.manifold import TSNE
from pymatgen.io.cif import CifWriter
import csv
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

#37种元素
element_list = ['Al','Sc','Ti','V','Cr','Mn','Fe','Co', 'Ni', 'Cu',
				'Zn', 'Ga', 'Ge','Y','Zr', 'Nb','Mo','Ru','Rh','Pd',
				'Ag', 'Cd', 'In', 'Sn', 'Sb','Hf','Ta','W','Re','Os',
				'Ir','Pt','Au','Hg','Tl','Pb','Bi']

'''获得所有包含某种元素的所有id'''
def id_finder():
	folder_name = 'id_csv_files'   # 此文件夹包含所有元素的id
	os.makedirs(folder_name, exist_ok = True)
	with MPRester("GlIXXT78HkkUld1quhdk97sLHjRrcL7W") as mpr:
		for elements_name in element_list:
			mp_ids = []
			docs = mpr.summary.search(elements=[elements_name], has_props=[HasProps.dos], fields=["material_id"])
			for i in docs:
				a = i.material_id
				mp_ids.append(a)
			os.makedirs(folder_name, exist_ok=True)
			file_path = os.path.join(folder_name, elements_name + '_id.csv')
			with open(file_path, 'w') as file:
				for i in mp_ids:
					file.write(i + '\n')
			with open('count.csv', 'a', newline='') as file:
				writer = csv.writer(file)
				writer.writerow([elements_name, len(mp_ids)])

def count_drawer(cout_file_name):
	name_list = []
	count_list = []
	total_num = int(0)
	with open (cout_file_name,'r') as file:
		lines = file.readlines()
	for i in lines:
		name = i.split(',')[0]
		count = int(i.strip().split(',')[-1])
		name_list.append(name)
		count_list.append(count)
		total_num += count
		print('number of materials containing {} is {}'.format(name, count))
	print('total number of materials is {}'.format(total_num))

	fig, ax = plt.subplots(figsize=(16, 4))  #  创建一个图形对象和坐标轴对象
	bars = ax.bar(name_list, count_list, color='olive', alpha=1)
	plt.xlabel('Elements', fontsize=20)
	plt.ylabel('Count', fontsize=20)
	plt.title('Distribution of the  DOS data', fontsize=20)  # weight = 'bold'
	ax.tick_params(axis='both', which = 'both',  labelsize=16) # 设置刻度线
	#plt.xticks(rotation = 45)
	x_min, x_max = ax.get_xlim()
	y_min, y_max = ax.get_ylim()
	print("横轴坐标范围：", x_min, "到", x_max)
	print("纵轴坐标范围：", y_min, "到", y_max)
	ax.text(1, 6500, '~164 k materials', fontsize=30, color='red') # 插入文字
	plt.savefig('count.jpg', dpi=300, bbox_inches='tight')
	plt.show()

'''用于根据id寻找dos,返回两个array: energy and density,并将它们以csv的形式保存'''
def dos_finder():
	all_id_file_folder = 'id_csv_file'
	file_path = os.path.join(all_id_file_folder, 'Cu_id.csv')
	with open (file_path,'r') as file:
		lines = file.readlines()
	id_list = []
	for i in lines:
		id_list.append(i.strip())
	for j in id_list[113:]:
		with MPRester("GlIXXT78HkkUld1quhdk97sLHjRrcL7W") as mpr:
			dos = mpr.get_dos_by_material_id(j)
		energy = dos.energies
		spin_density = dos.densities  # spin_density is a dict,它的values为两个array
		density = None
		for i in spin_density.values():
			if density is None:
				density = i
			else:
				density += i
		with open(j+'.csv', 'w', newline='') as file:
			writer = csv.writer(file)
			for i in range(len(energy)):
				writer.writerow([energy[i], density[i]])

'''从包含dos信息的csv文件获得dos.csv,foder_path是包含这些文件的文件夹'''
def dos_processor(folder_path):
	file_paths = glob.glob(os.path.join(folder_path,'./mp*'))
	feature_dict = {}  #key:样品名称，value: feature vector
	j = 0
	for file_path in file_paths:
		name = file_path.split('\\')[-1].split('.')[0]
		with open (file_path,'r') as file:
			lines = file.readlines()
			energy_list = []
			density_list = []
			for i in lines:
				energy_list.append(i.split(',')[0])
				density_list.append(i.strip().split(',')[1])
		x_list = [float(x) for x in energy_list]
		y_list = [float(x) for x in density_list]
		values = np.linspace(-15, 15, 200)   #dos的范围为-15eV到15eV
		dos_feature = np.zeros(200)
		feature_index = 0
		for i in values:
			nearest_value = min(x_list, key=lambda x: abs(x - i))
			index = x_list.index(nearest_value)
			dos_feature[feature_index] = y_list[index]
			feature_index += 1
		feature_dict[name] = dos_feature
		j += 1
		print(j)
	csv_file_name = folder_path + '_dos.csv'
	with open(csv_file_name, 'w', newline = '') as file:
		for key, value in feature_dict.items():
			line = key +',' +','.join(map(str, value)) +'\n'
			file.write(line)



'''根据包含id信息的csv文件生成这些id对应的cif文件'''
def structure_finder():
	id_csv = 'Co_id.csv'
	with open(id_csv, 'r') as file:
		lines = file.readlines()
	id_list = []
	for i in lines:
		id_list.append(i.strip())
	with MPRester("GlIXXT78HkkUld1quhdk97sLHjRrcL7W") as mpr:
		for j in id_list[:100]:
			structure = mpr.get_structure_by_material_id(j)
			c=CifWriter(structure)
			c.write_file('{}.cif'.format(j))

'''t-Distributed Stochastic Neighbor Embedding,输入为array,shape为(样本数，dimension)'''
def tsne():
	list = ['Co_all_dos.csv','Fe_all_dos.csv','Rh_all_dos.csv']
	colors = ['red','green', 'blue']
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	for l, m in zip(list,colors):
		feature = None
		with open (l, 'r') as file:
			lines = file.readlines()
		for line in lines[:200]:
			a = line.split(',')[1:]
			b = [float(x) for x in a]
			c = np.array(b)
			if feature is None:
				feature = c
			else:
				feature =  np.vstack((feature, c))
		tsne = TSNE(n_components=3, perplexity=10, early_exaggeration = 10)  # 设置降维后的维度为2
		X_tsne = tsne.fit_transform(feature)
		ax.scatter(X_tsne[:, 0], X_tsne[:, 1], X_tsne[:, 2], c=m, label=l)
	ax.set_xlabel('t-SNE Dimension 1',fontsize=15)
	ax.set_ylabel('t-SNE Dimension 2', fontsize=15)
	ax.set_zlabel('t-SNE Dimension 3', fontsize=15)
	plt.legend()
	plt.show()



if __name__ == '__main__':
	tsne()


