from mp_api.client import MPRester
from emmet.core.summary import HasProps
import glob
import os
from sklearn.manifold import TSNE
from pymatgen.io.cif import CifWriter
import csv
import matplotlib.pyplot as plt
import numpy as np


'''获得所有包含某种元素的所有id'''
def id_finder(elements_name):
	with MPRester("your MP API license") as mpr:
		mp_ids = []
		docs = mpr.summary.search(elements=[elements_name], has_props=[HasProps.dos], fields=["material_id"])
	for i in docs:
		a = i.material_id
		mp_ids.append(a)
	with open(elements_name+'_id.csv', 'w') as file:
		for i in mp_ids:
			file.write(i + '\n')
	print('the length of Co_mp_ids is {}'.format(len(mp_ids)))

'''用于根据id寻找dos,返回两个array: energy and density,并将它们以csv的形式保存'''
def dos_finder(id_csv):
	with open (id_csv,'r') as file:
		lines = file.readlines()
	id_list = []
	for i in lines:
		id_list.append(i.strip())
	for j in id_list[113:]:
		with MPRester("your MP API license") as mpr:
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
	feature_dict = {}   #key:样品名称，value: feature vector
	all_feature =  None    #all_feature会被定义为array, 没有做标记，只能用来画无标记的tSNE图
	for file_path in file_paths:
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
		feature_dict[file_path.split('\\')[1]] = dos_feature
		if all_feature is None:
			all_feature = dos_feature
		else:
			all_feature = np.vstack((all_feature,dos_feature))
	with open('Rh_dos.csv', mode='w', newline='') as file:
		writer = csv.writer(file)
		for key , value in feature_dict.items():
			writer.writerow([key, value.tolist()])
	print(all_feature.shape)
	return all_feature

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
def tsne(all_feature):
	tsne = TSNE(n_components=2, perplexity=2)  # 设置降维后的维度为2
	X_tsne = tsne.fit_transform(all_feature)
	plt.scatter(X_tsne[:, 0], X_tsne[:, 1])
	plt.xlabel('t-SNE Dimension 1')
	plt.ylabel('t-SNE Dimension 2')
	plt.show()

#if __name__ == '__main__':
#	dos_finder()


