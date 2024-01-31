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
from sklearn.preprocessing import StandardScaler
from matplotlib.colors import LinearSegmentedColormap


api_key = "GlIXXT78HkkUld1quhdk97sLHjRrcL7W"

'''get all materials id of materials have DOS information'''
def id_finder():
	folder_name = 'id_csv_files'   # 此文件夹包含所有元素的id
	os.makedirs(folder_name, exist_ok = True)
	element_name = input('which element you want to search(if you dont want to specific element just enter)')

	with MPRester(api_key) as mpr:
		mp_ids = []
		query_params = dict(has_props=[HasProps.dos], fields=["material_id"])

	if element_name:
		query_params['elements'] = [element_name]
		docs = mpr.summary.search(**query_params)
		for doc in docs:
			mp_ids.append(doc.material_id)
		file_path = os.path.join(folder_name, element_name + '_id.csv')
		with open(file_path, 'w') as file:
			for i in mp_ids:
				file.write(i + '\n')
		print('id_file is written in {}'.format(file_path))
	else:
		docs = mpr.summary.search(**query_params)
		for doc in docs:
			mp_ids.append(doc.material_id)
		file_path = os.path.join(folder_name,  'all_materials_with_dos_information.csv')
		with open(file_path, 'w') as file:
			for i in mp_ids:
				file.write(i + '\n')
		print('id_file is written in {}'.format(file_path))


'''find dos information according to material id,return two arrays: energy and density and save them'''
def dos_finder():
	all_id_file_folder = 'id_csv_files'
	element_name = input('which element you want to find dos of (if you dont want to specific element just enter)')
	if element_name:
		file_path = os.path.join(all_id_file_folder, element_name +'_id.csv')
		dos_file_folder = os.path.join(all_id_file_folder, element_name +'_dos')
	else:
		file_path = os.path.join(all_id_file_folder, 'all_materials_with_dos_information.csv')
		dos_file_folder = os.path.join(all_id_file_folder, 'all_dos')

	with open (file_path,'r') as file:
		lines = file.readlines()
	id_list = [i.strip() for i in lines]

	exist = [x.split('.csv')[0] for x in os.listdir(dos_file_folder)]
	exist_num = 0
	need_to_add = 0
	for j in id_list:
		if j in exist:
			exist_num += 1
		else:
			need_to_add += 1
			with MPRester(api_key) as mpr:
				dos = mpr.get_dos_by_material_id(j)
			energy = dos.energies
			spin_density = dos.densities  # spin_density is a dict,它的values为两个array
			density = None
			for i in spin_density.values():
				if density is None:
					density = i
				else:
					density += i
			with open(j + '.csv', 'w', newline='') as file:
				writer = csv.writer(file)
				for i in range(len(energy)):
					writer.writerow([energy[i], density[i]])
	print(exist_num, need_to_add)


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

'''从Metal_dos文件夹中读取每一个dos信息文件并计算出d带中心'''
def get_band_center():
	fold_name = [ x +'_dos' for x in vip_element]
	# for i in fold_name:
	# 	file_name = os.listdir(i)
	# 	for j in file_name:
	# 		if j.endswith("band_center"):
	# 			file_path = os.path.join(i, j)
	# 			os.remove(file_path)
	# 			print(f"已删除文件：{file_path}")
	for i in fold_name:
		file_name = glob.glob(os.path.join(i,'./mp*'))
		name_list = []
		band_center_list = []
		for j in file_name[:10]:
			with open (j,'r') as file:
				csv_reader = csv.reader(file)
				energy = float(0)
				energy_density = float(0)
				for row in csv_reader:
					energy = energy + float(row[0])
					a = float(row[0]) * float(row[1])
					energy_density = energy_density + a
					row.append(a)
				band_center = energy_density / energy
			name_list.append('mp-'+ j.split('-')[1])
			band_center_list.append(band_center)
		with open (i+'_band_center_10', 'w', newline = '') as file:
			writer = csv.writer(file)
			for item1, item2 in zip(name_list, band_center_list):
				writer.writerow([item1, item2])

'''根据包含id信息的csv文件生成这些id对应的cif文件'''
def structure_finder_from_csv():
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

'''根据包含dos信息的csv文件生成这些id对应的cif文件'''
def structure_finder_from_dos():
	fold_name = [ x +'_dos' for x in vip_element]
	for i in fold_name:
		new_folder_name = i.split('_')[0]+'structures'
		os.makedirs(new_folder_name, exist_ok=True)
		file_name = glob.glob(os.path.join(i,'./mp*'))
		with MPRester(api_key) as mpr:
			for j in file_name[:10]:
				cif_name = ('mp-'+j.split('-')[1]).split('.')[0]
				structure = mpr.get_structure_by_material_id(cif_name)
				c = CifWriter(structure)
				cif_file_path = os.path.join(new_folder_name, '{}.cif'.format(cif_name))
				c.write_file(cif_file_path)
def dos_filter():
	for element in vip_element:
		new_file_name = element+'_and_O.csv'
		new_lines = []
		with MPRester(api_key) as mpr:
			a = element +'-N'
			docs = mpr.summary.search(elements=[element,'O'], has_props=[HasProps.dos], fields=["material_id"])
			for i in docs:
				id = i.material_id
				print(id)
				name = element +'_all_dos.csv'
				with open (name,'r') as file:
					lines = file.readlines()
					for j in lines:
						if id == j.split(',')[0]:
							new_lines.append(j)
		with open (new_file_name,'w') as file:
			for j in new_lines:

				file.write(j)
		print('{} is finished'.format(element))








'''t-Distributed Stochastic Neighbor Embedding,输入为array,shape为(样本数，dimension)'''
def tsne():
	list = ['Fe_and_O.csv','Co_and_O.csv','Ni_and_O.csv', 'Cu_and_O.csv','Zn_and_O.csv']
	colors = ['red','green', 'blue', 'yellow', 'orange']
	fig = plt.figure()
	ax = fig.add_subplot(111)
	# ax = fig.add_subplot(111, projection='3d')
	all_feature = None
	label_dic = {}
	indics = []

	#creating a mapping from label to color

	for l, m in zip(list, colors):
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
		if all_feature is None:
			all_feature = feature
		else:
			all_feature = np.vstack((all_feature,feature))
		indics.append(all_feature.shape[0])
		label = l.split('.csv')[0]
		label_dic[label] = m

	scaler = StandardScaler()
	feature_normalized = scaler.fit_transform(all_feature)
	tsne = TSNE(n_components=2, perplexity = 10, early_exaggeration = 5)  # 设置降维后的维度为2
	X_tsne = tsne.fit_transform(feature_normalized)

	print(label_dic)
	print(indics)
	index = 0
	# scatter_list = []
	for key, value in label_dic.items():
		if index == 0:
			start_row =  0
			end_row = indics[index]
			scatter = ax.scatter(X_tsne[start_row:end_row,0], X_tsne[start_row:end_row,1], c =value, label=key)
		else:
			start_row = indics[index-1] + 1
			end_row = indics[index]
			scatter = ax.scatter(X_tsne[start_row:end_row, 0], X_tsne[start_row:end_row, 1], c=value, label=key)
		index = index+1
		# scatter_list.append(scatter)
	#
	# for scatter in scatter_list:
	# 	ax.add_collection(scatter)

	# ax.scatter(X_tsne[:, 0], X_tsne[:, 1], c=m, label=l.split('.csv')[0])
	ax.set_xlabel('t-SNE Dimension 1', fontsize=15)
	ax.set_ylabel('t-SNE Dimension 2', fontsize=15)
	ax.autoscale()
	plt.axis('equal')
	plt.legend()
	plt.show()

def structure_finder_for_Cu():
	folder_name = 'Fe_all'
	file_names = os.listdir(folder_name)
	new_file_name = 'Fe_structure_test'
	with MPRester(api_key) as mpr:
		for i in file_names:
			cif_name = ('mp-' + i.split('-')[1]).split('.')[0]
			structure = mpr.get_structure_by_material_id(cif_name)
			c = CifWriter(structure)
			cif_file_path = os.path.join(new_file_name, '{}.cif'.format(cif_name))
			c.write_file(cif_file_path)

if __name__ == '__main__':
	tsne()








#
#
#
#
#
# """==============get one structure=========================
# from pymatgen.io.cif import CifWriter
# from mp_api.client import MPRester
# with MPRester("GlIXXT78HkkUld1quhdk97sLHjRrcL7W") as mpr:
# 	structure = mpr.get_structure_by_material_id("mp-13")
# c=CifWriter(structure)
# c.write_file('mp-13.cif')
# ======================================================="""

# """=============get more than one structure containing at least Cu and O=================
# from pymatgen.io.cif import CifWriter
# from mp_api.client import MPRester
# with MPRester("GlIXXT78HkkUld1quhdk97sLHjRrcL7W") as mpr:
# 	docs=mpr.summary.search(elements=["Cu","O"],fields=["material_id","structure"])
# 	indx=1
# 	for doc in docs:
# 		structure=doc.structure
# 		c=CifWriter(structure)
# 		c.write_file('{}.cif'.format(indx))
# 		indx=indx+1
# ======================================================================================="""
#
# """=================get d band center of one material===========================
#
# from mp_api.client import MPRester
# from pymatgen.electronic_structure.dos import CompleteDos
# from pymatgen.electronic_structure.core import Spin, OrbitalType
#
# with MPRester("GlIXXT78HkkUld1quhdk97sLHjRrcL7W") as mpr:
# 	dos = mpr.get_dos_by_material_id("mp-126")  #<class 'pymatgen.electronic_structure.dos.CompleteDos'>
# 	print("the data type of dos is {}".format(type(dos)))
# 	structure= mpr.get_structure_by_material_id("mp-361")
# 	bandcenter=CompleteDos.get_band_center(dos,erange=[-15,15])#elements=["Cu"], spin=Spin.up
# 	print(bandcenter)
# 	data_type=type(bandcenter)
# 	print(" the data type of band center is {}".format(data_type))
# ===================================================================================="""
#
# """=====================get anll id of materials consisted of Pt=====================================================================
# from mp_api.client import MPRester
# from emmet.core.summary import HasProps
# import pandas as pd
# from pymatgen.io.cif import CifWriter
# with MPRester("GlIXXT78HkkUld1quhdk97sLHjRrcL7W") as mpr:
#
# 	docs=mpr.summary.search(elements=["Fe"],has_props=[HasProps.dos],fields=["material_id"])
# 	indx=0
# 	id_list=list()
# 	for doc in docs:
# 		a=[doc.material_id]   #a is a list
# 		id_list.append(a)
# 		structure= mpr.get_structure_by_material_id(a[0])
# 		c=CifWriter(structure)
# 		c.write_file('{}.cif'.format(indx))
# 		indx=indx+1
# 	id_result=pd.DataFrame(data=id_list)
# 	id_result.to_csv('/home/api-main/id_result.csv',encoding='gbk')
#
#
# ========================================================================================"""


# from mp_api.client import MPRester
# from emmet.core.summary import HasProps
# import pandas as pd
# from pymatgen.io.cif import CifWriter
# import csv
#
#
# with MPRester("GlIXXT78HkkUld1quhdk97sLHjRrcL7W") as mpr:
#
# 	docs=mpr.summary.search(elements=["Co"]) # ,has_props=[HasProps.dos],fields=["material_id"])
# 	id_list=[]
# 	for doc in docs:
# 		a=[doc.material_id]   #a is a list
# 		b=a[0]
# 		id_list.append(b)
# id_result=pd.DataFrame(data=id_list)
# id_result.to_csv('id_result.csv',encoding='gbk')

# with open('id_result.csv','r',encoding='utf-8') as f:
# 	id_list=list(csv.reader(f))
#
# 	indx=0
# 	while indx<len(id_list):
# 		a=id_list[indx]
# 		b=a[1]
# 		structure= mpr.get_structure_by_material_id(b)
# 		c=CifWriter(structure)
# 		c.write_file('{}.cif'.format(indx))
# 		indx=indx+1





# """==================get all id and dos information of materials consisted of only Cu and O==================
# from mp_api.client import MPRester
# from pymatgen.electronic_structure.dos import CompleteDos
# from emmet.core.summary import HasProps
# from pymatgen.electronic_structure.core import Spin, OrbitalType
# import pandas as pd
# from pymatgen.io.cif import CifWriter
# with MPRester("GlIXXT78HkkUld1quhdk97sLHjRrcL7W") as mpr:
#
# 	docs=mpr.summary.search(elements=["Pt"],has_props=[HasProps.dos],fields=["material_id"])
# 	dos_list=list()
# 	indx=0
# 	for doc in docs:
# 		a=[doc.material_id]   #a is a list
# 		dos=mpr.get_dos_by_material_id(a[0])
# 		bandcenter=CompleteDos.get_band_center(dos, erange=[-15,15])
# 		dos_list.append(bandcenter)
# 		structure= mpr.get_structure_by_material_id(a[0])
# 		c=CifWriter(structure)
# 		c.write_file('{}.cif'.format(indx))
# 		indx=indx+1
#
# 	result=pd.DataFrame(data=dos_list)
# 	result.to_csv('/home/api-main/api/result.csv',encoding='gbk')
#
#
# ========================================================================================================"""


# """============================calculate the number of materials with dos information===================
#
# from mp_api.client import MPRester
# from emmet.core.summary import HasProps
#
# with MPRester("GlIXXT78HkkUld1quhdk97sLHjRrcL7W") as mpr:
#
# 	docs=mpr.summary.search(elements=["Au"],has_props=[HasProps.dos])
# 	material_number=len(docs)
# 	print(material_number)
#
# ============================================================================================="""
#
#
#
#
#
#
#
#
#
#
#
#
#
# """======================================================================================================="""
#       def data_write(result.xslx,dos_list):
#               f = xlwt.Workbook()
#               sheet1 = f.add_sheet(u'sheet1',cell_overwrite_ok=True)
#               i = 0
#               for data in dos_list:
#                       for j in range(len(data)):
#                       sheet1.write(i,j,data[j])
#                       i = i + 1
#       f.save(file_path)











#import pandas as pd
#from pymatgen.core.structure import Structure
#from pymatgen.io.cif import CifWriter
#from mp_api.client import MPRester
#with MPRester("GlIXXT78HkkUld1quhdk97sLHjRrcL7W") as mpr:
#   docs = mpr.summary.search(material_ids=["mp-149", "mp-13"]
#from mp_api.client import MPRester
#from emmet.core.summary import HasProps

#with MPRester("GlIXXT78HkkUld1quhdk97sLHjRrcL7W") as mpr:

#    docs = mpr.summary.search(elements=["Cu","O","S"],has_props = [HasProps.electronic_structure],fields=["materials_id","dos"])
#    dos=mpr.get_dos_by_material_id("mp-149")


  ###  structure = docs[0].structure
    
    # -- Shortcut for a single Materials Project ID:
#	structure = mpr.get_structure_by_material_id("mp-13")
#docs.to_csv("docs.csv")
#file = open("aka", "w") 
#file.write(repr(structure) + '\n' )
#file.close() 
#data_type=type(cif)
#print(data_type)
#structure=Structure.from_dic(structure)
#c=CifWriter(structure)
#c.write_file('cif')
