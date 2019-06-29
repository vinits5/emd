import csv
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
import os
import numpy as np
import transforms3d.euler as t3d
import helper
import tensorflow as tf


###################### Data Handling Operations #########################

# Read names of files in given data_dictionary.
def read_files(data_dict):
	with open(os.path.join('data',data_dict,'files.txt')) as file:
		files = file.readlines()
		files = [x.split()[0] for x in files]
	return files[0]

# Read data from h5 file and return as templates.
def read_h5(file_name):
	import h5py
	f = h5py.File(file_name, 'r')
	templates = np.array(f.get('templates'))
	f.close()
	return templates

# Main function to load data and return as templates array.
def loadData(data_dict):
	files = read_files(data_dict)	# Read file names.
	print(files)
	templates = read_h5(files)		# Read templates from h5 file using given file_name.
	return templates

