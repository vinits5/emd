import csv
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
import os
import numpy as np
import transforms3d.euler as t3d
import helper
import tensorflow as tf

def print_(text="Test", color='w', style='no', bg_color=''):
	color_dict = {'b': 30, 'r': 31, 'g': 32, 'y': 33, 'bl': 34, 'p': 35, 'c': 36, 'w': 37}
	style_dict = {'no': 0, 'bold': 1, 'underline': 2, 'neg1': 3, 'neg2': 5}
	bg_color_dict = {'b': 40, 'r': 41, 'g': 42, 'y': 43, 'bl': 44, 'p': 45, 'c': 46, 'w': 47}
	if bg_color is not '':
		print("\033[" + str(style_dict[style]) + ";" + str(color_dict[color]) + ";" + str(bg_color_dict[bg_color]) + "m" + text + "\033[00m")
	else: print("\033["+ str(style_dict[style]) + ";" + str(color_dict[color]) + "m"+ text + "\033[00m")


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

