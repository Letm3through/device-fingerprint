import os
from glob import glob
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

LOW_PASS_ORDER = 3
f_sample = 100
f_cutoff = 30
nyquist_rate = f_cutoff/(f_sample/2)

SAVE_IMAGE_PATH = "IMAGE"

PATH = "./100hz_data" # state path  of the folder here
EXT = "*.csv"
all_csv_files = [file
                 for path, subdir, files in os.walk(PATH)
                 for file in glob(os.path.join(path, EXT))]

# map csv column name
axis = {1 : 'X-axis', 2 : 'Y-axis', 3 : 'Z-axis'}

# create directory to store processed data images
if not os.path.exists(SAVE_IMAGE_PATH):
	try:
		os.mkdir(SAVE_IMAGE_PATH)
	except OSError:
		print("Error creating" + SAVE_IMAGE_PATH + "directory")
		exit(1)

# creat subdirectory for axis
for i in range(1,4):
	if not os.path.exists(SAVE_IMAGE_PATH + "/" + axis[i]):
		try:
			os.mkdir(SAVE_IMAGE_PATH + "/" + axis[i])
		except OSError:
			print("Error creating" + SAVE_IMAGE_PATH + "directory for axis")
			exit(1)


# noise filter each file
for csv_file in all_csv_files:
	for i in range(1,4):
		data = np.genfromtxt(csv_file, delimiter=',')
		t = data[:,0]
		xn = data[:,i] 

		# order 3 lowpass butterworth filter
		b, a = signal.butter(LOW_PASS_ORDER, nyquist_rate)

		y = signal.filtfilt(b, a, xn)

		plt.figure(num=None, figsize=(20, 20), dpi=80, facecolor='w', edgecolor='k')
		# plot raw signal
		plt.plot(t, xn, 'b', alpha=0.75)

		# plot filtered signal
		plt.plot(t, y, 'k')
		plt.legend(('noisy signal', 'filtfilt'), loc='best')

		plt.grid(True)
		
		# get csv name
		csv_name = csv_file.split('/')[-1]

		print("Running on: " + csv_name)
		plt.savefig(SAVE_IMAGE_PATH + "/"+ axis[i] +"/"+ csv_name[:-3] + 'jpg')
		

