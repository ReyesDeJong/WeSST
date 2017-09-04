#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 21 10:46:48 2017

@author: asceta
"""

import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import sklearn as skl 
from sklearn import model_selection



def loadCsv(filename):
	lines = csv.reader(open(filename, "r")) #rb
	dataset = list(lines)
	for i in range(len(dataset)):
		dataset[i] = [float(x) for x in dataset[i]]
	return np.array(dataset);
 	
#SETEAR CAMINOOOOS
female_1 = loadCsv('/home/asceta/Documents/WeSST/Preliminar/EMG/DB_txt/db2_female_1.txt');
female_2 = loadCsv('/home/asceta/Documents/WeSST/Preliminar/EMG/DB_txt/db2_female_2.txt');
female_3 = loadCsv('/home/asceta/Documents/WeSST/Preliminar/EMG/DB_txt/db2_female_3.txt');
male_1 = loadCsv('/home/asceta/Documents/WeSST/Preliminar/EMG/DB_txt/db2_male_1.txt');
male_2 = loadCsv('/home/asceta/Documents/WeSST/Preliminar/EMG/DB_txt/db2_male_2.txt');

dataset = np.vstack((female_1, female_2, female_3, male_1, male_2));


#%%
#Division en conjunto de Train 80% y Test 20%             

Train, Test=model_selection.train_test_split(dataset, test_size=0.2,
                                                 stratify=dataset[:,6000],
                                                 random_state=21)

#Division de conjunto de Train en conjunto de Train 75%% y Validation 25% 
Train, Val=model_selection.train_test_split(Train, test_size=0.25,
                                                stratify=Train[:,6000],
                                               random_state=20)

#Con esto se obtiene una separacion total de dataset en Train 60% - Test 20% - Validation 20% 

#Se guarda cada conjunto para que se mantengan los mismos datos en cada experimento
np.save('Train.npy', Train)
np.save('Test.npy', Test)
np.save('Validation.npy', Val)
 
#%%

#Histograma de clases por conjunto
fig1 = plt.figure()
col=dataset[:,6000]
n, binswenos, patches = plt.hist(col,bins=11, color='red')

col=Train[:,6000]
n, binswenos, patches = plt.hist(col,bins=11, color='blue')

col=Test[:,6000]
n, binswenos, patches = plt.hist(col,bins=11, color='green')

col=Val[:,6000]
n, binswenos, patches = plt.hist(col,bins=11, color='yellow')

plt.title("Distribution of sets per class")
plt.xlabel("Class")
plt.ylabel("Frequency")
red_patch = mpatches.Patch(color='red', label='Data Base')
green_patch = mpatches.Patch(color='blue', label='Training set')
blue_patch = mpatches.Patch(color='green', label='Test set')
yellow_patch = mpatches.Patch(color='yellow', label='Validation set')
plt.legend(handles=[red_patch,green_patch,blue_patch,yellow_patch])
plt.xticks(np.arange(min(col), max(col)+1, 1.0))
plt.grid(linestyle='--')
plt.show()
#fig1.savefig("Conjuntos.png") 

#%%

#Plot of signals

import numpy as np
#import matplotlib
import matplotlib.pyplot as plt
#remove mpl.use('Agg'), and use fig.set_tight_layout(True)
#matplotlib.use ( 'Agg' )
#from pylab import *
import math

t = np.arange(0.0, 3.0, 0.001)
S1= Train[0,:3000]
S2= Train[0,3000:6000]
S3= Train[1,3000:6000]
t1 = np.arange(0.0, 5.0, 0.1)
t2 = np.arange(0.0, 5.0, 0.02)
t3 = np.arange(0.0, 2.0, 0.01) 

plt.subplot(311)
plt.plot(t, S2, 'b')
plt.grid(True)
plt.title('A tale of 2 subplots')
plt.ylabel('Damped')

plt.subplot(312)
plt.plot(t, S2, 'r')
plt.grid(True)
plt.xlabel('time (s)')
plt.ylabel('Undamped')

plt.subplot(313)
plt.plot(t, S3, 'r')
plt.grid(True)
plt.xlabel('time (s)')
plt.ylabel('Undamped')
#savefig ( 'MatplotlibExample.png' )

#https://stackoverflow.com/questions/27426668/row-titles-for-matplotlib-subplot