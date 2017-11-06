#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  2 15:33:29 2017

Explanation of EEGOOP

@author: asceta
"""

"""
Libraries, and classes importations
"""
from EEGOOP import DB

import glob
from sklearn.neural_network import MLPClassifier    
 
"""
Files Path & attributes of DB class
""" 
DatasetPath = "/home/asceta/Documents/Fatigue/DB/"
  
signalNames = sorted(glob.glob(DatasetPath+"*PSG.edf"))
#signalNames = sorted(glob.glob("/home/asceta/Documents/Fatigue/DB/SC4001E0-PSG.edf"))
signalHyp = sorted(glob.glob(DatasetPath+"*Hypnogram.edf"))
#signalHyp = sorted(glob.glob("/home/asceta/Documents/Fatigue/DB/SC4001EC-Hypnogram.edf"))
   
fs = 100
windowTime = 30

MLP = MLPClassifier(solver='adam', alpha=1e-5, tol=1e-5, hidden_layer_sizes=(300,50), max_iter = 10000, random_state=1)   

"""
DB class instantiation
""" 
DBinstanceMLP = DB(MLP, fs, windowTime, signalNames, signalHyp)
#%%
"""
Listo of EEG and its annotations
"""
signalList, annotationList = DBinstanceMLP.getLists()
#%%
"""
Generate DataSet for trainning and testing
"""
dataSet = DBinstanceMLP.getDB()

#%%
"""
Generate feature vectors as whole signal, and its labels
"""
features, classes = DBinstanceMLP.getFeatAndLabels()
#%%
"""
FeatureSeriesDB class instantiation, for new feature extraction method
""" 
from EEGOOP import FeatureSeriesDB

DBStadisticsFeatMLP = FeatureSeriesDB(MLP, fs, windowTime, signalNames, signalHyp)

#%%
"""
Generate feature vectors as simple stadistics, and its labels
"""
features, classes = DBStadisticsFeatMLP.getFeatAndLabels()
#%%
"""
Perform cross validation on whole dataset (7 iterations) (4 files of EEG are nedeed
as minimum)
"""
AccMLP = DBStadisticsFeatMLP.getCrossValidationAcc()

