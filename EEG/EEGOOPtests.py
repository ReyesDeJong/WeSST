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
Attributes of DB class
"""   
#signalNames = sorted(glob.glob("/home/asceta/Documents/Fatigue/DB/*PSG.edf"))
signalNames = sorted(glob.glob("/home/asceta/Documents/Fatigue/DB/SC4001E0-PSG.edf"))
#signalHyp = sorted(glob.glob("/home/asceta/Documents/Fatigue/DB/*Hypnogram.edf"))
signalHyp = sorted(glob.glob("/home/asceta/Documents/Fatigue/DB/SC4001EC-Hypnogram.edf"))
   
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
from EEGOOP import FeatureSeriesDB
"""
Generate feature vectors as simple stadistics, and its labels
"""
DBStadisticsFeatMLP = FeatureSeriesDB(MLP, fs, windowTime, signalNames, signalHyp)
features, classes = DBStadisticsFeatMLP.getFeatAndLabels()








#Labels = DBinstanceMLP.concatList(DBinstanceMLP.makeBinaryLabelsList(signalList, annotationList))






















#from EEGOOP import FeatureSeriesDB

#testSubjects=1
#testSubjectIndexs=np.random.randint(len(signalNames),size=testSubjects)
    
#first 12
#signalNamesTrain=signalNames[:len(signalNames)-testSubjects*2]
#signalNamesTest=signalNames[len(signalNames)-testSubjects*2:]
    
#signalHypTrain=signalHyp[:len(signalHyp)-testSubjects*2]
#signalHypTest=signalHyp[len(signalHyp)-testSubjects*2:]
    

    
#%%
#print("\nMLP")
#MLP = MLPClassifier(solver='adam', alpha=1e-5, tol=1e-5, hidden_layer_sizes=(300,50), max_iter = 10000, random_state=1)   
#DBsignalMLP = DB(MLP, fs, windowTime, signalNamesTrain, signalHypTrain)



#AccMLP = DBnormalFeatMLP.getCrossValidationAcc()
#    
#print("\nRF")
#RF = RandomForestClassifier(n_estimators=50,max_leaf_nodes=100,n_jobs=-1,random_state=0)   
#DBnormalFeatRF = DB(RF, fs, windowTime, signalNamesTrain, signalHypTrain)
#AccRF = DBnormalFeatRF.getCrossValidationAcc()