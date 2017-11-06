#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  2 15:33:29 2017

Different Feature Extraction methods

@author: asceta
"""

"""
Libraries, and classes importations
"""
from EEGOOP import FeatureSeriesDB

import glob
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier

"""
Files Path, fix number of test subjects, sample frecuency and sample window size
"""    
DatasetPath = "/home/asceta/Documents/Fatigue/DB/"
    
signalNames=sorted(glob.glob(DatasetPath+"*PSG.edf"))
signalHyp=sorted(glob.glob(DatasetPath+"*Hypnogram.edf"))
   
testSubjects=1
#testSubjectIndexs=np.random.randint(len(signalNames),size=testSubjects)
    
#first 12
signalNamesTrain=signalNames[:len(signalNames)-testSubjects*2]
signalNamesTest=signalNames[len(signalNames)-testSubjects*2:]
    
signalHypTrain=signalHyp[:len(signalHyp)-testSubjects*2]
signalHypTest=signalHyp[len(signalHyp)-testSubjects*2:]
    
fs = 100
windowTime = 30
    
#%%
"""
Create instances of FeatureSeriesDB for a MLP & RF, and perform CV
"""  
MLP = MLPClassifier(solver='adam', alpha=1e-5, tol=1e-5, hidden_layer_sizes=(300,50), max_iter = 10000, random_state=1)   
DBnormalFeatMLP = FeatureSeriesDB(MLP, fs, windowTime, signalNamesTrain, signalHypTrain)
    
RF = RandomForestClassifier(n_estimators=50, max_leaf_nodes=100, n_jobs=-1, random_state=0)   
DBnormalFeatRF = FeatureSeriesDB(RF, fs, windowTime, signalNamesTrain, signalHypTrain)

#%%
"""
Perform CV
"""
print("\nMLP CV")
AccMLP = DBnormalFeatMLP.getCrossValidationAcc()

print("\nMLP CV")
AccRF = DBnormalFeatRF.getCrossValidationAcc()

#%%
"""
Get test accuracy of models
""" 
print("\nMLP test Accuracy")
testAccMLP = DBnormalFeatMLP.getTestAcc(signalNamesTest, signalHypTest) 

print("\nRF test Accuracy")
testAccRF = DBnormalFeatRF.getTestAcc(signalNamesTest, signalHypTest) 