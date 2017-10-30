#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  2 15:33:29 2017

Different Feature Extraction methods

@author: asceta
"""

from EEGOOP import FeatureSeriesDB

import glob
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
    
    
signalNames=sorted(glob.glob("/home/asceta/Documents/Fatigue/DB/*PSG.edf"))
signalHyp=sorted(glob.glob("/home/asceta/Documents/Fatigue/DB/*Hypnogram.edf"))
   
testSubjects=1
#testSubjectIndexs=np.random.randint(len(signalNames),size=testSubjects)
    
#first 12
signalNamesTrain=signalNames[:len(signalNames)-testSubjects]
signalNamesTest=[]
signalNamesTest.append(signalNames[-1])
    
signalHypTrain=signalHyp[:len(signalHyp)-testSubjects]
signalHypTest=[]
signalHypTest.append(signalHyp[-1])
    
fs = 100
windowTime = 30
    
print("\nMLP")
MLP = MLPClassifier(solver='adam', alpha=1e-5, tol=1e-5, hidden_layer_sizes=(300,50), max_iter = 10000, random_state=1)   
DBnormalFeatMLP = FeatureSeriesDB(MLP, fs, windowTime, signalNamesTrain, signalHypTrain)
AccMLP = DBnormalFeatMLP.getCrossValidationAcc()
    
print("\nRF")
RF = RandomForestClassifier(n_estimators=50,max_leaf_nodes=100,n_jobs=-1,random_state=0)   
DBnormalFeatRF = FeatureSeriesDB(RF, fs, windowTime, signalNamesTrain, signalHypTrain)
AccRF = DBnormalFeatRF.getCrossValidationAcc()