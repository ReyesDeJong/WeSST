#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  1 15:36:48 2017

cv of 13 and 1 test

@author: asceta
"""


import glob
#import pyedflib
import numpy as np
#import matplotlib.pyplot as plt
#import matplotlib.patches as mpatches
#import sklearn as skl 
#from sklearn import model_selection
#import numpy as np
#import scipy as sp
from sklearn.neural_network import MLPClassifier
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

from Functions import getLists
from Functions import getWindowsList
from Functions import getBinaryLabelsList
from Functions import getConcatDB
from Functions import extr_feat
from Functions import TVFP



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

def getDB(signalNames, signalHyp):
    
    signalList, annotationsList = getLists(signalNames, signalHyp)
    windowsList = getWindowsList(signalList)
    biLabList=getBinaryLabelsList(signalList, annotationsList)
    DataBase = getConcatDB(windowsList, biLabList)
    
    return DataBase



#DataBaseTrain = getDB(signalNamesTrain, signalHypTrain)
#DataBaseTest = getDB(signalNamesTest, signalHypTest)

def getFeatAndLabels(signalNames, signalHyp):
    
    DB=getDB(signalNames, signalHyp)
    classes = DB[:,3000];
    features = (extr_feat(DB[:,0:3000]));
    
    return features, classes
    
 


#%%
    
#CV
#clf = RandomForestClassifier(n_estimators=50,max_leaf_nodes=100,n_jobs=-1,random_state=0)
#clf = MLPClassifier(solver='adam', alpha=1e-5, tol=1e-5, hidden_layer_sizes=(300,50), max_iter = 10000, random_state=1);
#%%
def crossValidation(signalNamesTrain, signalHypTrain, clf):

    Acc=np.zeros((len(signalNamesTrain),1))
    
    for i in range(0, Acc.size):
        
        #generate every CV Train and test
        signalNamesAux=signalNamesTrain[:]
        signalHypAux=signalHypTrain[:]
        
        signalNamesTestCV=[]
        signalNamesTestCV.append(signalNamesAux.pop(i))
        signalHypTestCV=[]
        signalHypTestCV.append(signalHypAux.pop(i))
        
        signalNamesTrainCV=signalNamesAux
        signalHypTrainCV=signalHypAux
        
        #feature extraxtion and labels
        caract_train, class_train = getFeatAndLabels(signalNamesTrainCV, signalHypTrainCV)
        caract_test, class_test = getFeatAndLabels(signalNamesTestCV, signalHypTestCV)
    
    
        #se normalizan caracteristicas de cada conjunto, con los parametros del conjunto de train
        scaler = preprocessing.StandardScaler().fit(caract_train);
        caract_train = scaler.transform(caract_train);
        caract_test = scaler.transform(caract_test);
        
        #train classifier
        clf.fit(caract_train, class_train);
        
        #predict
        pred = clf.predict(caract_test);
        #se calculan desempeños
        conf = confusion_matrix( class_test, pred);
        Rates, Acc[i]= TVFP(conf)

    return Acc


#%%
    
MLP = MLPClassifier(solver='adam', alpha=1e-5, tol=1e-5, hidden_layer_sizes=(300,50), max_iter = 10000, random_state=1)
AccMLP = crossValidation(signalNamesTrain, signalHypTrain, MLP) 
print("MLP Accuracy: %0.3f (+/- %0.3f)" % (AccMLP.mean()*100, AccMLP.std()*100))

#%%
RF = RandomForestClassifier(n_estimators=50,max_leaf_nodes=100,n_jobs=-1,random_state=0)
AccRF = crossValidation(signalNamesTrain, signalHypTrain, RF) 
print("RF Accuracy: %0.3f (+/- %0.3f)" % (AccRF.mean()*100, AccRF.std()*100))
#%%
"""    
caract_train, class_train = getFeatAndLabels(signalNamesTrain, signalHypTrain)
caract_test, class_test = getFeatAndLabels(signalNamesTest, signalHypTest)

#se normalizan caracteristicas de cada conjunto, con los parametros del conjunto de train
scaler = preprocessing.StandardScaler().fit(caract_train);
caract_train = scaler.transform(caract_train);
caract_test = scaler.transform(caract_test);


#%%
#se entrena con MLP
clf = MLPClassifier(solver='adam', alpha=1e-5, tol=1e-5, hidden_layer_sizes=(300,50), max_iter = 10000, random_state=1);
clf.fit(caract_train, class_train);

#se clasifica conjunto de validacion
pred = clf.predict(caract_test);

#se calculan desempeños
conf = confusion_matrix( class_test, pred);
Rates, Acc= TVFP(conf)
print('MLP')
print('Acc= ', Acc*100 );
#print('TVP= ', Rates[6,0]*100 );
#print('TFP= ', Rates[6,1]*100 );
print( conf );
#%%

#se repite el proceso anterior pero para un clasificador Random Forest
clf = RandomForestClassifier(n_estimators=50,max_leaf_nodes=100,n_jobs=-1,random_state=0)
clf.fit(caract_train, class_train);
pred = clf.predict(caract_test);

conf = confusion_matrix( class_test, pred);
Rates, Acc= TVFP(conf)
print('RF')
print('Acc= ', Acc*100 );
#print('TVP= ', Rates[6,0]*100 );
#print('TFP= ', Rates[6,1]*100 );
print( conf );
    
#7950000=22.0833333hrs
#sigbufs2 = np.zeros((n2, f2.getNSamples()))
#T=0.01
#import main_load_edf
#
#psg_dir = 'SC4001E0-PSG.edf'
#ann_dir = 'SC4001EC-Hypnogram.edf'
#
#main_load_edf(psg_dir, ann_dir, "file")
"""