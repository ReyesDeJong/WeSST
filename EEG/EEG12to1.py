#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  1 15:36:48 2017

trained over 12 subjects and 1 for test

@author: asceta
"""


import glob
import pyedflib
import numpy as np



signalNames=sorted(glob.glob("/home/asceta/Documents/Fatigue/DB/*PSG.edf"))
signalHyp=sorted(glob.glob("/home/asceta/Documents/Fatigue/DB/*Hypnogram.edf"))

testSubjects=1
testSubjectIndexs=np.random.randint(len(signalNames),size=testSubjects)

#first 12
signalNamesTrain=signalNames[:len(signalNames)-1]
signalNamesTest=[]
signalNamesTest.append(signalNames[-1])

signalHypTrain=signalHyp[:len(signalHyp)-1]
signalHypTest=[]
signalHypTest.append(signalHyp[-1])


##for random subject 
#signalNamesTest=signalNames.pop(testSubjectIndexs)
#signalNamesTrain=signalNames

#signalHypTest=signalHyp.pop(testSubjectIndexs)
#signalHypTrain=signalHyp

#np.concatenate((data[0],data[1]))
#len(signalHyp)
#%%
def getLists(signalNames, signalHyp):
    signalList=[]
    annotationsList=[]
    for i in range(0, len(signalNames)):
        file = pyedflib.EdfReader(signalNames[i])
        #read Fpz-Cz EEG wich is the first
        signalList.append(file.readSignal(0))
        file._close()
        del file
        
        file2 = pyedflib.EdfReader(signalHyp[i])
        annotationsList.append(file2.readAnnotations())
        file2._close()
        del file2
    return signalList, annotationsList

signalListTrain, annotationsListTrain = getLists(signalNamesTrain, signalHypTrain)
signalListTest, annotationsListTest = getLists(signalNamesTest, signalHypTest)
    
#%%

#input single channel signal
def createWindowsBySamples(SingleChannelSignal, WSizeSamples):
    
    WSizeSamples=int(WSizeSamples)
    WindowsInSignal=int(SingleChannelSignal.size/WSizeSamples)
    ArrayOfWindows = np.zeros((WindowsInSignal, WSizeSamples))
    for i in np.arange(WindowsInSignal):
        ArrayOfWindows[i,:]=SingleChannelSignal[(i*WSizeSamples):((i+1)*WSizeSamples)]
    
    return ArrayOfWindows


def getWindowsList(signalList):
    #100Hz
    T=0.01
    WindowTime=30
    WindowSamples=WindowTime/T
     
    windowsList=[]
    for i in range(0, len(signalList)):
        window = createWindowsBySamples(signalList[i],WindowSamples)
        windowsList.append(window)
    
    return windowsList

windowsListTrain = getWindowsList(signalListTrain)
windowsListTest = getWindowsList(signalListTest)

#%%
#154 lenght 
#1 duracion
#2 duracion intervalo
#SignalAnnotations=annotationsList[0]
#SingleChannelSignal=signalList[0]
#T=0.01
#WindowTime=30
#WindowSamples=WindowTime/T
#WSizeSamples=WindowSamples

def labelSignal(SingleChannelSignal, SignalAnnotations, T, WSizeSamples):
    
    #ArrayOfWindows=createWindowsBySamples(SingleChannelSignal)
    NumberOfWindows=int(SingleChannelSignal.size/WSizeSamples)
    LAnnotation=SignalAnnotations[0].size
    SignalIntervalLabels=SignalAnnotations[0]/T
    SignalLabels=SignalAnnotations[2]
    LabelArray=np.empty([NumberOfWindows], dtype='U25')
    start=0
    
    for i in np.arange(LAnnotation-1):
        LabelLenght=SignalIntervalLabels[i+1]-SignalIntervalLabels[i]
        CurrentLabel=SignalLabels[i]
        WindowsInInterval=int(LabelLenght/WSizeSamples)
        for j in np.arange(WindowsInInterval):
            LabelArray[j+start]=CurrentLabel
        start+=WindowsInInterval
    
    return LabelArray

def binaryLabel(LabelArray):
    
    BinaryArray = np.zeros((LabelArray.size))
    #print(LabelArray.size)
    for i in np.arange(LabelArray.size):
        #print(i)
        if LabelArray[i]=="":
            LabelArray[i]=LabelArray[i-1]
        if LabelArray[i][-1]=='W':
            BinaryArray[i]=1
    return BinaryArray

            
            
#LabelArray=labelSignal(SingleChannelSignal, SignalAnnotations, T, WindowSamples)
#BinaryLabels=binaryLabel(LabelArray)
def getBinaryLabelsList(signalList, annotationsList):
    biLabList=[]
    T=0.01
    WindowTime=30
    WindowSamples=WindowTime/T
    for i in range(0, len(annotationsList)):
        LabelArray=labelSignal(signalList[i], annotationsList[i], T, WindowSamples)
        BinaryLabels=binaryLabel(LabelArray)
        biLabList.append(BinaryLabels)
    
    return biLabList

biLabListTrain=getBinaryLabelsList(signalListTrain, annotationsListTrain)
biLabListTest=getBinaryLabelsList(signalListTest, annotationsListTest)
    
#BinaryLabels=binaryLabel(biLabList[3]) #This one had last 2 labels ""
#%%
#Generate DB as concatenation of things, for random split (if wanted)
#but not recommended due time correlations
def concatSigLab(windowsList, biLabList):
    DBlist=[]
    for i in range(0, len(biLabList)):
        DBelement=np.concatenate((windowsList[i], biLabList[i].reshape((1, biLabList[i].size)).T), axis=1)
        DBlist.append(DBelement)
    return DBlist   
        
def concatList(biLabList):
    DB = biLabList[0]
    for i in range(0, len(biLabList)-1):
        DB=np.concatenate((DB, biLabList[i+1]), axis=0)
    return DB

def getConcatDB(windowsList, biLabList):
    DBlist=concatSigLab(windowsList, biLabList)
    DB=concatList(DBlist)
    return DB

DataBaseTrain = getConcatDB(windowsListTrain, biLabListTrain)
DataBaseTest = getConcatDB(windowsListTest, biLabListTest)
#%%
#import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import sklearn as skl 
from sklearn import model_selection

#Train, Test=model_selection.train_test_split(DataBase, test_size=0.5,
#                                                 stratify=DataBase[:,3000],
#                                                 random_state=21)

        
sign_train = DataBaseTrain[:,0:3000];
class_train = DataBaseTrain[:,3000];

sign_test = DataBaseTest[:,0:3000];
class_test = DataBaseTest[:,3000];
#%%

import numpy as np
import scipy as sp
from sklearn.neural_network import MLPClassifier
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
#Se procede a extraer caracteristicas de las imf calculadas

#funciones de extraccion de caracteristicas entregadas
     
def crosses(my_array):
  return ((my_array[:,:-1] * my_array[:,1:]) < 0).sum(axis=1)

def slope_sign_changes(my_array):
    w = np.shape(my_array)[1];
    d1 = my_array[:,1:w-2] - my_array[:,2:w-1]
    d2 = my_array[:,2:w-1] - my_array[:,3:w]
    return np.double( np.sum((d1*d2) < 0, 1) );

def extr_feat(s):
    h = np.shape(s)[0]
    w = np.shape(s)[1];
    feat = np.zeros((h, 10))
    feat[:,0] = np.sum(np.abs(s),1); # iemg
    feat[:,1] = crosses(s); #zc
    feat[:,2] = slope_sign_changes(s); #ssc
    feat[:,3] = np.sum(np.abs(s[:,1:w-1] - s[:,2:w]),1); # wl
    feat[:,4] = np.sum(np.abs(s[:,1:w-1] - s[:,2:w] > 0.1),1); # wamp
    feat[:,5] = np.var(s,1);
    feat[:,6] = sp.stats.skew(s,1);
    feat[:,7] = sp.stats.kurtosis(s,1);
    feat[:,8] = np.median(s,1);
    feat[:,9] = np.std(s,1);
    return feat;

#se extraen las caracteristicas para Train y Validation
caract_train = (extr_feat(sign_train[:,0:3000]));

caract_test = (extr_feat(sign_test[:,0:3000]));






#%%

#se normalizan caracteristicas de cada conjunto, con los parametros del conjunto de train
scaler = preprocessing.StandardScaler().fit(caract_train);
caract_train = scaler.transform(caract_train);
caract_test = scaler.transform(caract_test);

#%%

#funcion para calcular desempeños
def TVFP(Confusion):
    #se crea arreglo con 0 para alberga TVP y TFP de cada clase
    TVFP=np.zeros((Confusion.shape[0],2))
    #suma sobre elementos de diagonal
    suma=0
    #Se itera sobre las clases de la matriz de confusion, par obtener TVP y TFP de cada
    #una
    for i in range(0, Confusion.shape[0]):
        #TVP se calcula dividiendo la diagonal de cada clase con el numero total
        #muestras de esa clase
        TVP= Confusion[i,i]/np.sum(Confusion[i,:])
        #TFP se calcula sumando los valores de la clase predicha de la clase y restandole el elemento diagonal,
        #luego esto se divide por todos los datos, menos la fila de la clase
        TFP=(np.sum(Confusion[:,i])-Confusion[i,i])/(np.sum(Confusion)-np.sum(Confusion[i,:]))
        TVFP[i,:]=[TVP,TFP]
        #suma diagonales
        suma=suma+Confusion[i,i]
    
    #Al final se agrega el promedio de TVP y TFP de la red   
    PromTVP=np.sum(TVFP[:,0])/TVFP.shape[0]
    PromTFP=np.sum(TVFP[:,1])/TVFP.shape[0]
    Prom=np.array([[PromTVP,PromTFP]])
    TVFP=np.concatenate((TVFP,Prom), axis=0)
    #se retorna el arreglo
    Acc=suma/np.sum(Confusion)
    return TVFP, Acc


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
