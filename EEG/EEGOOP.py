#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  2 10:57:47 2017

OOP Generate Feature extracted set
class DB & FeatureSeriesDB

@author: asceta
"""
import pyedflib
import numpy as np
import scipy as sp
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix

class DB:
    'Common base class for all perceptrons'
    '''

    '''
    def __init__(self, clf, f, Wtime, signalNames, signalHyp):
        
        self.classifier = clf 
        self.sampleFreq = f
        self.sampleT = 1/f
        self.windowTime = Wtime #in seconds
        self.signalNames = signalNames
        self.signalHyp = signalHyp
        self.windowSamples=self.windowTime/self.sampleT
      
    """

    """
   
    def getLists(self):#, signalNames, signalHyp):
        
        signalList=[]
        annotationsList=[]
        
        for i in range(0, len(self.signalNames)):
            
            file = pyedflib.EdfReader(self.signalNames[i])
            #read Fpz-Cz EEG wich is the first
            signalList.append(file.readSignal(0))
            file._close()
            del file
            
            file2 = pyedflib.EdfReader(self.signalHyp[i])
            annotationsList.append(file2.readAnnotations())
            file2._close()
            del file2
            
        return signalList, annotationsList
    
    
    def makeLists(self, signalNames, signalHyp):
        
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
    
    
        #input single channel signal
    def createWindowsBySamples(self, SingleChannelSignal):
        
        wSamples=int(self.windowSamples)
        WindowsInSignal=int(SingleChannelSignal.size/wSamples)
        ArrayOfWindows = np.zeros((WindowsInSignal, wSamples))
        for i in np.arange(WindowsInSignal):
            ArrayOfWindows[i,:]=SingleChannelSignal[(i*wSamples):((i+1)*wSamples)]
        
        return ArrayOfWindows


    def makeWindowsList(self, signalList):
        #100Hz
        #T=0.01
        #WindowTime=30
        #WindowSamples=WindowTime/T
         
        windowsList=[]
        for i in range(0, len(signalList)):
            window = self.createWindowsBySamples(signalList[i])
            windowsList.append(window)
        
        return windowsList


    def labelSignal(self, SingleChannelSignal, SignalAnnotations):
        
        #ArrayOfWindows=createWindowsBySamples(SingleChannelSignal)
        NumberOfWindows=int(SingleChannelSignal.size/self.windowSamples)
        LAnnotation=SignalAnnotations[0].size
        SignalIntervalLabels=SignalAnnotations[0]/self.sampleT
        SignalLabels=SignalAnnotations[2]
        LabelArray=np.empty([NumberOfWindows], dtype='U25')
        start=0
        
        for i in np.arange(LAnnotation-1):
            LabelLenght=SignalIntervalLabels[i+1]-SignalIntervalLabels[i]
            CurrentLabel=SignalLabels[i]
            WindowsInInterval=int(LabelLenght/self.windowSamples)
            for j in np.arange(WindowsInInterval):
                LabelArray[j+start]=CurrentLabel
            start+=WindowsInInterval
        
        return LabelArray
    
    def binaryLabel(self, LabelArray):
    
        BinaryArray = np.zeros((LabelArray.size))
        #print(LabelArray.size)
        for i in np.arange(LabelArray.size):
            #print(i)
            if LabelArray[i]=="":
                LabelArray[i]=LabelArray[i-1]
            if LabelArray[i][-1]=='W':
                BinaryArray[i]=1

        return BinaryArray


    def makeBinaryLabelsList(self, signalList, annotationsList):
    
        biLabList=[]
        #T=0.01
        #WindowTime=30
        #WindowSamples=self.windowTime/self.sampleT
        for i in range(0, len(annotationsList)):
            LabelArray=self.labelSignal(signalList[i], annotationsList[i])
            BinaryLabels=self.binaryLabel(LabelArray)
            biLabList.append(BinaryLabels)
        
        return biLabList
    
    
    def concatSigLab(self, windowsList, biLabList):
        
        DBlist=[]
        for i in range(0, len(biLabList)):
            DBelement=np.concatenate((windowsList[i], biLabList[i].reshape((1, biLabList[i].size)).T), axis=1)
            DBlist.append(DBelement)
        
        return DBlist   
        
    
    def concatList(self, biLabList):
        
        DB = biLabList[0]
        for i in range(0, len(biLabList)-1):
            DB=np.concatenate((DB, biLabList[i+1]), axis=0)
        
        return DB
    
    
    def concatDB(self, windowsList, biLabList):
    
        DBlist=self.concatSigLab(windowsList, biLabList)
        DB=self.concatList(DBlist)
        
        return DB
    
    
    def getDB(self):
    
        signalList, annotationsList = self.getLists()
        windowsList = self.makeWindowsList(signalList)
        biLabList= self.makeBinaryLabelsList(signalList, annotationsList)
        DataBase = self.concatDB(windowsList, biLabList)
        
        return DataBase


    def makeDB(self, signalNames, signalHyp):
        
            signalList, annotationsList = self.makeLists(signalNames, signalHyp)
            windowsList = self.makeWindowsList(signalList)
            biLabList= self.makeBinaryLabelsList(signalList, annotationsList)
            DataBase = self.concatDB(windowsList, biLabList)
            
            return DataBase
    """
        def crosses(self, my_array):
          return ((my_array[:,:-1] * my_array[:,1:]) < 0).sum(axis=1)
        
        def slope_sign_changes(self, my_array):
            w = np.shape(my_array)[1];
            d1 = my_array[:,1:w-2] - my_array[:,2:w-1]
            d2 = my_array[:,2:w-1] - my_array[:,3:w]
            return np.double( np.sum((d1*d2) < 0, 1) );
    """    
    def extr_feat(self, s):
        return s;

    def getFeatAndLabels(self):
        
        DB=self.getDB()
        classes = DB[:,3000];
        features = (self.extr_feat(DB[:,0:3000]));
        
        return features, classes
    
    def makeFeatAndLabels(self, signalNames, signalHyp):
        
        DB=self.makeDB(signalNames, signalHyp)
        classes = DB[:,3000];
        features = (self.extr_feat(DB[:,0:3000]));
        
        return features, classes



    #funcion para calcular desempeños
    def TVFP(self, Confusion):
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
    
    
    def getCrossValidationAcc(self):
    
        Acc=np.zeros((len(self.signalNames),1))
        
        print("\n[Cross Validation with %1.0f subjects]" % len(self.signalNames))
        
        for i in range(0, Acc.size):
            
            #generate every CV Train and test
            signalNamesAux=self.signalNames[:]
            signalHypAux=self.signalHyp[:]
            
            signalNamesTestCV=[]
            signalNamesTestCV.append(signalNamesAux.pop(i))
            signalHypTestCV=[]
            signalHypTestCV.append(signalHypAux.pop(i))
            
            signalNamesTrainCV=signalNamesAux
            signalHypTrainCV=signalHypAux
            
            #feature extraxtion and labels
            caract_train, class_train = self.makeFeatAndLabels(signalNamesTrainCV, signalHypTrainCV)
            caract_test, class_test = self.makeFeatAndLabels(signalNamesTestCV, signalHypTestCV)
        
        
            #se normalizan caracteristicas de cada conjunto, con los parametros del conjunto de train
            scaler = preprocessing.StandardScaler().fit(caract_train);
            caract_train = scaler.transform(caract_train);
            caract_test = scaler.transform(caract_test);
            
            #train classifier
            self.classifier.fit(caract_train, class_train);
            
            #predict
            pred = self.classifier.predict(caract_test);
            #se calculan desempeños
            conf = confusion_matrix(class_test, pred);
            Rates, Acc[i]= self.TVFP(conf)
            print("%1.0f iteration Accuracy: %0.3f)" % (i, Acc[i]*100))
    
        print("Model Overall Accuracy: %0.3f (+/- %0.3f)\n" % (AccMLP.mean()*100, AccMLP.std()*100))
        return Acc
    
      
class FeatureSeriesDB(DB):
    
    def crosses(self, my_array):
      return ((my_array[:,:-1] * my_array[:,1:]) < 0).sum(axis=1)
    
    def slope_sign_changes(self, my_array):
        w = np.shape(my_array)[1];
        d1 = my_array[:,1:w-2] - my_array[:,2:w-1]
        d2 = my_array[:,2:w-1] - my_array[:,3:w]
        return np.double( np.sum((d1*d2) < 0, 1) );
    
    def extr_feat(self, s):
        h = np.shape(s)[0]
        w = np.shape(s)[1];
        feat = np.zeros((h, 10))
        feat[:,0] = np.sum(np.abs(s),1); # iemg
        feat[:,1] = self.crosses(s); #zc
        feat[:,2] = self.slope_sign_changes(s); #ssc
        feat[:,3] = np.sum(np.abs(s[:,1:w-1] - s[:,2:w]),1); # wl
        feat[:,4] = np.sum(np.abs(s[:,1:w-1] - s[:,2:w] > 0.1),1); # wamp
        feat[:,5] = np.var(s,1);
        feat[:,6] = sp.stats.skew(s,1);
        feat[:,7] = sp.stats.kurtosis(s,1);
        feat[:,8] = np.median(s,1);
        feat[:,9] = np.std(s,1);
        return feat;
        
#add method to get accuracy over any set  

#small tests
if __name__ == "__main__":
#    main()
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