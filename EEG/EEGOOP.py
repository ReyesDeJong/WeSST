#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  2 10:57:47 2017

OOP General methodology for Feature extraction, and cross validation
of Data set.

@author: asceta
"""

import pyedflib #libary for reading EDF file
import numpy as np
import scipy as sp
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix

class DB:
    '''
    Contructor for a DB object, that contains the clf to use, the sample frecuency, 
    size of windows in seconds, signal names, signal hypnograms names, signal sample period
    and number of samples per window
    
    @param clf classifier to use
    @param f sample freucency
    @param Wtime windows size in seconds
    @param signalNames path of signal PSGs files
    @param signalHyp path of signal hypnograms
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
    Method that return a list containin the numeric values of the EEG signals
    from every file in signalNames, and a list with the annotation to the respective
    EEG signals, every annotation has a list of the labeled stages, the duration
    and interval of those stages
    
    @return singalList list of EEGs
    @return annotationList list of annotation for each EEG
    """ 
    def getLists(self):
        
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
    
          
    """
    Method that return a list containin the numeric values of the EEG signals
    from every file in signalNames, and a list with the annotation to the respective
    EEG signals, every annotation has a list of the labeled stages, the duration
    and interval of those stages. Same as previous, but can be performed on any
    input file paths
    
    @param signalNames signal PSGs paths
    @param singalHyp signal Hypnogram paths
    
    @return singalList list of EEGs
    @return annotationList list of annotation for each EEG
    """ 
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
    
    
    """
    Method that creates an array of windows for a single channel EEG, the size 
    of the windows is defined by windowSamples.
    
    @param SingleChannelSignal signal to get windows from
    
    @return ArrayOfWindows array of windows for the EEG.
    """ 
    def createWindowsBySamples(self, SingleChannelSignal):
        
        wSamples=int(self.windowSamples)
        WindowsInSignal=int(SingleChannelSignal.size/wSamples)
        ArrayOfWindows = np.zeros((WindowsInSignal, wSamples))
        for i in np.arange(WindowsInSignal):
            ArrayOfWindows[i,:]=SingleChannelSignal[(i*wSamples):((i+1)*wSamples)]
        
        return ArrayOfWindows

    """
    Method that creates an array of windows for every signal in a list, returning
    them as a list of windows arrays.
    
    @param signalList signal list to get windows from
    
    @return windowsList List of array of windows for each EEG file.
    """ 
    def makeWindowsList(self, signalList):
        
        windowsList=[]
        for i in range(0, len(signalList)):
            window = self.createWindowsBySamples(signalList[i])
            windowsList.append(window)
        
        return windowsList

    """
    Method that creates an array of labels for each window in a single channel
    EEG, giving them the label according to the period of sleep stage registered 
    in its respective annotations.
    
    @param SingleChannelSignal signal to get labels from
    @param SignalAnnotation annotations of signal
    
    @return LabelArray array of sleep stages labels.
    """ 
    def labelSignal(self, SingleChannelSignal, SignalAnnotations):
        
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
    
    """
    Method that binarize an array of windows labels in class 1 as awake or
    movement and class 0 the rest. Its important to note that if a window has
    no label, the method assign them previous window label.
    
    @param LabelArray array of EEG windows labels
    
    @return BinaryArray binarize array of labels.
    """ 
    def binaryLabel(self, LabelArray):
    
        BinaryArray = np.zeros((LabelArray.size))
        for i in np.arange(LabelArray.size):
            if LabelArray[i]=="":
                LabelArray[i]=LabelArray[i-1]
            if LabelArray[i][-1]=='W' or LabelArray[i][0]=='M':
                BinaryArray[i]=1

        return BinaryArray

    """
    Method that get list of label arrays from signals windows, makes them
    binary and form a list for each EEG label file.
    
    @param signalList list of EEGs
    @param annotationsList annottions of EEGs
    
    @return biLabList list of binary labels for each EEG window.
    """ 
    def makeBinaryLabelsList(self, signalList, annotationsList):
    
        biLabList=[]
        for i in range(0, len(annotationsList)):
            LabelArray=self.labelSignal(signalList[i], annotationsList[i])
            BinaryLabels=self.binaryLabel(LabelArray)
            biLabList.append(BinaryLabels)
        
        return biLabList
    
    """
    Method that concatenates a window with its label
    
    @param windowsList list of EEGs separated in 30s windows, to concatenate with labels
    @param biLabList list of binary labels for each EEG window, to put at the end 
    of each window
    
    @return DBlist list with feature vector and labels at the end of them.
    """    
    def concatSigLab(self, windowsList, biLabList):
        
        DBlist=[]
        for i in range(0, len(biLabList)):
            DBelement=np.concatenate((windowsList[i], biLabList[i].reshape((1, biLabList[i].size)).T), axis=1)
            DBlist.append(DBelement)
        
        return DBlist   
        
    """
    Method that concatenates a list elements in a numpy array, elements are put as
    a row at the end of the array

    @param biLabList list to be concatenated
    
    @return DB single array with all list elements concatenated.
    """  
    def concatList(self, biLabList):
        
        DB = biLabList[0]
        for i in range(0, len(biLabList)-1):
            DB=np.concatenate((DB, biLabList[i+1]), axis=0)
        
        return DB
    
    """
    Method that concatenates two List in a single array
    
    @param windowsList list of EEGs separated in 30s windows, to concatenate with labels
    @param biLabList list of binary labels for each EEG window, to put at the end 
    of each window
    
    @return DB single array dataset with feature vector and labels at the end of them.
    """    
    def concatDB(self, windowsList, biLabList):
    
        DBlist=self.concatSigLab(windowsList, biLabList)
        DB=self.concatList(DBlist)
        
        return DB
    
    """
    DataSet generation of all EEGs signal files and its labels. This mehots first
    get de EEGs and it's annotations, after ir separate each EEG in windows of 30 seconds
    each, then it gets it's banary labels (class 1 awake or movement), and finally, 
    concatenate all in a single array of feature vectors for each 30 second window
    in the whole dataset, which at the end they have thir corresponding label (1 or 0)
    
    @return DataBase array of data set windows and respective labels.
    """
    def getDB(self):
    
        signalList, annotationsList = self.getLists()
        windowsList = self.makeWindowsList(signalList)
        biLabList= self.makeBinaryLabelsList(signalList, annotationsList)
        DataBase = self.concatDB(windowsList, biLabList)
        
        return DataBase

    """
    DataSet generation of all EEGs signal files and its labels. This mehots first
    get de EEGs and it's annotations, after ir separate each EEG in windows of 30 seconds
    each, then it gets it's banary labels (class 1 awake or movement), and finally, 
    concatenate all in a single array of feature vectors for each 30 second window
    in the whole dataset, which at the end they have thir corresponding label (1 or 0).
    Same as above method, but can be performed on any input file paths.
    
    @param signalNames signal PSGs paths
    @param singalHyp signal Hypnogram paths
    
    @return DataBase array of data set windows and respective labels.
    """
    def makeDB(self, signalNames, signalHyp):
        
            signalList, annotationsList = self.makeLists(signalNames, signalHyp)
            windowsList = self.makeWindowsList(signalList)
            biLabList= self.makeBinaryLabelsList(signalList, annotationsList)
            DataBase = self.concatDB(windowsList, biLabList)
            
            return DataBase
   
    """
    Feature extraction method to be override, in this case it does nothing,
    just return the input
    
    @param s array of windows where features are going to be extracted
    
    @return s input array
    """
    def extr_feat(self, s):
        return s;

    """
    Separate Data set array in labels array and signal, where features are
    extracted
    
    @return fetures feature vectors of Data set.
    @return classes labels for feature vectors
    """
    def getFeatAndLabels(self):
        
        DB=self.getDB()
        classes = DB[:,3000];
        features = (self.extr_feat(DB[:,0:3000]));
        
        return features, classes
    
    """
    Separate Data set array in labels array and signal, where features are
    extracted. Same as above method, but can be performed on any input file paths.
    
    @param signalNames signal PSGs paths
    @param singalHyp signal Hypnogram paths
    
    @return fetures feature vectors of Data set.
    @return classes labels for feature vectors
    """
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
    
        Acc=np.zeros((int(len(self.signalNames)/2),1))#Acc=np.zeros((len(self.signalNames),1))#
        
        print("\n[Cross Validation with %1.0f subjects]" % Acc.size)#len(self.signalNames))#Acc.size
        
        for i in range(0, Acc.size):
            
            #generate every CV Train and test
            signalNamesAux=self.signalNames[:]
            signalHypAux=self.signalHyp[:]
            
            signalNamesTestCV=[]
            #signalNamesTestCV.append(signalNamesAux.pop(i))
            signalNamesTestCV.append(signalNamesAux.pop(i*2))
            signalNamesTestCV.append(signalNamesAux.pop(i*2))
            signalHypTestCV=[]
            #signalHypTestCV.append(signalHypAux.pop(i))
            signalHypTestCV.append(signalHypAux.pop(i*2))
            signalHypTestCV.append(signalHypAux.pop(i*2))
            
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
            print("%1.0f iteration Accuracy: %0.3f" % (i, Acc[i]*100))
    
        print("Model Overall Accuracy: %0.3f (+/- %0.3f)\n" % (Acc.mean()*100, Acc.std()*100))
        return Acc
    
      
class FeatureSeriesDB(DB):
    
    """
    Method to calculare 0 crosses of a signal
    
    @param my_array signal
    
    @return ((my_array[:,:-1] * my_array[:,1:]) < 0).sum(axis=1) number of 0 crosses
    """
    def crosses(self, my_array):
      return ((my_array[:,:-1] * my_array[:,1:]) < 0).sum(axis=1)
    
    """
    Method to calculare number of slope sign changes in a signal
    
    @param my_array signal
    
    @return np.double( np.sum((d1*d2) < 0, 1) ) number of slope sign changes
    """
    def slope_sign_changes(self, my_array):
        w = np.shape(my_array)[1];
        d1 = my_array[:,1:w-2] - my_array[:,2:w-1]
        d2 = my_array[:,2:w-1] - my_array[:,3:w]
        return np.double( np.sum((d1*d2) < 0, 1) )
    
    #@override
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