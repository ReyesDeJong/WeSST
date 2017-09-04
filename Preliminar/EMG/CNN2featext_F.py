import tensorflow as tf
import csv
import numpy as np
import scipy as sp
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import sklearn as skl 

from sklearn.neural_network import MLPClassifier

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from matplotlib.mlab import PCA
import numpy as np
from Dataset import Dataset

#funcion para medir desempeños
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


def new_conv_layer(input,              # Capa anterior.
                   weights,
                   biases,
                   num_input_channels, # Numero de canales de la capa anterior.
                   filter_size,        # Ancho y alto de cada filtro.
                   num_filters,        # Número de filtros.
                   use_pooling=True,   # Usar 2x2 max-pooling.
                   act='relu'):               # F activacion
                                      
    # Forma de los filtros convolucionales (de acuerdo a la API de TF).
    shape = [1,filter_size, num_input_channels, num_filters]
    #[1, 301, 2, 1]

    # Creación de los filtros.


    # Creación de la operación de convolución para TensorFlow.
    # Notar que se han configurado los strides en 1 para todas las dimensiones.
    # El primero y último stride siempre deben ser uno.
    # Si strides=[1, 2, 2, 1], entonces el filtro es movido
    # de 2 en 2 pixeles a lo largo de los ejes x e y de la imagen.
    # padding='SAME' significa que la imagen de entrada se rellena
    # con ceros para que el tamaño de la salida se mantenga.
    
    layer = tf.nn.conv2d(input=input,
                         filter=weights,
                         strides=[1, 1, 1, 1],
                         padding='SAME')

    # Agregar los biases a los resultados de la convolución.
    ##aqui se le suma un bias al resultado de la convolucion completo?
    layer += biases

    # Usar pooling para hacer down-sample de la entrada.
    if use_pooling:
        # Este es 2x2 max pooling, lo que significa que se considera
        # una ventana de 2x2 y se selecciona el valor mayor
        # de los 4 pixeles seleccionados. ksize representa las dimensiones de 
        # la ventana de pooling y el stride define cómo la ventana se mueve por la imagen.
        layer = tf.nn.max_pool(value=layer,
                               ksize=[1, 1, 3, 1],
                               strides=[1, 1, 3, 1],
                               padding='SAME')
    ##este padding en el max pool tiene efecto?
    
    ##se realiza la funcion de activacion sobre  todos los featmap
    
    ##simoide
    if act=='sig':
        layer=tf.nn.sigmoid(layer)
    # Rectified Linear Unit (ReLU).
    if act=='relu':
        layer = tf.nn.relu(layer)

    # La función retorna el resultado de la capa y los pesos aprendidos.
    ##de aqui se pueden obtener filtros
    return layer, weights

def flatten_layer(layer):
    # Obtener dimensiones de la entrada.
    layer_shape = layer.get_shape()

    # Obtener numero de características.
    num_features = layer_shape[1:4].num_elements()
    
    # Redimensionar la salida a [num_images, num_features].
    layer_flat = tf.reshape(layer, [-1, num_features])

    # Las dimensiones de la salida son ahora:
    # [num_images, img_height * img_width * num_channels]
    # Retornar
    return layer_flat, num_features






    

def main(Set):
    Train=np.load(Set)
    #Test=np.load('Test.npy')
    #Val=np.load('Validation.npy') 
    
    
    #class and signals
    sign_train = Train[:,0:6000];
    class_train = Train[:,6000];
    #sign_test = Test[:,0:6000];
    #class_test = Test[:,6000];
    #sign_val = Val[:,0:6000];
    #class_val = Val[:,6000];
    
    
    class_train=class_train.astype(int)
    n_values = np.max(class_train) + 1
    cls_train_oh = np.eye(n_values)[class_train]
    
    #class_val=class_val.astype(int)
    #n_values = np.max(class_val) + 1
    #cls_val_oh = np.eye(n_values)[class_val]
    
    #FFT
    
    #FFt plotting
    ## Number of samplepoints
    #N = 3000
    ## sample spacing
    #Fs = 500; 
    #T = 1.0 / Fs
    #Train=np.load('Train.npy')
    #sign_train = Train[:,0:6000];
    #x = np.linspace(0.0, N*T, N)
    #y = sign_train[signal,3000*EMR:3000+3000*EMR]
    #yf = np.fft.fft(y)
    #xf = np.linspace(0.0, 1.0/(2.0*T), N/2)
    #
    ##plt.subplot(2, 1, 1)
    ##plt.plot(xf, 2.0/N * np.abs(yf[0:int(N/2)]))
    ##plt.subplot(2, 1, 2)
    #plt.plot(xf[1:], 2.0/N * np.abs(yf[0:int(N/2)])[1:])
    
    
    # Number of samplepoints
    N = 3000
    # sample spacing
    Fs = 500; 
    # sampe period
    T = 1.0 / Fs
    
    #first signal fft
    yf1_train = np.fft.fft(sign_train[:,0:3000],axis=1)
    yfft1_train=2.0/N * np.abs(yf1_train[:,0:int(N/2)])[:,1:]
    
   # yf1_val = np.fft.fft(sign_val[:,0:3000],axis=1)
    #yfft1_val=2.0/N * np.abs(yf1_val[:,0:int(N/2)])[:,1:]
    
    #second signal fft
    yf2_train = np.fft.fft(sign_train[:,3000:6000],axis=1)
    yfft2_train=2.0/N * np.abs(yf2_train[:,0:int(N/2)])[:,1:]
    
    #yf2_val = np.fft.fft(sign_val[:,3000:6000],axis=1)
    #yfft2_val=2.0/N * np.abs(yf2_val[:,0:int(N/2)])[:,1:]
    
    #whole signal fft
    #yf = np.fft.fft(sign_train[:,:6000],axis=1)
    #yfft=4.0/N * np.abs(yf[:,0:int(N)])[:,1:]
    
    #for fft plotting
    #xfft = np.linspace(0.0, 1.0/(2.0*T), N)
    #plt.plot(xfft[1:], yfft[1,:])
    
    #plt.plot(xf[1:], yfft2[2,:])
    #plt.plot(xf[1:], yfft1[2,:])
    
    #ffts concat
    fft_train=np.concatenate((yfft1_train,yfft2_train),axis=1)
    #fft_val=np.concatenate((yfft1_val,yfft2_val),axis=1)
    
    
    
    
    
    dataTrain = Dataset(fft_train,cls_train_oh)
    
    #dataVal = Dataset(fft_val,cls_val_oh)
    
    #x,y=    data.next_batch(3)
    #FIND INDEX
    #np.argwhere(np.all(x[0,:] ==sign_train, axis=1, keepdims=True)==True)
    
    w1=np.load('w11.npy')
    w2=np.load('w21.npy') 
    w3=np.load('w31.npy') 
    w4=np.load('w41.npy')  
    b1=np.load('b11.npy')
    b2=np.load('b21.npy') 
    b3=np.load('b31.npy') 
    b4=np.load('b41.npy')  
    
    
    
    
    
    #Especifiacion de filtros para cada capa
    
    # Convolutional Layer 1.
    filter_size1 = 1001          # Filtros son de 5 x 5 pixeles.
    num_filters1 = 10         # Hay 16 de estos filtros.
    
    # Convolutional Layer 2.
    filter_size2 = 501          # Filtros son de 5x5 pixeles.
    num_filters2 = 10         # Hay 16 de estos filtros.
    
    # Convolutional Layer 2.
    filter_size3 = 101          # Filtros son de 5x5 pixeles.
    num_filters3 = 10         # Hay 16 de estos filtros.
    
    
    filter_size4 = 31          # Filtros son de 5x5 pixeles.
    num_filters4 = 10         # Hay 16 de estos filtros.
    
    
    # Número de neuronas de la capa fully-connected.
    
    # Fully-connected layer.
    fc_size1 = 500             # Número de neuronas de la capa fully-connected.
    fc_size2 = 100  
    
    
    #especificaciones imagenes
    
    # Las imágenes de MNIST son de 28 x 28 pixeles.
    img_size = 1499
    # Tamaño de arreglos unidimensionales que podrían
    # guardar los datos de estas imágenes.
    img_size_flat = img_size * 2
    # Tupla que sirve para redimensionar arreglos.
    img_shape = img_size
    # Número de canales de color de las imágenes.
    # Si las imágenes fueran a color, este número sería 3.
    num_channels = 2
    # Número de clases.
    num_classes = 6
    #[1, 1, 3000, 2])
    
    
    #place holders
    x = tf.placeholder(tf.float32, shape=[None, img_size_flat], name='x')
    x_image = tf.reshape(x, [-1, 1, img_size, num_channels])
    
    
    #Armar CNN
    
    layer_conv1, weights_conv1 = \
        new_conv_layer(input=x_image,
                       weights=w1,
                       biases=b1,
                       num_input_channels=num_channels,
                       filter_size=filter_size1,
                       num_filters=num_filters1,
                       use_pooling=True,
                       act='relu')    
        
    layer_conv2, weights_conv2 = \
        new_conv_layer(input=layer_conv1,
                       weights=w2,
                       biases=b2,
                       num_input_channels=num_filters1,
                       filter_size=filter_size2,
                       num_filters=num_filters2,
                       use_pooling=True,
                       act='relu')
    
    layer_conv3, weights_conv3 = \
        new_conv_layer(input=layer_conv2,
                          weights=w3,
                       biases=b3,
                       num_input_channels=num_filters2,
                       filter_size=filter_size3,
                       num_filters=num_filters3,
                       use_pooling=True,
                       act='relu')
        
    layer_conv4, weights_conv4 = \
        new_conv_layer(input=layer_conv3,
                       weights=w4,
                       biases=b4,
                       num_input_channels=num_filters3,
                       filter_size=filter_size4,
                       num_filters=num_filters4,
                       use_pooling=True,
                       act='relu')
        
    layer_flat, num_features = flatten_layer(layer_conv4)
        
    out=layer_flat
    
    #inicio de sesion tf
    session = tf.Session(config=tf.ConfigProto(log_device_placement=True))
    session.run(tf.global_variables_initializer())
    
    
    # Entrenamiento realizado por batches.

    def Feat(test_batch_size, dataset,num_features):
    
        # Número de imagenes en test-set.
        num_set = dataset.data.shape
    
        # Crea arreglo para guardar clases predichas.
        feat = np.zeros(shape=(num_set[0],num_features))
    
        # Calcular clases predichas.
        i = 0
        while i < num_set[0]:
            
            j = min(i + test_batch_size, num_set[0])
            images = dataset.data[i:j, :]
            labels = dataset.labels[i:j, :]
            feed_dict = {x: images}
    
            feat[i:j,:]= session.run(out, feed_dict=feed_dict)
            i = j
       # Labels reales.

        return  feat 
            
    F=Feat(54, dataTrain,num_features)

    
    #V=Feat(54, dataVal,num_features)
    
    return F
#%%
if __name__=="__main__":
    Train=main('Train.npy')
    Val=main('Validation.npy') 
##clf1 = MLPClassifier(solver='adam', alpha=1e-5, tol=1e-5, hidden_layer_sizes=(300, 50), max_iter = 10000, random_state=1);
#clf1 = RandomForestClassifier(n_estimators=50, n_jobs=-1)
##original feat
#
##whole signal
#caract_train = F 
#caract_val = V
#
#clf1.fit(caract_train, class_train);
#pred = clf1.predict(caract_val);
#conf = confusion_matrix(class_val, pred);
#Rates, Acc= TVFP(conf)

