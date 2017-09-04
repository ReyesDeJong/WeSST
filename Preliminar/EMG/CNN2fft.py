import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix
import time
from datetime import timedelta
import math
import CNN_builder as NN
import matplotlib.patches as mpatches
from IPython import get_ipython


#importar datos
from tensorflow.examples.tutorials.mnist import input_data
import itertools

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Greens):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=14)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label',fontsize=12)
    plt.xlabel('Predicted label',fontsize=12)

Train=np.load('Train.npy')
Test=np.load('Test.npy')
Val=np.load('Validation.npy') 


#class and signals
sign_train = Train[:,0:6000];
class_train = Train[:,6000];
sign_test = Test[:,0:6000];
class_test = Test[:,6000];
sign_val = Val[:,0:6000];
class_val = Val[:,6000];


class_train=class_train.astype(int)
n_values = np.max(class_train) + 1
cls_train_oh = np.eye(n_values)[class_train]

class_val=class_val.astype(int)
n_values = np.max(class_val) + 1
cls_val_oh = np.eye(n_values)[class_val]

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

yf1_val = np.fft.fft(sign_val[:,0:3000],axis=1)
yfft1_val=2.0/N * np.abs(yf1_val[:,0:int(N/2)])[:,1:]

#second signal fft
yf2_train = np.fft.fft(sign_train[:,3000:6000],axis=1)
yfft2_train=2.0/N * np.abs(yf2_train[:,0:int(N/2)])[:,1:]

yf2_val = np.fft.fft(sign_val[:,3000:6000],axis=1)
yfft2_val=2.0/N * np.abs(yf2_val[:,0:int(N/2)])[:,1:]

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
fft_val=np.concatenate((yfft1_val,yfft2_val),axis=1)



import numpy as np
from Dataset import Dataset

dataTrain = Dataset(fft_train,cls_train_oh)

dataVal = Dataset(fft_val,cls_val_oh)

#x,y=    data.next_batch(3)
#FIND INDEX
#np.argwhere(np.all(x[0,:] ==sign_train, axis=1, keepdims=True)==True)

#%%

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
#%%

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
#%%

#place holders
x = tf.placeholder(tf.float32, shape=[None, img_size_flat], name='x')
x_image = tf.reshape(x, [-1, 1, img_size, num_channels])
y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')
y_true_cls = tf.argmax(y_true, dimension=1)
#%%
#Armar CNN

layer_conv1, weights_conv1, bias1 = \
    NN.new_conv_layer(input=x_image,
                   num_input_channels=num_channels,
                   filter_size=filter_size1,
                   num_filters=num_filters1,
                   use_pooling=True,
                   act='relu')    
    
layer_conv2, weights_conv2, bias2 = \
    NN.new_conv_layer(input=layer_conv1,
                   num_input_channels=num_filters1,
                   filter_size=filter_size2,
                   num_filters=num_filters2,
                   use_pooling=True,
                   act='relu')

layer_conv3, weights_conv3, bias3 = \
    NN.new_conv_layer(input=layer_conv2,
                   num_input_channels=num_filters2,
                   filter_size=filter_size3,
                   num_filters=num_filters3,
                   use_pooling=True,
                   act='relu')
    
layer_conv4, weights_conv4, bias4 = \
    NN.new_conv_layer(input=layer_conv3,
                   num_input_channels=num_filters3,
                   filter_size=filter_size4,
                   num_filters=num_filters4,
                   use_pooling=True,
                   act='relu')
       
layer_flat, num_features = NN.flatten_layer(layer_conv4)

layer_fc1 = NN.new_fc_layer(input=layer_flat,
                         num_inputs=num_features,
                         num_outputs=fc_size1,
                         act='relu',
                         drop=1)

layer_fc2 = NN.new_fc_layer(input=layer_fc1,
                         num_inputs=fc_size1,
                         num_outputs=fc_size2,
                         act='relu',
                         drop=1)

layer_fc2 = NN.new_fc_layer(input=layer_fc2,
                         num_inputs=fc_size2,
                         num_outputs=num_classes,
                         act='hola',
                         drop=1)

#capa salida
y_pred = tf.nn.softmax(layer_fc2)
y_pred_cls = tf.argmax(y_pred, dimension=1)

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=layer_fc2,                                                            labels=y_true)
cost = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)

#medidas de desempeño
correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
y_predict=y_pred_cls

#inicio de sesion tf
session = tf.Session(config=tf.ConfigProto(log_device_placement=True))
session.run(tf.global_variables_initializer())


#%%

# Entrenamiento realizado por batches.

def optimize(num_iterations,
             train_batch_size):
    j=0
    ep=540/train_batch_size
    # Tiempo de inicio
    #IT=#samples*#epocas/#batch
    
    start_time = time.time()
    Train_acc = np.zeros(shape=(int(num_iterations/(ep/10)),2), dtype=np.float16)
    #tiempo por epoca
    epoch_times = np.zeros(shape=(int(num_iterations/ep),1), dtype=np.float16)
    start_aux_time=time.time()

    for i in range(num_iterations):

        # Obtener batch de conjunto de entrenamiento.
        x_batch, y_true_batch = dataTrain.next_batch(train_batch_size)

        # Se pone el batch en un diccionario asignándole nombres de las
        # variables placeholder antes definidas.
        feed_dict_train = {x: x_batch,
                           y_true: y_true_batch}

        # Ejecución del optimizador con los batches del diccionario.
        #if red=='sig':
         #   session.run(optimizersig, feed_dict=feed_dict_train)
        #if red=='relu':
        session.run(optimizer, feed_dict=feed_dict_train)
            
        # Se imprime cuando ha pasado una época.
        if i % ep == 0:
            msg = "Época: {0:>6}"
            print(msg.format(j+1))
            j+=1
        
        if i % ep == 0 and i>0:
            epoch_times[int(i/ep)-1]=time.time()-start_aux_time
            start_aux_time+=time.time()-start_aux_time
            
            
        # Se imprime el progreso cada 55 iteraciones.
#        if i % 55 == 0 and red=='sig':
#            acc = session.run(accuracysig, feed_dict=feed_dict_train)
#            msg = "Iterations: {0:>6}, Training Accuracy: {1:>6.1%}"
#            print(msg.format(i, acc))
#            Train_acc[int(i/55),:]=[i,acc]

        if i % (ep/5.4) == 0:# and red=='relu':
            acc = session.run(accuracy, feed_dict=feed_dict_train)
            msg = "Iterations: {0:>6}, Training Accuracy: {1:>6.1%}"
            print(msg.format(i, acc))
            Train_acc[int(i/(ep/10)),:]=[i,acc]
            
    if (num_iterations) % ep == 0 and num_iterations!=ep:
        epoch_times[int(num_iterations/ep)-1]=time.time()-start_aux_time

    # Tiempo de finalización.
    end_time = time.time()

    # Tiempo transcurrido.
    time_dif = end_time - start_time

    mean_time=np.mean(epoch_times)
    
    print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))
    print("Mean time usage per epoch: " + str(timedelta(seconds=int(round(mean_time)))))
    return time_dif, mean_time, Train_acc



# Dividir test set en batches. (Usa batches mas pequeños si la RAM falla).
def print_test_accuracy(test_batch_size):

    # Número de imagenes en test-set.
    num_test = len(dataVal.data)

    # Crea arreglo para guardar clases predichas.
    cls_pred = np.zeros(shape=num_test, dtype=np.int)

    # Calcular clases predichas.
    i = 0
    while i < num_test:
        
        j = min(i + test_batch_size, num_test)
        images = dataVal.data[i:j, :]
        labels = dataVal.labels[i:j, :]
        feed_dict = {x: images,
                     y_true: labels}

        cls_pred[i:j] = session.run(y_pred_cls, feed_dict=feed_dict)
        i = j
    
    # Labels reales.
    cls_true = np.argmax(dataVal.labels, axis=1)

    # Arreglo booleano de clasificaciones correctas.
    correct = (cls_true == cls_pred)
    
    #Número de clasificaciones correctas.
    correct_sum = correct.sum()

    # Accuracy
    acc = float(correct_sum) / num_test
    msg = "Accuracy on Test-Set: {0:.1%} ({1} / {2})"
    print(msg.format(acc, correct_sum, num_test))
    return  acc, cls_true,  cls_pred
        

#%%
#Definir número de iteraciones que desea entrenar a la red
reluTime, reluEpochTime, reluAcc=optimize(num_iterations=5400,
                                          train_batch_size=20) 

#%%
#Test
relu_acc, relu_true, relu_pred=print_test_accuracy(test_batch_size=10)
reluMatrix= confusion_matrix(relu_true, relu_pred)

cls=['0','1','2','3','4','5']
fig=plt.figure()
plot_confusion_matrix(reluMatrix, classes=cls,title='Confusion Matrix')

#%%

w1=weights_conv1.eval(session=session)
w2=weights_conv2.eval(session=session)
w3=weights_conv3.eval(session=session)
w4=weights_conv4.eval(session=session)

b1=bias1.eval(session=session)
b2=bias2.eval(session=session)
b3=bias3.eval(session=session)
b4=bias4.eval(session=session)

np.save('w11.npy', w1)
np.save('w21.npy', w2) 
np.save('w31.npy', w3) 
np.save('w41.npy', w4)  
np.save('b11.npy', b1)
np.save('b21.npy', b2) 
np.save('b31.npy', b3) 
np.save('b41.npy', b4)  
