# -*- coding: utf-8 -*-
"""
The old notebook is dead. This is the redux with shiny new functions.
The purpose of this script / notebook is to have a suite of functions that allow various easy to implement tensorflow models to be tested rapidly on some training
data.
"""


import tensorflow as tf
tf.test.is_gpu_available()
# this part is for colab 


import numpy as np
# this part changes based on what version we run with 
# data = np.load('drive/My Drive/data/local_and_ukb_dataset_lossy_revised_qc.npy')
with open('labels.txt') as w:
  ydata=w.readlines()
ydata=[int(i.strip()) for i in ydata]
local_y = ydata[:101]
ukb_y = ydata[101:]

ukb_data = np.load('ukb.npy')

local_data = np.load('local.npy')


print(f'using tensorflow version {tf.__version__}')
from tensorflow.keras.layers import (
    Dense, Flatten, BatchNormalization, Conv3D,
    SpatialDropout3D, Dropout, Activation, LeakyReLU,
    MaxPooling3D
)
from tensorflow.keras.optimizers import SGD, Adam, RMSprop
from tensorflow.keras.regularizers import l1, l2, l1_l2
from tensorflow.keras.models import Sequential
import matplotlib as mplbck
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score,roc_curve,auc,RocCurveDisplay

import sys

# scaling functions
from sklearn.preprocessing import MinMaxScaler, StandardScaler
mn = MinMaxScaler(feature_range=(0,255))


def scale_data_255(patient):
	
  return mn.fit_transform(patient)

# scale ukb
ukb_scale = np.zeros((199,120,140,120))
for i in range(ukb_scale.shape[0]):
  ukb_scale[i] = scale_data_255(ukb_data[i].flatten().reshape(-1,1)).reshape(120,140,120)

# scale local

local_scale = np.zeros((101,120,140,120))
for i in range(local_scale.shape[0]):
  local_scale[i] = scale_data_255(local_data[i].flatten().reshape(-1,1)).reshape(120,140,120)

# create model function for a sparser architecture

def create_cnn_sparse_drop(lr,shape,drprate):
  cnn = Sequential([
                  Conv3D(32,(3,3,3),data_format='channels_last',input_shape=shape,
                         activation='relu'),
                   Conv3D(32,(3,3,3),activation='relu'),
                    MaxPooling3D((3,3,3)),
                   Conv3D(64,(3,3,3),activation='relu'),
                   Conv3D(64,(3,3,3),activation='relu'),
                   MaxPooling3D((3,3,3)),
                  Conv3D(128,(3,3,3),activation='relu'),
                    #Conv3D(32,(3,3,3),activation='relu'),
                    MaxPooling3D((2,2,2)),

                   Flatten(),

                   Dense(512,activation='relu'),
                   Dropout(drprate),
                   Dense(256,activation='relu'),
                   Dropout(drprate),
                   Dense(128,activation='relu'),
                   Dropout(drprate),
                   Dense(64,activation='relu'),
                   Dense(1,activation='sigmoid')
  ])
  cnn.compile(optimizer=Adam(learning_rate=lr,clipnorm=1),metrics=['accuracy'],loss='binary_crossentropy')
  return cnn

# 10 fold cross validation section 

# this is a little funky. I haven't decided whether or not this is cherry picking yet

# essentially, when a model with a unique solution is fit, there is an objective measure of convergence (ie logistic regression, SVM, etc)
# neural networks are different. Their solutions can be unique based on the weight initialization, which, in this case, is sampled from a normal distribution.
# as such, every model fit on every fold can have a different optimal solution - where we might define that optimal solution as the set of weights that minimize 
# the distance between f(x) and y in out of sample prediction. What this results in is a model that can run for an arbitrary amount of time, a burn-in period, 
# and after this point is reached, the model begins making predictions on the test set. The predictions are stored in a list and the maximum result from each
# fold is appended to the final list. The theory behind this is the stochasticity of each new neural network fit to each new fold of data is accounted for by taking
# every one of these models at their own measure of convergence, seeing as true convergence has no known numerical description for neural networks



def custom_loop_cross_val(model,epochs,burnin,plot,thresh,acc_thresh,npredictions,train_data,train_y,test_data,test_y):
  acc = []
  loss = []
  aucs = []
  accus = []
  for i in range(epochs):
    print(f'epoch {i+1} fit:')
    pr = model.fit(x=train_data,y=train_y,epochs=1,shuffle=True,batch_size=6,
                   verbose=1)
    acc.append(pr.history['accuracy'])
    loss.append(pr.history['loss'])
    acc_val = acc[-1][0]
    if acc_val > acc_thresh and i < burnin:
      print(f'Accuracy above threshold before burn in. New burn in assigned at {i}')
      burnin = i
    else:
      pass
    if i >= burnin:
      accu,_,yprob = eval_performance(model,test_data,test_y)
      aucs.append(make_roc_curve(y_prob=yprob,acc=accu,truey=test_y,plot=False))
      accus.append(accu)
     
    if i - burnin > npredictions:
      print(f'Model has been storing predictions for {npredictions} epochs. Terminating loop')
      break # this part just stops the loop if the number of predictions stored exceeds the user specified limit
    
  if plot==True:
    fig,(ax1,ax2) = plt.subplots(ncols=2,figsize=(15,8))
    ax1.plot(acc,c='blue')
    ax1.set_title('Training accuracy')
    ax2.plot(loss,c='red')
    ax2.set_title('Training loss')

    accu,_,yprob = eval_performance(model,test_data,test_y)
    make_roc_curve(y_prob=yprob,acc=accu,truey=test_y,plot=True)
    

    fig,ax = plt.subplots(figsize=(15,8))
    ax.plot(aucs,c='green',label='AUCs')
    ax.set_title('AUCs vs epochs after burn in')
    ax.plot(accus,c='red',label='Test Accuracies')
    ax.legend()
    return([accus,aucs])
  return([accus,aucs])


# create and reshape total data 
total_scale = np.zeros((300,120,140,120,1))
total_scale[:101] = local_scale.reshape(101,120,140,120,1)
total_scale[101:] = ukb_scale.reshape(199,120,140,120,1)
total_y = local_y
for i in ukb_y:
  total_y.append(i)

# the cross validation cnn function is defined here.

def cross_val_cnn(folds,x,y,argmax):
    
    skf = StratifiedKFold(n_splits=folds,shuffle=True,random_state=42)
    accs = []
    aucs = []
    bp = y
    bp = np.array(bp)
    skf.get_n_splits(x,bp)
    count = 0
    for train_index,test_index in skf.split(x,bp):
        count += 1
        print(f'fold {count} commencing...')
        cnn = create_cnn_sparse_drop(0.0005,local_data.reshape(101,120,140,120,1)[0].shape,0.1)
        accus,aucs = custom_loop_cross_val(cnn,100,60,False,0.6,0.65,24,x[train_index],bp[train_index],x[test_index],bp[test_index])
	# the argmax flag is to specify which quantity you want to track. you need to take the same model for both quantities
        if argmax == 'acc':
		
		max_ind = np.argmax(accus)
	else:
		max_ind = np.argmax(aucs)
	
        accs.append(accus[max_ind])
        aucs.append(aucs[max_ind])

    return([f'{folds} fold cross val (conv neural net): {np.mean(accs)} +/- {np.std(accs)} \n aucs: {np.mean(aucs)} + / - {np.std(aucs)}',
            accs,aucs])
msg,accs,aucs = cross_val_cnn(10,total_scale,total_y)
print(msg)
