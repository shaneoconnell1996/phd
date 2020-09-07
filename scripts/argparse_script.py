import argparse
import pandas as pd 
import numpy as np
import tensorflow as tf
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
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,roc_curve,auc,RocCurveDisplay
import sys
import os

"""Script that implements 3D-CNN with specified params - the mnames argument can be any numpy file of random names that can be prefixes for all model related 
performance metrics """
mplbck.use("Agg")
mplbck.style.use(["fast"])
parser = argparse.ArgumentParser()


# add args and just return them

parser.add_argument('-c_1','--conv1',dest='conv_1',type=int,required=True,help="The number of units at the first layer of the CNN")
parser.add_argument('-c_2','--conv2',dest='conv_2',type=int,required=True,help="The number of units at the second layer of the CNN")
parser.add_argument('-c_3','--conv3',dest='conv_3',type=int,required=True,help="The number of units at the third layer of the CNN")
parser.add_argument('-c_4','--conv4',dest='conv_4',type=int,required=True,help="The number of units at the fourth layer of the CNN")
parser.add_argument('-c_5','--conv5',dest='conv_5',type=int,required=True,help="The number of units at the fifth layer of the CNN")

parser.add_argument('-a','--activation',dest='activation',type=str,default='relu',
	help="The activation function to be used throughout the CNN. Default is rectified linear unit")
parser.add_argument('-init','--initializer',dest='initializer',type=str,default='uniform',
	help="The weight initialization scheme for the CNN")
parser.add_argument('-c_d','--spatial_dropout',dest='conv_dropout',default=False,required=True,action='store_true',
        help="Whether or not to use spatial dropout at the convolutional layers")
parser.add_argument('-c_d_r','--spatial_dropout_rate',dest='cnv_drp_rate',type=float,default=0.1,
        help="The dropout rate for spatial dropout layers")
parser.add_argument('-fs','--filter_size',dest='filt_size',type=int,default=3,
        help="The filter size for the network, which conforms to a 3 * input schema, e.g. 3 will result in a pool size of (3,3,3)")
parser.add_argument('-ps','--pool_size',dest='pool_size',type=int,default=3,
        help="The (max) pooling size for the network, which conforms to a 3 * input schema, e.g. 3 will result in a pool size of (3,3,3)")
parser.add_argument('-l2','--l2_penalty',dest='l2_penalty',type=float,default=0.0,
        help="The L2 penalty regularizer for convolutional layers of the network")
parser.add_argument('-fc_1','--fc_layer_1',dest='ff1',type=int,required=True,
        help="The number of units at the first fully connected layer after flattening")
parser.add_argument('-fc_2','--fc_layer_2',dest='ff2',type=int,required=True,
        help="The number of units at the second fully connected layer after flattening")
parser.add_argument('-fc_3','--fc_layer_3',dest='ff3',type=int,required=True,
        help="The number of units at the third fully connected layer after flattening")
parser.add_argument('-fc_4','--fc_layer_4',dest='ff4',type=int,required=True,
        help="The number of units at the fourth fully connected layer after flattening")
parser.add_argument('-bn','--batch_norm',dest='batch_norm',action='store_true',
        help="Whether or not to use batch normalization throughout the fully connected layers")
parser.add_argument('-bn_m','--batch_norm_momentum',dest='mom',type=float,default=0.9,
        help="The momentum hyperparameter for batch normalisation layers")
parser.add_argument('-fc_d','--fc_dropout',dest='ffdrop',required=True,action='store_true',
        help="Whether or not to add dropout at the fully connected layers")
parser.add_argument('-fc_d_r','--fc_dropout_rate',dest='drp_rate',type=float,default=0.2,
        help="The rate at which to apply dropout in the fully connected layers")
parser.add_argument('-p','--padding',dest='padding',type=str,required=True,default='valid',
        help="Padding schema to use, where options are 'valid' and 'same' ")

# next set of args are fed to compiler function 
parser.add_argument('-opt','--optimizer',dest='optimizer',type=str,required=True,
        help="Optimizer to use: choice of rmsprop, adam, or stochastic gradient descent")
parser.add_argument('-eps','--epsilon',dest='epsilon',type=float,default=1e-7,
        help="Epsilon constant used during update steps (for ADAM, the default may not be optimal, and may suit values of 0.1 or 1.0 instead)")
parser.add_argument('-m','--momentum',dest='momentum',type=float,default=0.95,
        help="Momentum term to use for either RMSprop or SGD (generally values closer to 1 are preferred)")
parser.add_argument('-nm','--nesterov',dest='nesterov',action='store_true',default=True,
        help="Whether or not to use nesterov momentum with SGD (slightly updates usual momentum term to reflect intermediate parameter value)")
parser.add_argument('-lr','--learning-rate',dest='lr',type=float,default=0.01,
	help="The learning rate of the CNN")
# parse arguments
parser.add_argument('-data','--datapath',dest='datapath',type=str,required=True,
        help="The data to use for training")
parser.add_argument('-y','--ypath',dest='ypath',type=str,required=True,
        help="The labels to use for training")
parser.add_argument('-seed','--seed',dest='seed',type=int,default=13,
        help="Random seed to use for training")
parser.add_argument('-bs','--batch_size',dest='batch_size',type=int,default=5,
        help="Batch size to use during training")
parser.add_argument('-ts','--test_size',dest='test_size',type=float,default=0.3,
        help="Train test split size to use")
parser.add_argument('-vs','--val_size',dest='val_size',type=float,default=0.3,
        help="Train test split size to use")
parser.add_argument('-v','--verbosity',dest='verbosity',type=int,default=0,
        help="Verbosity during training")
parser.add_argument('-ep','--epochs',dest='epochs',type=int,default=10,
        help="How long to train")


args=parser.parse_args()
model_names=np.load('model_names.npy') # numpy file of string model names that will serve as identifier 
# create summary and model directory 

def create_summary(args):
    df = pd.DataFrame.from_dict(vars(args),orient='index')
    df['fields']=df.index
    df.columns = ['values','fields']
    return(df)
import datetime
date=datetime.datetime.now()
dt=date.strftime("%d_%m_%y-%H_%M")


filelist=os.listdir('models/')
possible_names=[]
for i in range(len(model_names)):
    if model_names[i] not in filelist:
        possible_names.append(model_names[i])
if len(possible_names) == 0:
        sys.exit("You have run out of unique model names. Things must be pretty bad!")
mname=list(np.random.choice(possible_names,1))[0]
os.mkdir(f'models/{mname}')
sum_df=create_summary(args)
sum_df.to_csv(f'models/{mname}/{mname}_{dt}_params.log',index=False)

print('Arguments parsed.')

### load data ### 
data=np.load(args.datapath)
y=np.load(args.ypath)
print(f'file from {args.datapath} loaded, with shape {data.shape}')
### check shape ###
if len(data.shape) == 2:
    data=data.reshape(data.shape[0],125,145,125)
if len(y) != data.shape[0]:
	sys.exit("Inconsistent number of samples in label and data file")
### reshape data ###
s = data.shape
s = s+(1,)


data=data.reshape(s)


### train and test split ###


x_train,x_test,y_train,y_test=train_test_split(data,y,random_state=args.seed,
                                               test_size=args.test_size)


### build model ### 

def create_model(conv_1,conv_2,conv_3,conv_4,conv_5,activation,initializer,conv_dropout,cnv_drp_rate,
                filt_size,pool_size,l2_penalty,ff1,ff2,ff3,ff4,batch_norm,mom,ffdrop,drp_rate,
                padding):
    
    """creates convolutional neural network with tensorflow keras layers in 3 distinct blocks"""
    model=Sequential()
    
    model.add(Conv3D(conv_1,filt_size,input_shape=(125,145,125,1),activation=activation,
                    kernel_initializer=initializer,kernel_regularizer=l2(l2_penalty),
                    name='conv_1',padding=padding))
    model.add(Conv3D(conv_2,filt_size,activation=activation,
                    kernel_initializer=initializer,kernel_regularizer=l2(l2_penalty),
                    name='conv_2',padding=padding))
    if conv_dropout == True:
        model.add(SpatialDropout3D(cnv_drp_rate,name='spatial_dropout_1'))
    model.add(MaxPooling3D(pool_size,name='max_pool_1'))
    
    
    model.add(Conv3D(conv_3,filt_size,activation=activation,
                    kernel_initializer=initializer,kernel_regularizer=l2(l2_penalty),
                    name='conv_3',padding=padding))
    model.add(Conv3D(conv_4,filt_size,activation=activation,
                    kernel_initializer=initializer,kernel_regularizer=l2(l2_penalty),
                    name='conv_4',padding=padding))
    if conv_dropout == True:
        model.add(SpatialDropout3D(cnv_drp_rate,name='spatial_dropout_2'))
    model.add(MaxPooling3D(pool_size,name='max_pool_2'))
    
    
    model.add(Conv3D(conv_5,filt_size,activation=activation,
                    kernel_initializer=initializer,kernel_regularizer=l2(l2_penalty),
                    name='conv_5',padding=padding))
    if conv_dropout == True:
        model.add(SpatialDropout3D(cnv_drp_rate,name='spatial_dropout_3'))
    model.add(MaxPooling3D(pool_size,name='max_pool_3'))
    
    model.add(Flatten(name='flatten'))
    
        
    model.add(Dense(ff1,activation=activation,
                    kernel_initializer=initializer,kernel_regularizer=l2(l2_penalty),
                   name='dense_1'))
    
    if batch_norm==True:
        model.add(BatchNormalization(momentum=mom,name='batch_norm_1'))
    if ffdrop==True:
        model.add(Dropout(drp_rate,name='dense_dropout_1'))
    model.add(Dense(ff2,activation=activation,
                    kernel_initializer=initializer,kernel_regularizer=l2(l2_penalty),
                   name='dense_2'))
    if batch_norm==True:
        model.add(BatchNormalization(momentum=mom,name='batch_norm_2'))
    if ffdrop==True:
        model.add(Dropout(drp_rate,name='dense_dropout_2'))
    model.add(Dense(ff3,activation=activation,
                    kernel_initializer=initializer,kernel_regularizer=l2(l2_penalty),
                   name='dense_3'))
    if batch_norm==True:
        model.add(BatchNormalization(momentum=mom,name='batch_norm_3'))
    if ffdrop==True:
        model.add(Dropout(drp_rate,name='dense_dropout_3'))
    model.add(Dense(ff4,activation=activation,
                    kernel_initializer=initializer,kernel_regularizer=l2(l2_penalty),
                   name='dense_4'))
    
    model.add(Dense(1,activation='sigmoid',name='prediction'))
    
    return(model)

def compile_nn(model,optimizer,lr,epsilon,momentum,nesterov):
    """compiles given model with specified params"""
    if optimizer=='sgd':
        if nesterov == True:
            model.compile(optimizer=SGD(learning_rate=lr,momentum=momentum,nesterov=True),metrics=['accuracy'],
                         loss='binary_crossentropy')
        else:
            model.compile(optimizer=SGD(learning_rate=lr,momentum=momentum,nesterov=False),metrics=['accuracy'],
                         loss='binary_crossentropy')
    if optimizer=='adam':
        model.compile(optimizer=Adam(learning_rate=lr,epsilon=epsilon),metrics=['accuracy'],
                         loss='binary_crossentropy')
    if optimizer=='rmsprop':
        model.compile(optimizer=RMSprop(learning_rate=lr,momentum=momentum,epsilon=epsilon),metrics=['accuracy'],
                         loss='binary_crossentropy')
        

##### create training function that takes epochs to run as another parameter 


def train_model(epochs,verbose,valsize,nn,x,y,bs):
    # Trains the given nn according to input params and returns the progress object
    progress=nn.fit(x,y,epochs=epochs,verbose=verbose,validation_split=valsize,batch_size=bs,shuffle=True)
    return(progress)

def plot_and_save_prog(progress,mname):
    # plots and saves model
    fig,(ax1,ax2) = plt.subplots(ncols=2,nrows=1,figsize=(15,8))
    ax1.plot(progress.history['accuracy'],label='Training')
    ax1.plot(progress.history['val_accuracy'],label='Validation')
    ax1.set_title('Accuracy vs. Epochs')
    ax1.set_ylabel('Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.legend()

    ax2.plot(progress.history['loss'],label='Training')
    ax2.plot(progress.history['val_loss'],label='Validation')
    ax2.set_title('Loss vs. Epochs')
    ax2.set_ylabel('Loss')
    ax2.set_xlabel('Epoch')
    ax2.legend()
    
    fig.savefig(f'models/{mname}/learning_curve.png')

def eval_performance(model):
    pred_prob=[]
    for i in range(x_test.shape[0]):
      pred_prob.append(float(model.predict(np.array([x_test[i]]))))
    y_pred=[]
    for i in pred_prob:
      if i > 0.5:
        y_pred.append(1)
      else:
        y_pred.append(0)
    return(accuracy_score(y_test,y_pred),pred_prob)

def make_roc_curve(y_prob,acc,mname):
    fpr,tpr,_=roc_curve(y_test,y_prob)
    auc_cnn=auc(fpr,tpr)
    fig,ax=plt.subplots(figsize=(15,8))
    ax.plot(fpr,tpr,label=f'{mname} acc = {acc}')
    ax.plot([0,1],c='red',label='Baseline')
    ax.legend()
    ax.text(x=0.8,y=0.05,s=f'AUC={auc_cnn}')
    ax.set_xlabel('False positive rate')
    ax.set_ylabel('True positive rate')
    ax.set_title('ROC curve with AUC')

    fig.savefig(f'models/{mname}/{mname}_performance.png')
### string all functions together 
filt_size=(args.filt_size,args.filt_size,args.filt_size)
pool_size=(args.pool_size,args.pool_size,args.pool_size)
nn = create_model(conv_1=args.conv_1,conv_2=args.conv_2,conv_3=args.conv_3,conv_4=args.conv_4,conv_5=args.conv_5,
                  activation=args.activation,initializer=args.initializer,conv_dropout=args.conv_dropout,cnv_drp_rate=args.cnv_drp_rate,
                 filt_size=filt_size,pool_size=pool_size,l2_penalty=args.l2_penalty,ff1=args.ff1,ff2=args.ff2,ff3=args.ff3,ff4=args.ff4,
                 batch_norm=args.batch_norm,mom=args.mom,ffdrop=args.ffdrop,drp_rate=args.drp_rate,padding=args.padding)


# compile and train model 

compile_nn(model=nn,optimizer=args.optimizer,lr=args.lr,epsilon=args.epsilon,momentum=args.momentum,nesterov=args.nesterov)

progress = train_model(epochs=args.epochs,verbose=args.verbosity,valsize=args.val_size,nn=nn,x=x_train,y=y_train,bs=args.batch_size)

plot_and_save_prog(progress=progress,mname=mname)

# save trained model 

nn.save(f'models/{mname}/{mname}_trained.h5')

# evaluate the model


acc,y_pred=eval_performance(model=nn)

make_roc_curve(y_prob=y_pred,title=f'CNN with BN: acc = {acc}',mname=mname)
