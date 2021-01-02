import numpy as np;
import math
import random
from scipy.stats import special_ortho_group
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential,Model,load_model
from keras.layers import *
from keras.engine.topology import Layer

#----------------------------------------------------------------------------------
# Training Functions
# Trains the model using Shape Preserving Data Augmentation specified by some hyperparameters 
#----------------------------------------------------------------------------------




def trainModelwSPDA(model, length, dim, closed, otrData, otrLabels, e, trainsize= None, reparamn=10, verbose = True):
    """
    Parameters
    ----------
    model : keras model
        keras model to be trained, can be custom or created with with defineModel
    length : int
        length of the curves for training ()
    dim : int
        dimension of the curves
    closed : bool
        whether or not curves are closed
    otrData : numpy array
        np array with pairs of curves  for training. Shape should be (n,length,2*dim) where otrData(i,:,0:dim) and otrData(i,:,dim:2*dim) make up ith pair of curves 
    otrLabels : numpy array
        the precise distances computed for each case in otrData
    trainsize : int
        batchsize for each epoch of training
    reparamn : int
        number of curves to generate from each shape class during shape preserving data augmentation
    verbose : bool
        whether or not to display training progress
        
    Returns        
    ----------
    model: keras model
        trained model

    """
    
    if trainsize==None:
        trainsize=otrData.shape[0];
    for j in range(0,e):
        perm = np.random.permutation(otrData.shape[0]);
        otrData=otrData[perm,:]
        otrLabels=otrLabels[perm,:]
        trData=np.zeros((reparamn*trainsize, length,2*dim))
        trLabels=np.zeros((reparamn*trainsize,1))
        for i in range(0,trainsize):
            f1=otrData[i,0:length,0:dim]
            f2=otrData[i,0:length,dim:2*dim]
            for k in range(0,reparamn):
                trData[i*reparamn+k,0:length,0:dim]=randomCurveFromShapeClass(f1,length,dim,closed)
                trData[i*reparamn+k,0:length,dim:2*dim]=randomCurveFromShapeClass(f2,length,dim,closed)
                trLabels[i*reparamn+k]=otrLabels[i]
            if verbose:
                print("Epoc: {}    Percent: {}".format(j,(i+1)*100/trainsize),end="\r")

        trData1=trData[:,:,0:dim]
        trData2=trData[:,:,dim:2*dim]

        model.fit( x=[trData1,trData2], y=trLabels, batch_size =1000, epochs = 1, shuffle=True, use_multiprocessing=True, verbose = 0)
     
    return model

def trainAndValidateModelwSPDA(model, length, dim, closed, otrData, otrLabels, tData, tLabels, e, trainsize, reparamn=10):
    """
    Parameters
    ----------
    model : keras model
        keras model to be trained, can be custom or created with with defineModel
    length : int
        length of the curves for training ()
    dim : int
        dimension of the curves
    closed : bool
        whether or not curves are closed
    otrData : numpy array
        np array with pairs of curves for training.Shape should be (n,length,2*dim) where otrData(i,:,0:dim) and otrData(i,:,dim:2*dim) make up ith pair of curves 
    otrLabels : numpy array
        the precise distances computed for each case in otrData used as labels for training
    tData : numpy array
        np array with pairs of curves for testing/validation. Shape should be (n,length,2*dim) where otrData(i,:,0:dim) and otrData(i,:,dim:2*dim) make up ith pair of curves 
    tLabels : numpy array
        the precise distances computed for each case in tData for testing/validation
    trainsize : int
        batchsize for each epoch of training
    reparamn : int
        number of curves to generate from each shape class during shape preserving data augmentation
    verbose : bool
        whether or not to display training progress
        
    Returns        
    ----------
    model: keras model
        trained model
    trainMSE: numpy array
        mean squared error for otrData at each epoch
    testMSE: numpy array
        mean squared error for tData at each epoch
        
    """
    
    tData1=tData[:,:,0:dim]
    tData2=tData[:,:,dim:2*dim]

    testMSE=np.zeros(e)
    trainMSE=np.zeros(e)
    for j in range(0,e):
        perm = np.random.permutation(otrData.shape[0]);
        otrData=otrData[perm,:]
        otrLabels=otrLabels[perm,:]
        trData=np.zeros((reparamn*trainsize, length,2*dim))
        trLabels=np.zeros((reparamn*trainsize,1))
        for i in range(0,trainsize):
            f1=otrData[i,0:length,0:dim]
            f2=otrData[i,0:length,dim:2*dim]
            for k in range(0,reparamn):
                trData[i*reparamn+k,0:length,0:dim]=randomCurveFromShapeClass(f1,length,dim,closed)
                trData[i*reparamn+k,0:length,dim:2*dim]=randomCurveFromShapeClass(f2,length,dim,closed)
                trLabels[i*reparamn+k]=otrLabels[i]
            print("Epoc: {}    Percent: {}".format(j,(i+1)*100/trainsize),end="\r")

        trData1=trData[:,:,0:dim]
        trData2=trData[:,:,dim:2*dim]

        model.fit( x=[trData1,trData2], y=trLabels, batch_size =1000, epochs = 1, shuffle=True, use_multiprocessing=True, verbose = 0)
        test=model.predict(x=[tData1,tData2], verbose = 0)
        train=model.predict(x=[otrData[:,:,0:dim],otrData[:,:,dim:2*dim]], verbose = 0) 
        testMSE[j]=np.mean(np.square(test-tLabels))
        trainMSE[j]=np.mean(np.square(train-otrLabels))
        print("Train: {}    Test: {}".format(trainMSE[j],testMSE[j]))
    return model, trainMSE, testMSE
#----------------------------------------------------------------------------------
# Network Definition Functions
# Defines the network as described in [associated publication]
# Returns keras model class
#----------------------------------------------------------------------------------

def defineModel(length, dim, closed, verbose=True):
    """
    Parameters
    ----------
    length : int
        length of the curves the model
    dim : int
        dimension of the curves
    closed : bool
        whether or not curves are closed
    verbose : bool
        whether or not to display model summary
        
    Returns        
    ----------
    model: keras model
        new model with layers as defined in [title] 
        
    """
    if closed:
        return defineModelClosed(length, dim, verbose)
    else:
        return defineModelOpen(length, dim, verbose)
    
    
def defineModelOpen(length, dim, verbose):
    #-------------------------------------------------------------#
    # Make keras model with widths based on length/dim 
    #-------------------------------------------------------------#
    act='relu'
    opt= 'adam'
    error ='mean_squared_error'
    kernalsize=5;


    input1= Input((length,dim))
    input2= Input((length,dim))

    convlayers = Sequential();

    convlayers.add(CyclicPad())
    convlayers.add(Conv1D(filters=2*dim, kernel_size=kernalsize, padding= 'same'))
    convlayers.add(BatchNormalization(axis=1))
    convlayers.add(Activation(act))


    convlayers.add(CyclicPad())
    convlayers.add(Conv1D(filters=4*dim, kernel_size=kernalsize, padding= 'same'))
    convlayers.add(BatchNormalization(axis=1))
    convlayers.add(Activation(act))
    convlayers.add(MaxPooling1D(pool_size=2, padding= 'same'))

    convlayers.add(CyclicPad())
    convlayers.add(Conv1D(filters=8, kernel_size=kernalsize, padding= 'same'))
    convlayers.add(BatchNormalization(axis=1))
    convlayers.add(Activation(act))
    convlayers.add(MaxPooling1D(pool_size=2, padding= 'same'))

    convlayers.add(CyclicPad())
    convlayers.add(Conv1D(filters=16*dim, kernel_size=kernalsize, padding= 'same'))
    convlayers.add(BatchNormalization(axis=1))
    convlayers.add(Activation(act))
    convlayers.add(MaxPooling1D(pool_size=2, padding= 'same'))

    convlayers.add(CyclicPad())
    convlayers.add(Conv1D(filters=32*dim, kernel_size=kernalsize, padding='same'))
    convlayers.add(BatchNormalization(axis=1))
    convlayers.add(Activation(act))
    convlayers.add(MaxPooling1D(pool_size=2, padding= 'same'))

    convlayers.add(CyclicPad())
    convlayers.add(Conv1D(filters=64*dim, kernel_size=kernalsize, padding='same'))
    convlayers.add(BatchNormalization(axis=1))
    convlayers.add(Activation(act))
    convlayers.add(MaxPooling1D(pool_size=2, padding= 'same'))

    convlayers.add(CyclicPad())
    convlayers.add(Conv1D(filters=128*dim, kernel_size=kernalsize, padding='same'))
    convlayers.add(BatchNormalization(axis=1))
    convlayers.add(Activation(act))
    convlayers.add(MaxPooling1D(pool_size=2, padding= 'same'))

    convlayers.add(CyclicPad())
    convlayers.add(Conv1D(filters=256*dim, kernel_size=kernalsize, padding='same'))
    convlayers.add(BatchNormalization(axis=1))
    convlayers.add(Activation(act))
    convlayers.add(MaxPooling1D(pool_size=2, padding= 'same'))

    out1=convlayers(input1);
    out2=convlayers(input2);

    out=Concatenate(axis=2)([out1,out2])
    out=Flatten()(out)
    out=Dense(256*dim, activation=act)(out)
    out=Dense(64*dim, activation=act)(out)
    out=Dense(16*dim, activation=act)(out)
    out=Dense(4*dim, activation=act)(out)
    out=Dense(1, activation='linear')(out)
    model = Model([input1,input2],out)
    model.compile(optimizer=opt,loss = error)
    
    if verbose:
        model.summary()
        
    return model


def defineModelClosed(length, dim, verbose):
    #-------------------------------------------------------------#
    # Make keras model with widths based on length/dim 
    #-------------------------------------------------------------#
    act='relu'
    opt= 'adam'
    error ='mean_squared_error'
    kernalsize=5;


    input1= Input((length,dim))
    input2= Input((length,dim))

    convlayers = Sequential();

    convlayers.add(Conv1D(filters=2*dim, kernel_size=kernalsize, padding= 'same'))
    convlayers.add(BatchNormalization(axis=1))
    convlayers.add(Activation(act))


    convlayers.add(Conv1D(filters=4*dim, kernel_size=kernalsize, padding= 'same'))
    convlayers.add(BatchNormalization(axis=1))
    convlayers.add(Activation(act))
    convlayers.add(MaxPooling1D(pool_size=2, padding= 'same'))

    convlayers.add(Conv1D(filters=8, kernel_size=kernalsize, padding= 'same'))
    convlayers.add(BatchNormalization(axis=1))
    convlayers.add(Activation(act))
    convlayers.add(MaxPooling1D(pool_size=2, padding= 'same'))

    convlayers.add(Conv1D(filters=16*dim, kernel_size=kernalsize, padding= 'same'))
    convlayers.add(BatchNormalization(axis=1))
    convlayers.add(Activation(act))
    convlayers.add(MaxPooling1D(pool_size=2, padding= 'same'))

    convlayers.add(Conv1D(filters=32*dim, kernel_size=kernalsize, padding='same'))
    convlayers.add(BatchNormalization(axis=1))
    convlayers.add(Activation(act))
    convlayers.add(MaxPooling1D(pool_size=2, padding= 'same'))

    convlayers.add(Conv1D(filters=64*dim, kernel_size=kernalsize, padding='same'))
    convlayers.add(BatchNormalization(axis=1))
    convlayers.add(Activation(act))
    convlayers.add(MaxPooling1D(pool_size=2, padding= 'same'))

    convlayers.add(Conv1D(filters=128*dim, kernel_size=kernalsize, padding='same'))
    convlayers.add(BatchNormalization(axis=1))
    convlayers.add(Activation(act))
    convlayers.add(MaxPooling1D(pool_size=2, padding= 'same'))

    convlayers.add(Conv1D(filters=256*dim, kernel_size=kernalsize, padding='same'))
    convlayers.add(BatchNormalization(axis=1))
    convlayers.add(Activation(act))
    convlayers.add(MaxPooling1D(pool_size=2, padding= 'same'))

    out1=convlayers(input1);
    out2=convlayers(input2);

    out=Concatenate(axis=2)([out1,out2])
    out=Flatten()(out)
    out=Dense(256*dim, activation=act)(out)
    out=Dense(64*dim, activation=act)(out)
    out=Dense(16*dim, activation=act)(out)
    out=Dense(4*dim, activation=act)(out)
    out=Dense(1, activation='linear')(out)
    model = Model([input1,input2],out)
    model.compile(optimizer=opt,loss = error)
    
    if verbose:
        model.summary()
        
    return model


#----------------------------------------------------------------------------------
# Shape Preserving Data Augmentation
#----------------------------------------------------------------------------------

def getShapeClass(f1,d):
    """
    Parameters
    ----------
    f1 : numpy array
        curve to get the shape class of
    d : int
        dimension of the curves
        
    Returns        
    ----------
    f : numpy array
        trimmed array representing the shape class of f1
    """
    length=f1.shape[0];
    f1=f1.reshape(length,d)
    eps=.0001
    f=[]
    f.append(f1[0,:])
    df=np.diff(f1,axis=0);
    e1=df[0,:]
    n=np.linalg.norm(e1)
    if n>0:
        ne1=e1/n
    else:
        ne1=e1
        
    for i in range(1,df.shape[0]):
        e2=df[i,:]
        n=np.linalg.norm(e2)
        if n>0:
            ne2=e2/n
        else:
            ne2=e2
            
            
        if np.abs(np.dot(ne1,ne2)-1)>eps:
            f.append(f1[i,:]);
        ne1=ne2
    f.append(f1[len(f1)-1,:])
    return np.array(f)

def randomCurveFromShapeClass(f1,newlength,d,closed):
    """
    Parameters
    ----------
    f1 : numpy array
        curve to get the shape class of
    newlength : int
        desired length for the returned curve
    d : int
        dimension of the curves
    closed : bool
        whether or not f1 is closed
        
    Returns        
    ----------
    f : numpy array
        curve discretized at newlength points from the same shape class as f1
    """
    f1=getShapeClass(f1,d)
    length=f1.shape[0];
    f1=f1.reshape(length,d)
    #if closed apply a random change of starting point
    if closed:
        s1=random.randint(0,length-1)
        f1=np.concatenate((f1[s1:length],f1[0:s1]),0) 
        f1=f1-f1[s1]
    
    #if a curve apply a random rotation
    if d>1:
        R1 = special_ortho_group.rvs(d)        
        f1 = np.dot(f1,R1)
        
        
    t=np.linspace(0,1,length).reshape(length,1)
    newpoints=newlength-length;
    nt=np.sort(np.concatenate((t,np.random.rand(newpoints,1)),0),0)
    f=np.zeros((newlength,d))
    for k in range(0,d):
        f[:,k]=np.interp(np.squeeze(nt),np.squeeze(t),np.squeeze(f1[:,k]))
    return f.reshape(newlength,d)
    
#----------------------------------------------------------------------------------
# Keras Layer Definition for cyclic padding 
# Used for closed curves 
#----------------------------------------------------------------------------------

class CyclicPad(Layer):
    def __init__(self, **kwargs):
        self.kernalsize = 5
        super(CyclicPad, self).__init__(**kwargs)
        

    def call(self, inputs):
        length=inputs.shape[1]
        n=math.floor(self.kernalsize/2)
        a=inputs[:,length-n:length,:]
        b=inputs[:,0:n,:]
        
        return tf.concat([a, inputs, b], 1)
    
    def compute_output_shape(self, input_shape):
        return((input_shape[0],input_shape[1]+2*math.floor(self.kernalsize/2),input_shape[2]))

