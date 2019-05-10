#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import librosa, librosa.display
import sklearn
import keras
from keras.models import Model, Sequential, Input
from keras.layers import Conv2D
import keras.backend as K


# In[2]:


from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())


# # Load in Data and peek at dataframe

# 1. 25863 mp3 mono audio files, each of 30 seconds long
# 2. 188 labels associated with each song

# In[3]:


df = pd.read_csv("../annotations_final.csv",sep="\t")
df2=pd.read_csv("../clip_info_final.csv",sep="\t")


# In[4]:


print(df.shape,df2.shape)


# In[5]:


df.head(10)


# In[6]:


df2.head(6)


# # Ground truth Labels

# In[7]:


y=df.values[:,1:-1]
#this y contains clip id as its first column
print(y,y.shape)


# # Feature extraction

# 1. use feature extraction methods from VGGish, preprocess.py
# 2. perform transfer learning on VGGish network

# In[8]:


#split dataset into train and test set
n=df.shape[0]
n_test=n//4
test_idx=np.arange(0,n_test)
train_idx=np.arange(n_test,n)
batch_size = 32


# In[23]:


#this takes input as batch size and output the input batch and ground truth labels
def batchmatrix(batch_size,df,idx):
    #idx=np.arange(0,df.shape[0])
    np.random.shuffle(idx)
    idx_shuffled=idx[:batch_size]
    batchmatrix=np.zeros((batch_size,96,64))
    batchy=np.zeros((batch_size,188))
    for m,i in enumerate(idx_shuffled):
        x,fs=librosa.load(df["mp3_path"][i])
        X=preprocess_sound.preprocess_sound(x, fs)
        X_slice=X[random.randint(0,X.shape[0]-1),:,:]
        batchmatrix[m,:,:]=X_slice
        batchy[m,:]=df.values[i,1:-1]
    return batchmatrix,batchy


# # Transfer Learning from VGGish net

# In[24]:


import vggish_input
import vggish_params
import vggish_postprocess
import vggish_slim
import vggish
import preprocess_sound
import random


# In[25]:


from __future__ import print_function
# from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.models import load_model #save and load models
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization


# In[26]:


model=vggish.VGGish(load_weights=True, weights='audioset',
           input_tensor=None, input_shape=None,
           out_dim=None, include_top=False, pooling='avg')
new_model=Sequential()
for layer in model.layers:
    new_model.add(layer)
    layer.trainable=False
    

#new_model.add(Flatten())
new_model.add(BatchNormalization())
new_model.add(Dropout(0.5))
new_model.add(Dense(512, activation = 'relu'))
new_model.add(BatchNormalization())
new_model.add(Dropout(0.5))
new_model.add(Dense(188, activation='softmax'))


# In[27]:


new_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[28]:


new_model.summary()


# # Batch processing

# 1. using pescador to generate streamers

# In[29]:


def batchmatrix(batch_size,df,idx):
    #idx=np.arange(0,df.shape[0])
    np.random.shuffle(idx)
    idx_shuffled=idx[:batch_size]
    batchmatrix=np.zeros((batch_size,96,64,1))
    batchy=np.zeros((batch_size,188))
    for m,i in enumerate(idx_shuffled):
        x,fs=librosa.load(df["mp3_path"][i])
        X=preprocess_sound.preprocess_sound(x, fs)
        X_slice=X[random.randint(0,X.shape[0]-1),:,:]
        X_slice.reshape(-1,96,64,1)
        batchmatrix[:,m,:,:]=X_slice
        batchy[m,:]=df.values[i,1:-1]
    return batchmatrix,batchy


# In[30]:


def feature_sampler(df,idx):
    #idx=np.arange(0,df.shape[0])
    np.random.shuffle(idx)
    i=idx[0]
    #feature=np.zeros((96,64))
    y=df.values[i,1:-1]
    x,fs=librosa.load(df["mp3_path"][i])
    X=preprocess_sound.preprocess_sound(x, fs)
    X_slice=X[random.randint(0,X.shape[0]-1),:,:]
    #X_slice.reshape(-1,96,64,1)
    while True:
        yield{'X': X_slice[:,:,None],'y': y}


    
def data_generator(batch_size, train_idx, active_streamers=1,
                        rate=64):
    #batch_size=32
    streamer = pescador.Streamer(feature_sampler, df,train_idx)
    #seeds.append(streamer)

    # Randomly shuffle the seeds
    #random.shuffle(seeds)

    #mux = pescador.StochasticMux(streamer, active_streamers, rate=rate)
    mux=streamer
    if batch_size == 1:
        return mux
    else:
        return pescador.maps.buffer_stream(mux, batch_size)


# # Train the model

# In[31]:


import pescador


# In[32]:


epochs=12
batch_size=32
batches=data_generator(batch_size,train_idx,active_streamers=1,rate=64)
steps_per_epoch = len(train_idx) // batch_size


# In[36]:


#create X_test and y_test
X_test,Y_test=batchmatrix(32,df,test_idx)
print(X_test.shape,Y_test.shape)


# In[21]:


hist = new_model.fit_generator(
        pescador.tuples(batches, 'X', 'y'),
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        verbose=1,
        validation_data=(X_test, Y_test)
        )


# In[34]:


hist = new_model.fit_generator(
        pescador.maps.keras_tuples(data_generator, 'X', 'y'),
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        verbose=1,
        validation_data=(X_test, Y_test)
        )


# In[ ]:


scores = new_model.evaluate(X_test, Y_test, verbose=0)


# In[ ]:


new_model.fit_generator(
        pescador.tuples(batches, 'X', 'y'),
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        verbose=1,
        #validation_data=(X_test, Y_test)
        )
#except KeyboardInterrupt:
#    print("Stopping early")
#finally:
#print("Finished: {}".format(datetime.datetime.now()))

scores = new_model.evaluate(X_test, Y_test, verbose=0)


# # Use pickle to store preprocessed features

# In[17]:


#this takes input as batch size and output the input batch and ground truth labels
def batchmatrix_pickle(df,idx, folder):
    #idx=np.arange(0,df.shape[0])
    #np.random.shuffle(idx)
    #idx_shuffled=idx[:batch_size]
    batchmatrix=np.zeros((1,96,64))
    batchy=np.zeros((batch_size,188))
    for m,i in enumerate(idx):
        x,fs=librosa.load(df["mp3_path"][i])
        X=preprocess_sound.preprocess_sound(x, fs)
        X_slice=X[random.randint(0,X.shape[0]-1),:,:]
        batchmatrix=X_slice
        batchy=df.values[i,1:-1]
        output = open('./pickles/'+folder+'/'+str(i)+'.pkl', 'wb')
        pickle.dump([batchmatrix,batchy], output)
        output.close()
        
    return 0


# In[18]:


import pickle
batchmatrix_pickle(df,test_idx, test)
batchmatrix_pickle(df,train_idx, train)

