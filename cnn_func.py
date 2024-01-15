import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
np.random.seed(32113)

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
# from keras.layers.convolutional import ZeroPadding2D
import np_utils
import keras.utils
from keras.models import load_model


#filepath = "./data/raw_data/full%2Fraw%2Fdog.ndjson"
#df = pd.read_json(filepath, lines=True)

##############################################################################
#                             Aggregated functions                           #
##############################################################################

def CNN_image_recognition(df1,df2,df3,df4,sample=30000,binary=False,\
                        convlayer =64,neuron =100, batchsize =500, epoch =10):
    '''
    function:
    - CNN_image_recognition with 4 different topics
    - Creates CNN Keras model for image recognition
    - Outputs model and training and test data (X and Y)

    CNN's structure
    1. convlayer(default value = 64) convolution layers with 5*5 kernel
        - activation function = relu
    2. max pooling applied after convoluliton layers (pool size = 2*2)
    3. 1 layer of feed foward neural network with 100 neurons
        - activation function = relu
    4. dropout rate assigned as 20 percent to prevent overfitting
    5. final activation function = softmax

    uses adam optimizer with MSE loss
    during model fit, 20 percent of training data is used as validation.
    **************************neural network codes*****************************

    model = Sequential()
    model.add(Convolution2D(convlayer, 5, 5, activation='relu', input_shape=(42,28,1)))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Flatten())
    model.add(Dense(neuron, activation='relu'))
    model.add(Dropout(.20))
    model.add(Dense(category, activation='softmax'))

    model.compile(loss='mean_squared_error',
              optimizer='adam',
              metrics=['accuracy'])

    model.fit(X_train, y_train,
          batch_size=batchsize, nb_epoch=epoch, verbose=1,validation_split=0.2)

    ***************************************************************************
    - if you want to change optimizer, loss, kernel size, activation function,
    go to "CNN_model_builder" function and change them.

    - Filtering applied:
      "sample" value determines number of sample extract from each dataframe.
      for instance if sample = 30000,
      30000 rows are randomly chosen from one dataframe.

    input:
    dataframes = should be the output dataframe from "feature_engineering_CNN"
                 from feature_engineering_func.py
    df1 = dataframe 1
    df2 = dataframe 2
    df3 = dataframe 3
    df4 = dataframe 4
    sample= number of rows you want to extract frim each dataframe
    binary= if True, it uses bianry 1 and 0 for pixelated image, if false
            3rd dimension of each image will be time [boolean(True/False)]
    convlayer = number of convolution layer used for NN. [int]
    neuron = number of neuron used for 1 layer feed foward NN. [int]
    batchsize = number of batch size used for NN. [int]
    epoch = number of epoch used for NN. [int]

    Output:
    model = CNN model generated with keras
    it will also output training and test set in case you want to
    calculate model probabilities and evaluation.
    X_train
    X_test
    y_train
    y_test
    '''
    #from input dataframes, create 1 X dataframe and label
    Xtemp1, labeldump = image_identification_datasetup(df1,df2,sample=sample)
    Xtemp2, labeldump = image_identification_datasetup(df3,df4,sample=sample)
    label = sample*[0]+sample*[1]+sample*[2]+sample*[3]
    label = pd.Series(label)
    X = pd.concat([Xtemp1,Xtemp2], axis = 0)
    # runs CNN_model_builder with X and label. returns CNN model and X,Y train and test dataset
    return CNN_model_builder(X,label,binary=binary,category = 4,\
                            cnnp =[convlayer,neuron], fitp =[batchsize,epoch])




##############################################################################
#           functions for feature engineeering for ensemble methods          #
##############################################################################

def image_identification_datasetup(df1,df2,sample=30000):
    '''
    Function:
    - takes two dataframe (dataframe should be the output dataframe
      from "feature_engineering_CNN" of feature_engineering_func.py) and
      convine two dataframe into one.
    - it also creates label pd.series for CNN image recognition

    filter applied:
    - "sample" value determines number of sample extract from each dataframe.
       for instance if sample = 30000,
       30000 rows are randomly chosen from df1,df2,df3 and df4.
    - it also takeout countrycode and word columns

    inputs:
    2 dataframe
    sample = number of rows you want to extract frim each dataframe
    outputs:
    dataframe and a label

    '''
    random_index1 = np.random.choice(list(df1.index), sample, replace=False)
    random_index2 = np.random.choice(list(df2.index), sample, replace=False)
    df1 = df1.loc[list(random_index1)]
    df2 = df2.loc[list(random_index2)]

    df_test = pd.concat([df1,df2],axis = 0)
    df_test = df_test.drop(['countrycode','word'], axis=1)
    label = [1]*sample+[0]*sample
    # 1= df1, 0 = df2
    label = np.array(label)
    label = pd.Series(label)
    label.index = df_test.index
    return df_test,label



def CNN_model_builder(X,label,binary=False, category =4, cnnp =[64,100], fitp =[500,10]):
    '''
    Function:
    - Creates CNN Keras model for image recognition
    - Outputs model and training and test data (X and Y)

    CNN's structure
    1. convlayer(default value = 64) convolution layers with 5*5 kernel
        - activation function = relu
    2. max pooling applied after convoluliton layers (pool size = 2*2)
    3. 1 layer of feed foward neural network with 100 neurons
        - activation function = relu
    4. dropout rate assigned as 20 percent to prevent overfitting
    5. final activation function = softmax

    uses adam optimizer with MSE loss
    during model fit, 20 percent of training data is used as validation.

    **************************neural network codes*****************************

    model = Sequential()
    model.add(Convolution2D(cnnp[0], 5, 5, activation='relu', input_shape=(42,28,1)))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Flatten())
    model.add(Dense(cnnp[1], activation='relu'))
    model.add(Dropout(.20))
    model.add(Dense(category, activation='softmax'))

    model.compile(loss='mean_squared_error',
              optimizer='adam',
              metrics=['accuracy'])

    model.fit(X_train, y_train,
          batch_size=fitp[0], nb_epoch=fitp[1], verbose=1,validation_split=0.2)

    ***************************************************************************
    - if you want to change optimizer, loss, kernel size, activation function,
    you have to change code manually.

    input:
    X, label = output of image_identification_datasetup
    binary= if True, it uses bianry 1 and 0 for pixelated image, if false
            3rd dimension of each image will be time [boolean(True/False)]
    cnnp[0] = number of convolution layer used for NN. [int]
    cnnp[1] = number of neuron used for 1 layer feed foward NN. [int]
    fitp[0] = number of batch size used for NN. [int]
    fitp[1] = number of epoch used for NN. [int]

    Output:
    model = CNN model generated with keras
    it will also output training and test set in case you want to
    calculate model probabilities and evaluation.
    X_train
    X_test
    y_train
    y_test

    '''
    # when binary is True, the image is represented with 0 and 1.
    # meaning that non-zero values are replaced with 1.
    if binary:
        X = np.array(X)
        X[X != 0.0] = 1
        data_np = X
    # when binary is False, images contains time values as a 3rd dimension value.
    # the time values (in sec) is normalized 
    else:
        data_np = np.array(X)
        #normalizing time values
        data_np /= 10000
        data_np += 1
        data_np[data_np == 1.0] = 0

    label2 = keras.utils.to_categorical(label, category)
    data_np = data_np.reshape(len(data_np),107,11,1)

    #train,test split
    X_train,X_test,y_train,y_test =train_test_split(data_np,label2,test_size = 0.15, \
                                                    random_state=831713, stratify = label2)
    # KERAS model (explained above)
    model = Sequential()
    model.add(Convolution2D(cnnp[0], 5, 5, activation='relu', input_shape=(107,11,1)))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Flatten())
    model.add(Dense(cnnp[1], activation='relu'))
    model.add(Dropout(.20))
    model.add(Dense(category, activation='softmax'))

    model.compile(loss='mean_squared_error',
              optimizer='adam',
              metrics=['accuracy'])
    #once model is created, it will be fitted with training datasets
    history = model.fit(X_train, y_train,
          batch_size=fitp[0], epochs=fitp[1], verbose=1,validation_split=0.2)

    #score = model.evaluate(X_test, y_test,  batch_size=100)
    return model,X_train,y_train,X_test,y_test, history
