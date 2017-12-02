THEANO_FLAGS='device=cpu,device=gpu0'
THEANO_FLAGS='floatX=float32,device=gpu0,lib.cnmem=1'

import theanets
import climate
import numpy as np
import pandas as pd
import math
from sklearn import preprocessing
from datetime import datetime


climate.enable_default_logging()  # to print downhill's iteration result

path= 'C:/Users/Administrator/Desktop/Data';

SNR2 = [0, 5, 10]
MFCC = 22


X_train = [None] * MFCC
X_train = np.array(np.float64(X_train))
# print X_train.shape

start0 = datetime.now()

print ('loading MAG_y_train...')
for n in range(1,6):
    for level in SNR2:
        X_train_0 = np.genfromtxt(path +'/MFCCs_train'+ '_SNR_'+ str(level)+ '_noise_'+ str(n)+'.csv', delimiter=',')
        X_train = np.vstack((X_train, X_train_0))
        
X_train = np.array(X_train)
X_train = X_train[1:,:]
print ('X_train_Size: ', X_train.shape)
print ('MAG_y_train is now loaded!')

print ('loading MAG_y_train...')
Y_train_0 = np.genfromtxt(path + '/MAG_y_train_half+1.csv', delimiter=',')


Y_train=np.vstack([Y_train_0]* 15)
print ('MAG_y_train_Size: ', Y_train.shape)
print ('MAG_y_train is now loaded!')


len= X_train.shape[0]

train_size= np.int(math.floor(len*0.8))
train_data_0 =  X_train[0:train_size,:]
train_target_0 =  Y_train[0:train_size,:]

valid_data_0 =  X_train[train_size:len ,:]
valid_target_0 =  Y_train[train_size:len ,:]
                        # print(len)
                        # print(train_size)
                        # print train_data_0.shape
SNR = [ 0, 5, 10 ]

print('DATA loaded')
# for n in range(1,6):            #n=1,2,3,4,5
for n in range(1,6):

    for level in SNR:           #level=0,5,10

        print ('loading MAG_n_train...')
        n_train_0 = np.genfromtxt(path +'/MAG_n_train_half+1'+ '_SNR_'+ str(level)+ '_noise_'+ str(n)+'.csv', delimiter=',')
        # print n_train_0.shape   #(137246, 513)
        n_train =np.vstack([n_train_0]* 15)
        print ('MAG_n_train_Size: ', n_train.shape)
        print ('MAG_n_train is now loaded!')
        train_target_n_0 =  n_train [0:train_size,:]
        valid_target_n_0 =  n_train [train_size:len ,:]

        Y_train_0= np.hstack((train_target_0, train_target_n_0))
        Y_valid_0= np.hstack((valid_target_0, valid_target_n_0))
        layer=[1024,1024]
        output=1026
        lr=0.0003
        epoch=25
        # print Y_train_0.shape
        # print Y_valid_0.shape
        # print train_data_0.shape
        # print valid_data_0.shape
        stop0 = datetime.now()
        print ('Pre-processing Time: '+ str(stop0-start0))
        start1 = datetime.now()
        net = theanets.feedforward.Regressor( [MFCC,
                                              (layer[0],'relu'),
                                              (layer[1],'relu'),
                                              (output,'linear')],
                                              loss='mse')
        net.train([train_data_0, Y_train_0 ],
                  [valid_data_0, Y_valid_0 ],
                  algo='RProp',
                  learning_rate=lr,
                  # batch_size=128,
                  # hidden_dropout=0.3,
                  # optimize='layerwise',
                  iteration_size=epoch,
                  hidden_l2=0.01 )
        stop1 = datetime.now()
        print ('Training Time: '+ str(stop1-start1))
        start2 = datetime.now()
        for i in range(1,25):
            X_test_0 = np.genfromtxt(path +'/MFCCs_'+ str(i) +'_SNR_'+ str(level)+ '_noise_'+ str(n)+'_test.csv', delimiter=',')
            print('Processing the '+'sp_'+ str(i) +'_SNR_'+ str(level)+ '_noise_'+str(n) ) #or use str(index)
            Y_Predict= net.predict(X_test_0)
            name0 = path + '/Results' + '/out_hid+'+ str(layer[0])+ '+' + str(layer[1])+ '_lr'+str(lr)+ '_sp'+ str(i) +'_SNR'+ str(level)+ '_noise'+ str(n)+'.csv'
            np.savetxt(name0, Y_Predict, delimiter=',')

        net = None
        stop2 = datetime.now()
        print ('Testing Time: '+ str(stop2-start2))

