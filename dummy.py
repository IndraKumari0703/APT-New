import warnings
from numpy import isnan
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore")
import os
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import BatchNormalization
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Input
from keras.utils import plot_model
import pandas as pd
import numpy as np
import ipaddress
from FS.ssa import jfs as jfs_1
from FS.hho import jfs as jfs_2
from FS.HHO_SSA_Hyb import jfs as jfs_3
from FS.ga import jfs as jfs_0
from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, matthews_corrcoef
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Embedding, Flatten
from keras.layers import LSTM, SimpleRNN, GRU, Bidirectional, BatchNormalization,Convolution1D,MaxPooling1D, Reshape, GlobalAveragePooling1D
from keras.utils import to_categorical
from imblearn.over_sampling import RandomOverSampler
import lightgbm as lgb
from sklearn.ensemble import AdaBoostClassifier
# from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.callbacks import ModelCheckpoint,EarlyStopping
import tensorflow as tf
from scipy.stats import kurtosis,skew
import smote_variants as sv
from Sub_functions import Main_perf_val_acc_sen_spe_1_prc, Main_perf_val_acc_sen_spe_1,Main_perf_val_acc_sen_spe_2, \
                        Main_perf_val_acc_sen_spe_3, Main_perf_val_acc_sen_spe_4, ext_main_prc
# from sklearn.externals import joblib
from sklearn import svm
import joblib
from sklearn import metrics
from pycm import *


model = Sequential()
model.add(Bidirectional(LSTM(64, activation='relu'), input_shape=(85,1)))
model.add(Reshape((128, 1), input_shape=(128,)))
model.add(MaxPooling1D(pool_size=(5)))
model.add(BatchNormalization())
model.add(Bidirectional(LSTM(128, return_sequences=False)))
model.add(Dropout(0.06))
model.add(Dense(5))
model.compile(optimizer='sgd', loss='mse')
#Create Flow Chart
plot_model(model, to_file='chart/main_BiLSTM_LiGBM_Classifier_Mod_ROC_AU_model_flowchart.png', show_shapes=True, show_layer_names=True)



model = Sequential()
model.add(Bidirectional(LSTM(64, activation='relu'), input_shape=(85,1)))
model.add(Reshape((128, 1), input_shape=(128,)))
model.add(MaxPooling1D(pool_size=(5)))
model.add(BatchNormalization())
model.add(Bidirectional(LSTM(128, return_sequences=False)))
model.add(Dropout(0.06))
model.add(Dense(5))
model.compile(optimizer='sgd', loss='mse')
plot_model(model, to_file='chart/main_BiLSTM_LiGBM_Classifier_ROC_AUC_flowchart.png', show_shapes=True, show_layer_names=True)


model = Sequential()
model.add(Bidirectional(LSTM(64, activation='relu'), input_shape=(85,1)))
model.add(Reshape((128, 1), input_shape=(128,)))
model.add(MaxPooling1D(pool_size=(5)))
model.add(BatchNormalization())
model.add(Bidirectional(LSTM(128, return_sequences=False)))
model.add(Dropout(0.06))

model.add(Dense(5))
model.compile(optimizer='sgd', loss='mse')
plot_model(model, to_file='chart/main_BiLSTM_Classifier_ROC_AUC_flowchart.png', show_shapes=True, show_layer_names=True)



model = Sequential()
model.add(Convolution1D(64, kernel_size=64, padding="same", activation="relu", input_shape=(85, 1)))
model.add(MaxPooling1D(pool_size=(10)))
model.add(BatchNormalization())
model.add(Bidirectional(LSTM(64, return_sequences=False)))
model.add(Reshape((128, 1), input_shape=(128,)))
model.add(MaxPooling1D(pool_size=(5)))
model.add(BatchNormalization())
model.add(Bidirectional(LSTM(128, return_sequences=False)))
model.add(Dropout(0.06))
model.add(Dense(5))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
plot_model(model, to_file='chart/main_CNN_LSTM_Classifier_ROC_AUC_flowchart.png', show_shapes=True, show_layer_names=True)

n_inputs=85
visible = Input(shape=(n_inputs,))
# encoder level 1
e = Dense(n_inputs * 2)(visible)
e = BatchNormalization()(e)
e = LeakyReLU()(e)
# encoder level 2
e = Dense(n_inputs)(e)
e = BatchNormalization()(e)
e = LeakyReLU()(e)
# bottleneck
n_bottleneck = n_inputs
bottleneck = Dense(n_bottleneck)(e)
# define decoder, level 1
d = Dense(n_inputs)(bottleneck)
d = BatchNormalization()(d)
d = LeakyReLU()(d)
# decoder level 2
d = Dense(n_inputs * 2)(d)
d = BatchNormalization()(d)
d = LeakyReLU()(d)
# output layer
output = Dense(5, activation='linear')(d)
# define autoencoder model
model = Model(inputs=visible, outputs=output)
# compile autoencoder model
model.compile(optimizer='adam', loss='mse')
plot_model(model, to_file='chart/main_SAE_LSTM_flowchart.png', show_shapes=True, show_layer_names=True)
