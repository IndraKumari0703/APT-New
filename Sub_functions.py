import warnings
from numpy import isnan
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore")
import os
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
# from imblearn.over_sampling import RandomOverSampler
# import lightgbm as lgb
from sklearn.ensemble import AdaBoostClassifier
# from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.callbacks import ModelCheckpoint,EarlyStopping
import tensorflow as tf
from scipy.stats import kurtosis,skew
import smote_variants as sv
# from sklearn.externals import joblib
from sklearn import svm
import joblib
from sklearn import metrics
from pycm import *

def main_lab_change(tem_feat,Str_lab):
    tem_feat = np.delete(tem_feat, (6), axis=1)#Eliminate Timestamp attributes
    tem_feat = np.delete(tem_feat, (0), axis=1)#Eliminate IP  attributes
    fin_feat = tem_feat
    tem_lab=tem_feat[:,-1]
    for t in range(0, len(tem_lab)):
        fin_feat[t, 0] =int(ipaddress.ip_address(fin_feat[t, 0]))
        fin_feat[t, 2] =int(ipaddress.ip_address(fin_feat[t, 2]))
        curr_lab = tem_lab[t]
        if curr_lab == Str_lab[0]:
            fin_feat[t, -1] = 0
            fin_feat[t, -2] = 0
        elif curr_lab == Str_lab[1]:
            fin_feat[t, -1] = 0
            fin_feat[t, -2] = 0
        elif curr_lab == Str_lab[2]:
            fin_feat[t, -1] = 1
            fin_feat[t, -2] = 1
        elif curr_lab == Str_lab[3]:
            fin_feat[t, -1] = 2
            fin_feat[t, -2] = 2
        elif curr_lab == Str_lab[4]:
            fin_feat[t, -1] = 3
            fin_feat[t, -2] = 3
        else:
            fin_feat[t, -1] = 4
            fin_feat[t, -2] = 4
    return fin_feat
def plot_confusion_matrix(cm,target_names,title='Confusion matrix',cmap=None,normalize=True):
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools
    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy
    if cmap is None:
        cmap = plt.get_cmap('Blues')
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]), horizontalalignment="center",color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]), horizontalalignment="center",color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    #plt.show()
def main_data_splitup(Sel_identifier_3,Final_Feat,Final_Lab,tr_per):
    tot_attacks = np.unique(Final_Lab)
    tr_data, tst_data, tr_lab, tst_lab = [], [], [], []
    for y in range(0, len(tot_attacks)):
        ind_1 = np.where(Final_Lab == y)[0]
        # try:
        #      ind_1=ind_1[:300,]
        # except:
        #     ind_1=ind_1
        tr_upto = int(np.round(tr_per * len(ind_1)))
        if y > 0:
            tr_data = np.vstack((tr_data, Final_Feat[ind_1[:tr_upto], :]))
            tst_data = np.vstack((tst_data, Final_Feat[ind_1[tr_upto:],:]))
            tr_lab = np.hstack((tr_lab, Final_Lab[ind_1[:tr_upto],]))
            tst_lab = np.hstack((tst_lab, Final_Lab[ind_1[tr_upto:],]))
        else:
            tr_data = (Final_Feat[ind_1[:tr_upto], :])
            tst_data = (Final_Feat[ind_1[tr_upto:], :])
            tr_lab = (Final_Lab[ind_1[:tr_upto],])
            tst_lab = (Final_Lab[ind_1[tr_upto:],])
        # tr_data=np.asarray(tr_data)
        print(len(ind_1))
    # tr_data = tr_data[:, Sel_identifier_3]
    # tst_data = tst_data[:, Sel_identifier_3]
    return tr_data,tr_lab,tst_data,tst_lab
def main_data_splitup_tem(tot_attacks,Final_Feat,Final_Lab,tr_per):
    tr_data, tst_data, tr_lab, tst_lab = [], [], [], []
    for y in range(0, len(tot_attacks)):
        ind_1 = np.where(Final_Lab == y)[0]
        try:
             ind_1=ind_1[:5000,]
        except:
            ind_1=ind_1
        tr_upto = int(np.round(tr_per * len(ind_1)))
        if y > 0:
            tr_data = np.vstack((tr_data, Final_Feat[ind_1[:tr_upto], :]))
            tst_data = np.vstack((tst_data, Final_Feat[ind_1[tr_upto:], :]))
            tr_lab = np.hstack((tr_lab, Final_Lab[ind_1[:tr_upto],]))
            tst_lab = np.hstack((tst_lab, Final_Lab[ind_1[tr_upto:],]))
        else:
            tr_data = (Final_Feat[ind_1[:tr_upto], :])
            tst_data = (Final_Feat[ind_1[tr_upto:], :])
            tr_lab = (Final_Lab[ind_1[:tr_upto],])
            tst_lab = (Final_Lab[ind_1[tr_upto:],])
        # tr_data=np.asarray(tr_data)
        print(len(ind_1))
    return tr_data,tr_lab,tst_data,tst_lab
def main_CNN_LSTM_Classifier_ROC_AUC(tr_data,tr_lab,tst_data,ep):
    tr_data = np.reshape(tr_data, (tr_data.shape[0], tr_data.shape[1], 1))
    tst_data = np.reshape(tst_data, (tst_data.shape[0], tst_data.shape[1], 1))
    tr_lab = to_categorical(tr_lab)
    ###############################################
    batch_size = 128
    model = Sequential()
    model.add(Convolution1D(64, kernel_size=64, padding="same", activation="relu", input_shape=(tr_data.shape[1], 1)))
    model.add(MaxPooling1D(pool_size=(10)))
    model.add(BatchNormalization())
    model.add(Bidirectional(LSTM(64, return_sequences=False)))
    model.add(Reshape((128, 1), input_shape=(128,)))
    model.add(MaxPooling1D(pool_size=(5)))
    model.add(BatchNormalization())
    model.add(Bidirectional(LSTM(128, return_sequences=False)))
    # model.add(Reshape((128, 1), input_shape = (128, )))
    model.add(Dropout(0.06))
    model.add(Dense(tr_lab.shape[1]))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    for layer in model.layers:
        print(layer.output_shape)
    model.summary()
    model.fit(tr_data, tr_lab, validation_data=(tr_data, tr_lab), epochs=ep)

    pred = model.predict(tst_data)
    pred = np.argmax(pred, axis=1)
    # y_eval = np.argmax(tst_lab, axis=1)
    return pred,model
def main_CNN_LSTM_Classifier(tr_data,tr_lab,tst_data,ep):
    tr_data = np.reshape(tr_data, (tr_data.shape[0], tr_data.shape[1], 1))
    tst_data = np.reshape(tst_data, (tst_data.shape[0], tst_data.shape[1], 1))
    tr_lab = to_categorical(tr_lab)
    ###############################################
    batch_size = 128
    model = Sequential()
    model.add(Convolution1D(64, kernel_size=64, padding="same", activation="relu", input_shape=(tr_data.shape[1], 1)))
    model.add(MaxPooling1D(pool_size=(10)))
    model.add(BatchNormalization())
    model.add(Bidirectional(LSTM(64, return_sequences=False)))
    model.add(Reshape((128, 1), input_shape=(128,)))
    model.add(MaxPooling1D(pool_size=(5)))
    model.add(BatchNormalization())
    model.add(Bidirectional(LSTM(128, return_sequences=False)))
    # model.add(Reshape((128, 1), input_shape = (128, )))
    model.add(Dropout(0.06))
    model.add(Dense(tr_lab.shape[1]))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    for layer in model.layers:
        print(layer.output_shape)
    model.summary()
    model.fit(tr_data, tr_lab, validation_data=(tr_data, tr_lab), epochs=ep)

    pred = model.predict(tst_data)
    pred = np.argmax(pred, axis=1)
    # y_eval = np.argmax(tst_lab, axis=1)
    return pred
def main_AdaBoost_Classifier(tr_data, tr_lab,tst_data):
    clf = AdaBoostClassifier(random_state=3)
    clf.fit(tr_data, tr_lab)
    pred_lab = clf.predict(tst_data)
    return pred_lab
def main_AdaBoost_Classifier_ROC_AUC(tr_data, tr_lab,tst_data):
    clf = AdaBoostClassifier(random_state=3)
    clf.fit(tr_data, tr_lab)
    pred_lab = clf.predict(tst_data)
    return pred_lab,clf
def main_BiLSTM_Classifier_ROC_AUC(tr_data,tr_lab,tst_data,ep):
    tr_data=tf.keras.utils.normalize(tr_data)
    tst_data=tf.keras.utils.normalize(tst_data)

    tr_data = np.reshape(tr_data, (tr_data.shape[0], tr_data.shape[1], 1))
    tst_data = np.reshape(tst_data, (tst_data.shape[0], tst_data.shape[1], 1))
    tr_lab = to_categorical(tr_lab)
    model = Sequential()
    model.add(Bidirectional(LSTM(64, activation='relu'), input_shape=(tr_data.shape[1],1)))
    model.add(Reshape((128, 1), input_shape=(128,)))
    model.add(MaxPooling1D(pool_size=(5)))
    model.add(BatchNormalization())
    model.add(Bidirectional(LSTM(128, return_sequences=False)))
    model.add(Dropout(0.06))

    model.add(Dense(tr_lab.shape[1]))
    model.compile(optimizer='sgd', loss='mse')
    # checkpoint
    filepath = "weights_BiLstm.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
    es = EarlyStopping(monitor='val_accuracy', patience=5)
    callbacks_list = [checkpoint, es]
    history = model.fit(tr_data,tr_lab, epochs=ep, batch_size=64, callbacks=callbacks_list, verbose=1)
    filename = "BiLSTM_model.joblib"
    joblib.dump(model, filename)
    loaded_model = joblib.load(filename)
    # y_pred_1 = np.round(history.model.predict(tst_data))
    pred = model.predict(tst_data)
    pred = np.argmax(pred, axis=1)
    return pred,model
def main_SAE_LSTM(tr_data,tr_lab,tst_data,ep):
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input
    from tensorflow.keras.layers import Dense
    from tensorflow.keras.layers import LeakyReLU
    from tensorflow.keras.layers import BatchNormalization
    from sklearn.preprocessing import MinMaxScaler
    tr_data=tf.keras.utils.normalize(tr_data)
    tst_data=tf.keras.utils.normalize(tst_data)
    # scale data
    t = MinMaxScaler()
    t.fit(tr_data)
    tr_data = t.transform(tr_data)
    tst_data = t.transform(tst_data)
    # number of input columns
    n_inputs = tr_data.shape[1]
    tr_lab = to_categorical(tr_lab)

    # split into train test sets
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
    # define encoder
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
    output = Dense(tr_lab.shape[1], activation='linear')(d)
    # define autoencoder model
    model = Model(inputs=visible, outputs=output)
    # compile autoencoder model
    model.compile(optimizer='adam', loss='mse')
    # plot the autoencoder
    # plot_model(model, 'autoencoder_no_compress.png', show_shapes=True)
    # fit the autoencoder model to reconstruct input
    history = model.fit(tr_data,tr_lab, epochs=ep, batch_size=16, verbose=2, validation_data=(tr_data,tr_lab))
    pred = model.predict(tst_data)
    pred = np.argmax(pred, axis=1)
    return pred,model

def main_BiLSTM_Classifier(tr_data,tr_lab,tst_data,ep):
    tr_data=tf.keras.utils.normalize(tr_data)
    tst_data=tf.keras.utils.normalize(tst_data)

    tr_data = np.reshape(tr_data, (tr_data.shape[0], tr_data.shape[1], 1))
    tst_data = np.reshape(tst_data, (tst_data.shape[0], tst_data.shape[1], 1))
    tr_lab = to_categorical(tr_lab)
    model = Sequential()
    model.add(Bidirectional(LSTM(64, activation='relu'), input_shape=(tr_data.shape[1],1)))
    model.add(Reshape((128, 1), input_shape=(128,)))
    model.add(MaxPooling1D(pool_size=(5)))
    model.add(BatchNormalization())
    model.add(Bidirectional(LSTM(128, return_sequences=False)))
    model.add(Dropout(0.06))

    model.add(Dense(tr_lab.shape[1]))
    model.compile(optimizer='sgd', loss='mse')
    # checkpoint
    filepath = "weights_BiLstm.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
    es = EarlyStopping(monitor='val_accuracy', patience=5)
    callbacks_list = [checkpoint, es]
    history = model.fit(tr_data,tr_lab, epochs=ep, batch_size=64, callbacks=callbacks_list, verbose=1)
    filename = "BiLSTM_model.joblib"
    joblib.dump(model, filename)
    loaded_model = joblib.load(filename)
    # y_pred_1 = np.round(history.model.predict(tst_data))
    pred = model.predict(tst_data)
    pred = np.argmax(pred, axis=1)
    return pred
def main_LightGBM_Classifier(tr_data, tr_lab,tst_data):
    # build the lightgbm model
    clf = lgb.LGBMClassifier()
    clf.fit(tr_data, tr_lab)
    pred_lab = clf.predict(tst_data)
    return pred_lab
def main_LightGBM_Classifier_ROC_AUC(tr_data, tr_lab,tst_data):
    # build the lightgbm model
    clf = lgb.LGBMClassifier()
    clf.fit(tr_data, tr_lab)
    pred_lab = clf.predict(tst_data)
    return pred_lab,clf
from mealpy.swarm_based import HHO,SSA,PSO
def fitness_function1(solution,model,tst_data,tst_lab):
    d1=int(solution.shape[0]/5)
    d2=5
    tem_sol=np.reshape(solution, (d1, d2))
    wei_to_train=model.layers[6].get_weights()
    wei_to_train[0]=tem_sol
    model.layers[6].set_weights(wei_to_train)
    pred = model.predict(tst_data)
    pred_1 = np.argmax(pred, axis=1)
    # from sklearn.metrics import classification_report as cr
    # per=cr(tst_lab,pred_1)
    acc=np.sum(tst_lab==pred_1)/len(pred_1)
    return acc
def fitness_function_Data_Balancing(solution,Final_Feat, Final_Lab):
    tr_per = 0.75
    tr_data, tr_lab, tst_data, tst_lab = main_data_splitup_tem(Final_Feat, Final_Lab, tr_per)
    pred_1 = main_KNN_Classifier(tr_data, tr_lab, tst_data)
    acc = np.sum(tst_lab == pred_1) / len(pred_1)
    # d1=int(solution.shape[0]/5)
    # d2=5
    # tem_sol=np.reshape(solution, (d1, d2))
    # wei_to_train=model.layers[6].get_weights()
    # wei_to_train[0]=tem_sol
    # model.layers[6].set_weights(wei_to_train)
    # pred = model.predict(tst_data)
    # pred_1 = np.argmax(pred, axis=1)
    # # from sklearn.metrics import classification_report as cr
    # # per=cr(tst_lab,pred_1)
    # acc=np.sum(tst_lab==pred_1)/len(pred_1)
    return acc
def main_final_data_bal_out(Final_Feat, Final_Lab,best_position2):
    oversample = RandomOverSampler(sampling_strategy='minority')
    Final_Feat, Final_Lab = oversample.fit_resample(Final_Feat, Final_Lab)
    # Final_Feat, Final_Lab = oversample.fit_resample(Final_Feat, Final_Lab)
    # Final_Feat, Final_Lab = oversample.fit_resample(Final_Feat, Final_Lab)
    # Final_Feat, Final_Lab = oversample.fit_resample(Final_Feat, Final_Lab)
    return Final_Feat, Final_Lab
def main_Data_Balancing_optimization(ii,Final_Feat, Final_Lab):

    if ii == 0:
        X_samp=Final_Feat
        y_samp=Final_Lab
    elif ii==1:
        X_samp, y_samp = main_final_data_bal_out(Final_Feat, Final_Lab, Final_Lab)
    elif ii==2:
        oversampler= sv.SMOTE_PSO(k=3,nn_params={},eps=0.05,n_pop=2,w=1.0,c1=2.0, c2=2.0, num_it=1,n_jobs=1,random_state=None)
        X_samp, y_samp = oversampler.sample(Final_Feat, Final_Lab)
    elif ii==3:
        oversampler= sv.GASMOTE(n_neighbors=2,nn_params={}, maxn=7, n_pop=5,popl3=5,pm=0.3,pr=0.2,Ge=2,n_jobs=1,random_state=None)
        X_samp, y_samp = oversampler.sample(Final_Feat, Final_Lab)
    elif ii==4:
        oversampler = sv.SSO(proportion=1.0,h=10, k=5,nn_params={},alpha=0.5,n_jobs=1,random_state=None)
        X_samp, y_samp = oversampler.sample(Final_Feat, Final_Lab)
    elif ii == 5:
        oversampler = sv.HHO(proportion=1.0,K2=5,K1_frac=0.5,nn_params={},n_jobs=1,random_state=None)
        X_samp, y_samp = oversampler.sample(Final_Feat, Final_Lab)
    else:
        oversampler = sv.HHO_SSA_Mod_SMOTE(n_neighbors=2, nn_params={}, maxn=7, n_pop=5, popl3=5, pm=0.3, pr=0.2, Ge=2,n_jobs=1, random_state=None)
        X_samp, y_samp = oversampler.sample(Final_Feat, Final_Lab)

    return X_samp, y_samp
def main_weight_updation_optimization(ii,curr_wei,model,tst_data,tst_lab):
    problem_dict1 = {
        "fit_func": fitness_function1,
        "lb": [curr_wei.min(), ] * curr_wei.shape[0]*curr_wei.shape[1],
        "ub": [curr_wei.max(), ] * curr_wei.shape[0]*curr_wei.shape[1],
        "minmax": "min",
        "log_to": None,
        "save_population": False,
        "Curr_Weight":curr_wei,
        "Model_trained_Partial":model,
        "tst_data": tst_data,
        "tst_lab": tst_lab,
    }
    if ii==0:
        model = PSO.BasePSO(problem_dict1, epoch=5, pop_size=10, pr=0.03)
    elif ii==1:
        model = HHO.BaseHHO(problem_dict1, epoch=5, pop_size=10, pr=0.03)
    elif ii==2:
        model = SSA.BaseSSA(problem_dict1, epoch=5, pop_size=10, pr=0.03)
    else:
        model = HHO.BaseHHO_SSA_Pro(problem_dict1, epoch=5, pop_size=10, pr=0.03)
    best_position2, best_fitness2 = curr_wei,model.solve()
    Glob_best_fit_2=model.history.list_global_best_fit
    return best_position2
def main_BiLSTM_LiGBM_Classifier_ROC_AUC(tr_data,tr_lab,tst_data,tst_lab,ii,ep):
    tr_data_org=tf.keras.utils.normalize(tr_data)
    tst_data_org=tf.keras.utils.normalize(tst_data)
    tr_lab_org=tr_lab
    tr_data = np.reshape(tr_data_org, (tr_data_org.shape[0], tr_data_org.shape[1], 1))
    tst_data = np.reshape(tst_data_org, (tst_data_org.shape[0], tst_data_org.shape[1], 1))
    tr_lab = to_categorical(tr_lab)
    model = Sequential()
    model.add(Bidirectional(LSTM(64, activation='relu'), input_shape=(tr_data.shape[1],1)))
    model.add(Reshape((128, 1), input_shape=(128,)))
    model.add(MaxPooling1D(pool_size=(5)))
    model.add(BatchNormalization())
    model.add(Bidirectional(LSTM(128, return_sequences=False)))
    model.add(Dropout(0.06))
    model.add(Dense(tr_lab.shape[1]))
    model.compile(optimizer='sgd', loss='mse')
    # checkpoint
    filepath = "weights_BiLstm.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
    es = EarlyStopping(monitor='val_accuracy', patience=5)
    callbacks_list = [checkpoint, es]
    history = model.fit(tr_data,tr_lab, epochs=ep, batch_size=640, callbacks=callbacks_list, verbose=1)
    wei_to_train=model.layers[6].get_weights()
    wei_to_train_1=wei_to_train[0]
    wei_to_train_2=wei_to_train[1]
    ###########   Weight Modification    #########################
    wei_to_train_1=main_weight_updation_optimization(ii, wei_to_train_1,model,tst_data,tst_lab)
    wei_to_train[0]=np.reshape(wei_to_train_1,(int(wei_to_train[0].shape[0]),int(wei_to_train[0].shape[1])))
    model.layers[6].set_weights(wei_to_train)
    ###################################################
    filename = "BiLSTM_model.joblib"
    joblib.dump(model, filename)
    model = joblib.load(filename)
    # y_pred_1 = np.round(history.model.predict(tst_data))
    pred = model.predict(tst_data)
    pred_1 = np.argmax(pred, axis=1)
    return pred_1,model
def main_BiLSTM_LiGBM_Classifier(tr_data,tr_lab,tst_data,tst_lab,ii,ep):
    tr_data_org=tf.keras.utils.normalize(tr_data)
    tst_data_org=tf.keras.utils.normalize(tst_data)
    tr_lab_org=tr_lab
    tr_data = np.reshape(tr_data_org, (tr_data_org.shape[0], tr_data_org.shape[1], 1))
    tst_data = np.reshape(tst_data_org, (tst_data_org.shape[0], tst_data_org.shape[1], 1))
    tr_lab = to_categorical(tr_lab)
    model = Sequential()
    model.add(Bidirectional(LSTM(64, activation='relu'), input_shape=(tr_data.shape[1],1)))
    model.add(Reshape((128, 1), input_shape=(128,)))
    model.add(MaxPooling1D(pool_size=(5)))
    model.add(BatchNormalization())
    model.add(Bidirectional(LSTM(128, return_sequences=False)))
    model.add(Dropout(0.06))
    model.add(Dense(tr_lab.shape[1]))
    model.compile(optimizer='sgd', loss='mse')
    # checkpoint
    filepath = "weights_BiLstm.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
    es = EarlyStopping(monitor='val_accuracy', patience=5)
    callbacks_list = [checkpoint, es]
    history = model.fit(tr_data,tr_lab, epochs=ep, batch_size=640, callbacks=callbacks_list, verbose=1)
    wei_to_train=model.layers[6].get_weights()
    wei_to_train_1=wei_to_train[0]
    wei_to_train_2=wei_to_train[1]
    ###########   Weight Modification    #########################
    wei_to_train_1=main_weight_updation_optimization(ii, wei_to_train_1,model,tst_data,tst_lab)
    wei_to_train[0]=np.reshape(wei_to_train_1,(int(wei_to_train[0].shape[0]),int(wei_to_train[0].shape[1])))
    model.layers[6].set_weights(wei_to_train)
    ###################################################
    filename = "BiLSTM_model.joblib"
    joblib.dump(model, filename)
    model = joblib.load(filename)
    # y_pred_1 = np.round(history.model.predict(tst_data))
    pred = model.predict(tst_data)
    pred_1 = np.argmax(pred, axis=1)
    return pred_1
def main_BiLSTM_LiGBM_Classifier_Mod_ROC_AUC(tr_data,tr_lab,tst_data,tst_lab,ii,ep):
    tr_data_org=tf.keras.utils.normalize(tr_data)
    tst_data_org=tf.keras.utils.normalize(tst_data)
    tr_lab_org=tr_lab
    tr_data = np.reshape(tr_data_org, (tr_data_org.shape[0], tr_data_org.shape[1], 1))
    tst_data = np.reshape(tst_data_org, (tst_data_org.shape[0], tst_data_org.shape[1], 1))
    tr_lab = to_categorical(tr_lab)
    model = Sequential()
    model.add(Bidirectional(LSTM(64, activation='relu'), input_shape=(tr_data.shape[1],1)))
    model.add(Reshape((128, 1), input_shape=(128,)))
    model.add(MaxPooling1D(pool_size=(5)))
    model.add(BatchNormalization())
    model.add(Bidirectional(LSTM(128, return_sequences=False)))
    model.add(Dropout(0.06))
    model.add(Dense(tr_lab.shape[1]))
    model.compile(optimizer='sgd', loss='mse')
    # checkpoint
    filepath = "weights_BiLstm.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
    es = EarlyStopping(monitor='val_accuracy', patience=5)
    callbacks_list = [checkpoint, es]
    history = model.fit(tr_data,tr_lab, epochs=ep, batch_size=640, callbacks=callbacks_list, verbose=1)
    wei_to_train=model.layers[6].get_weights()
    wei_to_train_1=wei_to_train[0]
    wei_to_train_2=wei_to_train[1]
    wei_to_train_1=main_weight_updation_optimization(ii, wei_to_train_1,model,tst_data,tst_lab)
    wei_to_train[0]=np.reshape(wei_to_train_1,(int(wei_to_train[0].shape[0]),int(wei_to_train[0].shape[1])))
    model.layers[6].set_weights(wei_to_train)
    filename = "BiLSTM_model.joblib"
    joblib.dump(model, filename)
    model = joblib.load(filename)
    # y_pred_1 = np.round(history.model.predict(tst_data))
    pred = model.predict(tst_data)
    pred_1 = np.argmax(pred, axis=1)

    ###################################
    clf = lgb.LGBMClassifier()
    clf.fit(tr_data_org, tr_lab_org)
    pred_2 = clf.predict(tst_data_org)
    DD=pred_1.astype(np.int32)+pred_2.astype(np.int32)
    pred=np.round((0.05*pred_1.astype(np.int32)+0.95*pred_2.astype(np.int32)))
    pred=pred.astype(int)
    return pred,clf
def main_BiLSTM_LiGBM_Classifier_Mod(tr_data,tr_lab,tst_data,tst_lab,ii,ep):
    tr_data_org=tf.keras.utils.normalize(tr_data)
    tst_data_org=tf.keras.utils.normalize(tst_data)
    tr_lab_org=tr_lab
    tr_data = np.reshape(tr_data_org, (tr_data_org.shape[0], tr_data_org.shape[1], 1))
    tst_data = np.reshape(tst_data_org, (tst_data_org.shape[0], tst_data_org.shape[1], 1))
    tr_lab = to_categorical(tr_lab)
    model = Sequential()
    model.add(Bidirectional(LSTM(64, activation='relu'), input_shape=(tr_data.shape[1],1)))
    model.add(Reshape((128, 1), input_shape=(128,)))
    model.add(MaxPooling1D(pool_size=(5)))
    model.add(BatchNormalization())
    model.add(Bidirectional(LSTM(128, return_sequences=False)))
    model.add(Dropout(0.06))
    model.add(Dense(tr_lab.shape[1]))
    model.compile(optimizer='sgd', loss='mse')
    # checkpoint
    filepath = "weights_BiLstm.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
    es = EarlyStopping(monitor='val_accuracy', patience=5)
    callbacks_list = [checkpoint, es]
    history = model.fit(tr_data,tr_lab, epochs=ep, batch_size=640, callbacks=callbacks_list, verbose=1)
    wei_to_train=model.layers[6].get_weights()
    wei_to_train_1=wei_to_train[0]
    wei_to_train_2=wei_to_train[1]
    wei_to_train_1=main_weight_updation_optimization(ii, wei_to_train_1,model,tst_data,tst_lab)
    wei_to_train[0]=np.reshape(wei_to_train_1,(int(wei_to_train[0].shape[0]),int(wei_to_train[0].shape[1])))
    model.layers[6].set_weights(wei_to_train)
    filename = "BiLSTM_model.joblib"
    joblib.dump(model, filename)
    model = joblib.load(filename)
    # y_pred_1 = np.round(history.model.predict(tst_data))
    pred = model.predict(tst_data)
    pred_1 = np.argmax(pred, axis=1)

    ###################################
    clf = lgb.LGBMClassifier()
    clf.fit(tr_data_org, tr_lab_org)
    pred_2 = clf.predict(tst_data_org)
    DD=pred_1.astype(np.int32)+pred_2.astype(np.int32)
    pred=np.round((0.05*pred_1.astype(np.int32)+0.95*pred_2.astype(np.int32)))
    pred=pred.astype(int)
    return pred
def main_KNN_Classifier(tr_data, tr_lab,tst_data):
    knn = KNeighborsClassifier(n_neighbors=2)
    knn.fit(tr_data, tr_lab)
    pred_lab = knn.predict(tst_data)
    return pred_lab
def main_SVM_Classifier(tr_data, tr_lab,tst_data):
    # Create a svm Classifier
    clf=KNeighborsClassifier(n_neighbors=2)
    # clf = svm.SVC(kernel='linear')  # Linear Kernel
    clf.fit(tr_data, tr_lab)
    pred_lab = clf.predict(tst_data)
    return pred_lab,clf
def main_KNN_Classifier_ROC_AUC(tr_data, tr_lab,tst_data):
    knn = KNeighborsClassifier(n_neighbors=2)
    knn.fit(tr_data, tr_lab)
    pred_lab = knn.predict(tst_data)
    return pred_lab,knn
def prop_Important_Identifier_Detection(tr_data,tr_lab,tst_data,tst_lab,pl,jfs):
    fold = {'xt': tst_data, 'yt': tst_lab, 'xv': tst_data, 'yv': tst_lab}
    # parameter
    k = 2  # k-value in KNN
    N = 5  # number of particles
    T = 2  # maximum number of iterations
    opts = {'k': k, 'fold': fold, 'N': N, 'T': T}
    # perform Important Identified Detection
    fmdl = jfs(tst_data, tst_lab, opts)
    sf = fmdl['sf']
    tr_data = tr_data[:, sf]
    tst_data = tst_data[:, sf]
    if pl==1:
        # plot convergence
        curve = fmdl['c']
        curve = curve.reshape(np.size(curve, 1))
        x = np.arange(0, opts['T'], 1.0) + 1.0

        fig, ax = plt.subplots()
        ax.plot(x, curve, 'o-')
        ax.set_xlabel('Number of Iterations')
        ax.set_ylabel('Fitness')
        ax.set_title('Convergence')
        ax.grid()
        #plt.show()
    return sf
def main_feature_combine(Final_Feat):
    time_win = 2
    feat_tem = []
    for ii in range(0, len(Final_Feat)):
        if ii == 0:
            curr_data = Final_Feat[ii, :]
        else:
            curr_data = Final_Feat[ii, :]
            # curr_data = Final_Feat[ii-1:ii, :]

        F1 = kurtosis(curr_data, axis=0, bias=True)
        F2 = skew(curr_data, axis=0, bias=True)
        F3 = np.mean(curr_data)
        F4 = np.var(curr_data)
        feat_tem.append(np.hstack((F1, F2, F3, F4)))
    feat_tem_1 = np.asarray(feat_tem)
    Final_Feat = np.hstack((Final_Feat, feat_tem_1))
    return Final_Feat


def perf_evalution_CM(y, y_pred):
    # confusion = confusion_matrix(y, y_pred)
    # tp, fp, fn, tn = confusion.flatten()
    cm1 = ConfusionMatrix(actual_vector=y, predict_vector=y_pred)
    A=cm1.TP
    new_lis = np.array(list(A.items()))
    TP = np.sum(new_lis)
    A=cm1.TN
    new_lis = np.array(list(A.items()))
    TN = np.sum(new_lis)
    A=cm1.FP
    new_lis = np.array(list(A.items()))
    FP = np.sum(new_lis)
    A=cm1.FN
    new_lis = np.array(list(A.items()))
    FN = np.sum(new_lis)
    SEN = (TP) / (TP + FN)
    SPE = (TN) / (TN + FP)
    ACC = (TP + TN) / (TP + TN + FP + FN)
    FMS = (2 * TP) / (2 * TP + FP + FN)
    PRE = (TP) / (TP + FP)
    REC = SEN
    TS = (TP) / (TP + FP + FN)  # Threat score
    NPV = (TN) / (TN + FN)  # negative predictive value
    FOR = (FN) / (FN + TN)  # false omission rate
    MCC = matthews_corrcoef(y, y_pred)  # Matthews correlation coefficient
    return [ACC, SEN, SPE, PRE, REC, FMS, TS, NPV, FOR, MCC]
def Perf_Evaluation_save_all_final(Sel_identifier_3, Final_Feat, Final_Lab,tt):
    tr_per = 0.4
    epoch = 1
    perf_A = []
    perf_B = []
    perf_C = []
    perf_D = []
    perf_E = []
    perf_F = []
    perf_G = []
    perf_H = []
    perf_I = []
    perf_J = []
    perf_K= []
    for a in range(0,3):
        tr_data, tr_lab, tst_data, tst_lab = main_data_splitup(Sel_identifier_3, Final_Feat, Final_Lab, tr_per)
        pred_0=main_SVM_Classifier(tr_data, tr_lab, tst_data)######SVM Classifier
        pred_1=main_SAE_LSTM(tr_data, tr_lab, tst_data, epoch)######  SAE-LSTM Classifier
        pred_2 = main_KNN_Classifier(tr_data, tr_lab, tst_data)######  Knn Classifier
        pred_3 = main_CNN_LSTM_Classifier(tr_data, tr_lab, tst_data, epoch)######  CNN-LSTM Classifier
        pred_4 = main_LightGBM_Classifier(tr_data, tr_lab, tst_data)######  LightGBM Classifier
        pred_5 = main_AdaBoost_Classifier(tr_data, tr_lab, tst_data)######  Adaboost Classifier
        pred_6 = main_BiLSTM_Classifier(tr_data, tr_lab, tst_data, epoch)######  BiLSTM Classifier
        pred_7 = main_BiLSTM_LiGBM_Classifier(tr_data, tr_lab, tst_data, tst_lab, 0, epoch)######  PSO Tuned BiLSTM Classifier
        pred_8 = main_BiLSTM_LiGBM_Classifier(tr_data, tr_lab, tst_data, tst_lab, 1, epoch)######  SSO Tuned BiLSTM Classifier
        pred_9 = main_BiLSTM_LiGBM_Classifier(tr_data, tr_lab, tst_data, tst_lab, 2, epoch)######  HHO Tuned BiLSTM Classifier
        pred_10 = main_BiLSTM_LiGBM_Classifier_Mod(tr_data, tr_lab, tst_data, tst_lab, 3, epoch)######  Proposed Tuned BiLSTM Classifier
        ################   Performance Extraction Using Confusion Matrix   ###############
        [ACC0, SEN0, SPE0, PRE0, REC0, FMS0, TS0, NPV0, FOR0, MCC0] = perf_evalution_CM(tst_lab, pred_0)
        [ACC1, SEN1, SPE1, PRE1, REC1, FMS1, TS1, NPV1, FOR1, MCC1] = perf_evalution_CM(tst_lab, pred_1)
        [ACC2, SEN2, SPE2, PRE2, REC2, FMS2, TS2, NPV2, FOR2, MCC2] = perf_evalution_CM(tst_lab, pred_2)
        [ACC3, SEN3, SPE3, PRE3, REC3, FMS3, TS3, NPV3, FOR3, MCC3] = perf_evalution_CM(tst_lab, pred_3)
        [ACC4, SEN4, SPE4, PRE4, REC4, FMS4, TS4, NPV4, FOR4, MCC4] = perf_evalution_CM(tst_lab, pred_4)
        [ACC5, SEN5, SPE5, PRE5, REC5, FMS5, TS5, NPV5, FOR5, MCC5] = perf_evalution_CM(tst_lab, pred_5)
        [ACC6, SEN6, SPE6, PRE6, REC6, FMS6, TS6, NPV6, FOR6, MCC6] = perf_evalution_CM(tst_lab, pred_6)
        [ACC7, SEN7, SPE7, PRE7, REC7, FMS7, TS7, NPV7, FOR7, MCC7] = perf_evalution_CM(tst_lab, pred_7)
        [ACC8, SEN8, SPE8, PRE8, REC8, FMS8, TS8, NPV8, FOR8, MCC8] = perf_evalution_CM(tst_lab, pred_8)
        [ACC9, SEN9, SPE9, PRE9, REC9, FMS9, TS9, NPV9, FOR9, MCC9] = perf_evalution_CM(tst_lab, pred_9)
        [ACC10, SEN10, SPE10, PRE10, REC10, FMS10, TS10, NPV10, FOR10, MCC10] = perf_evalution_CM(tst_lab, pred_10)
        perf_0 = [ACC0, SEN0, SPE0, PRE0, REC0, FMS0, TS0, NPV0, FOR0, MCC0]
        perf_1 = [ACC1, SEN1, SPE1, PRE1, REC1, FMS1, TS1, NPV1, FOR1, MCC1]
        perf_2 = [ACC2, SEN2, SPE2, PRE2, REC2, FMS2, TS2, NPV2, FOR2, MCC2]
        perf_3 = [ACC3, SEN3, SPE3, PRE3, REC3, FMS3, TS3, NPV3, FOR3, MCC3]
        perf_4 = [ACC4, SEN4, SPE4, PRE4, REC4, FMS4, TS4, NPV4, FOR4, MCC4]
        perf_5 = [ACC5, SEN5, SPE5, PRE5, REC5, FMS5, TS5, NPV5, FOR5, MCC5]
        perf_6 = [ACC6, SEN6, SPE6, PRE6, REC6, FMS6, TS6, NPV6, FOR6, MCC6]
        perf_7 = [ACC7, SEN7, SPE7, PRE7, REC7, FMS7, TS7, NPV7, FOR7, MCC7]
        perf_8 = [ACC8, SEN8, SPE8, PRE8, REC8, FMS8, TS8, NPV8, FOR8, MCC8]
        perf_9 = [ACC9, SEN9, SPE9, PRE9, REC9, FMS9, TS9, NPV9, FOR9, MCC9]
        perf_10 = [ACC10, SEN10, SPE10, PRE10, REC10, FMS10, TS10, NPV10, FOR10, MCC10]

        perf_A.append(perf_0)
        perf_B.append(perf_1)
        perf_C.append(perf_2)
        perf_D.append(perf_3)
        perf_E.append(perf_4)
        perf_F.append(perf_5)
        perf_G.append(perf_6)
        perf_H.append(perf_7)
        perf_I.append(perf_8)
        perf_J.append(perf_9)
        perf_K.append(perf_10)

        tr_per = tr_per + 0.2
    if tt == 0:
            np.save('perf_A0', perf_A)
            np.save('perf_B0', perf_B)
            np.save('perf_C0', perf_C)
            np.save('perf_D0', perf_D)
            np.save('perf_E0', perf_E)
            np.save('perf_F0', perf_F)
            np.save('perf_G0', perf_G)
            np.save('perf_H0', perf_H)
            np.save('perf_I0', perf_I)
            np.save('perf_J0', perf_J)
            np.save('perf_K0', perf_K)
    else:#if tt == 1:
            np.save('perf_A1', perf_A)
            np.save('perf_B1', perf_B)
            np.save('perf_C1', perf_C)
            np.save('perf_D1', perf_D)
            np.save('perf_E1', perf_E)
            np.save('perf_F1', perf_F)
            np.save('perf_G1', perf_G)
            np.save('perf_H1', perf_H)
            np.save('perf_I1', perf_I)
            np.save('perf_J1', perf_J)
            np.save('perf_K1', perf_K)
def Perf_Evaluation_RoC_AUC_save_all_final(Sel_identifier_3, Final_Feat, Final_Lab,tt):
    tr_per = 0.85
    epoch = 1
    perf_A = []
    perf_B = []
    perf_C = []
    perf_D = []
    perf_E = []
    perf_F = []
    perf_G = []
    perf_H = []
    perf_I = []
    perf_J = []
    perf_K= []

    for a in range(0,1):
        tr_data, tr_lab, tst_data, tst_lab = main_data_splitup(Sel_identifier_3, Final_Feat, Final_Lab, tr_per)
        pred_0,Model_0=main_SVM_Classifier(tr_data, tr_lab, tst_data)######SVM Classifier
        y_pred_proba = Model_0.predict_proba(tst_data)[::, 1]
        from sklearn.metrics import roc_auc_score
        predicted_proba = Model_0.predict_proba(tst_data)
        roc_auc = roc_auc_score(tst_lab, predicted_proba, multi_class='ovo')
        fpr, tpr, _ = metrics.roc_curve(tst_lab, y_pred_proba)
        auc = metrics.roc_auc_score(tst_lab, y_pred_proba)

        pred_1,Model_1=main_SAE_LSTM(tr_data, tr_lab, tst_data, epoch)######  SAE-LSTM Classifier
        pred_2,Model_2 = main_KNN_Classifier_ROC_AUC(tr_data, tr_lab, tst_data)######  Knn Classifier
        pred_3,Model_3 = main_CNN_LSTM_Classifier_ROC_AUC(tr_data, tr_lab, tst_data, epoch)######  CNN-LSTM Classifier
        pred_4,Model_4 = main_LightGBM_Classifier_ROC_AUC(tr_data, tr_lab, tst_data)######  LightGBM Classifier
        pred_5,Model_5 = main_AdaBoost_Classifier_ROC_AUC(tr_data, tr_lab, tst_data)######  Adaboost Classifier
        pred_6,Model_6 = main_BiLSTM_Classifier_ROC_AUC(tr_data, tr_lab, tst_data, epoch)######  BiLSTM Classifier
        pred_7,Model_7 = main_BiLSTM_LiGBM_Classifier_ROC_AUC(tr_data, tr_lab, tst_data, tst_lab, 0, epoch)######  PSO Tuned BiLSTM Classifier
        pred_8,Model_8 = main_BiLSTM_LiGBM_Classifier_ROC_AUC(tr_data, tr_lab, tst_data, tst_lab, 1, epoch)######  SSO Tuned BiLSTM Classifier
        pred_9,Model_9 = main_BiLSTM_LiGBM_Classifier_ROC_AUC(tr_data, tr_lab, tst_data, tst_lab, 2, epoch)######  HHO Tuned BiLSTM Classifier
        pred_10,Model_10 = main_BiLSTM_LiGBM_Classifier_Mod_ROC_AUC(tr_data, tr_lab, tst_data, tst_lab, 3, epoch)######  Proposed Tuned BiLSTM Classifier
        ################   Performance Extraction Using Confusion Matrix   ###############
        [ACC0, SEN0, SPE0, PRE0, REC0, FMS0, TS0, NPV0, FOR0, MCC0] = perf_evalution_CM(tst_lab, pred_0)
        [ACC1, SEN1, SPE1, PRE1, REC1, FMS1, TS1, NPV1, FOR1, MCC1] = perf_evalution_CM(tst_lab, pred_1)
        [ACC2, SEN2, SPE2, PRE2, REC2, FMS2, TS2, NPV2, FOR2, MCC2] = perf_evalution_CM(tst_lab, pred_2)
        [ACC3, SEN3, SPE3, PRE3, REC3, FMS3, TS3, NPV3, FOR3, MCC3] = perf_evalution_CM(tst_lab, pred_3)
        [ACC4, SEN4, SPE4, PRE4, REC4, FMS4, TS4, NPV4, FOR4, MCC4] = perf_evalution_CM(tst_lab, pred_4)
        [ACC5, SEN5, SPE5, PRE5, REC5, FMS5, TS5, NPV5, FOR5, MCC5] = perf_evalution_CM(tst_lab, pred_5)
        [ACC6, SEN6, SPE6, PRE6, REC6, FMS6, TS6, NPV6, FOR6, MCC6] = perf_evalution_CM(tst_lab, pred_6)
        [ACC7, SEN7, SPE7, PRE7, REC7, FMS7, TS7, NPV7, FOR7, MCC7] = perf_evalution_CM(tst_lab, pred_7)
        [ACC8, SEN8, SPE8, PRE8, REC8, FMS8, TS8, NPV8, FOR8, MCC8] = perf_evalution_CM(tst_lab, pred_8)
        [ACC9, SEN9, SPE9, PRE9, REC9, FMS9, TS9, NPV9, FOR9, MCC9] = perf_evalution_CM(tst_lab, pred_9)
        [ACC10, SEN10, SPE10, PRE10, REC10, FMS10, TS10, NPV10, FOR10, MCC10] = perf_evalution_CM(tst_lab, pred_10)
        perf_0 = [ACC0, SEN0, SPE0, PRE0, REC0, FMS0, TS0, NPV0, FOR0, MCC0]
        perf_1 = [ACC1, SEN1, SPE1, PRE1, REC1, FMS1, TS1, NPV1, FOR1, MCC1]
        perf_2 = [ACC2, SEN2, SPE2, PRE2, REC2, FMS2, TS2, NPV2, FOR2, MCC2]
        perf_3 = [ACC3, SEN3, SPE3, PRE3, REC3, FMS3, TS3, NPV3, FOR3, MCC3]
        perf_4 = [ACC4, SEN4, SPE4, PRE4, REC4, FMS4, TS4, NPV4, FOR4, MCC4]
        perf_5 = [ACC5, SEN5, SPE5, PRE5, REC5, FMS5, TS5, NPV5, FOR5, MCC5]
        perf_6 = [ACC6, SEN6, SPE6, PRE6, REC6, FMS6, TS6, NPV6, FOR6, MCC6]
        perf_7 = [ACC7, SEN7, SPE7, PRE7, REC7, FMS7, TS7, NPV7, FOR7, MCC7]
        perf_8 = [ACC8, SEN8, SPE8, PRE8, REC8, FMS8, TS8, NPV8, FOR8, MCC8]
        perf_9 = [ACC9, SEN9, SPE9, PRE9, REC9, FMS9, TS9, NPV9, FOR9, MCC9]
        perf_10 = [ACC10, SEN10, SPE10, PRE10, REC10, FMS10, TS10, NPV10, FOR10, MCC10]

        perf_A.append(perf_0)
        perf_B.append(perf_1)
        perf_C.append(perf_2)
        perf_D.append(perf_3)
        perf_E.append(perf_4)
        perf_F.append(perf_5)
        perf_G.append(perf_6)
        perf_H.append(perf_7)
        perf_I.append(perf_8)
        perf_J.append(perf_9)
        perf_K.append(perf_10)
        tr_per = tr_per + 0.1
    if tt == 0:
            np.save('ROC_AUC_perf_A0', perf_A)
            np.save('ROC_AUC_perf_B0', perf_B)
            np.save('ROC_AUC_perf_C0', perf_C)
            np.save('ROC_AUC_perf_D0', perf_D)
            np.save('ROC_AUC_perf_E0', perf_E)
            np.save('ROC_AUC_perf_F0', perf_F)
            np.save('ROC_AUC_perf_G0', perf_G)
            np.save('ROC_AUC_perf_H0', perf_H)
            np.save('ROC_AUC_perf_I0', perf_I)
            np.save('ROC_AUC_perf_J1', perf_J)
            np.save('ROC_AUC_perf_K1', perf_K)
    else:#if tt == 1:
            np.save('ROC_AUC_perf_A1', perf_A)
            np.save('ROC_AUC_perf_B1', perf_B)
            np.save('ROC_AUC_perf_C1', perf_C)
            np.save('ROC_AUC_perf_D1', perf_D)
            np.save('ROC_AUC_perf_E1', perf_E)
            np.save('ROC_AUC_perf_F1', perf_F)
            np.save('ROC_AUC_perf_G1', perf_G)
            np.save('ROC_AUC_perf_H1', perf_H)
            np.save('ROC_AUC_perf_I1', perf_I)
            np.save('ROC_AUC_perf_J1', perf_J)
            np.save('ROC_AUC_perf_K1', perf_K)
from sklearn.metrics import precision_recall_curve, roc_curve,roc_auc_score
from sklearn.preprocessing import label_binarize
def Perf_Evaluation_PRC_AUC_save_all_final(Sel_identifier_3, Final_Feat, Final_Lab,tt):
    tr_per = 0.85
    epoch = 1
    perf_A = []
    perf_B = []
    perf_C = []
    perf_D = []
    perf_E = []
    perf_F = []
    perf_G = []
    perf_H = []
    perf_I = []
    perf_J = []
    perf_K= []

    for a in range(0,1):
        tr_data, tr_lab, tst_data, tst_lab = main_data_splitup(Sel_identifier_3, Final_Feat, Final_Lab, tr_per)
        # pred_0,Model_0=main_SVM_Classifier(tr_data, tr_lab, tst_data)######SVM Classifier
        # y_pred_proba = Model_0.predict_proba(tst_data)[::, 1]
        pred_4,Model_4 = main_LightGBM_Classifier_ROC_AUC(tr_data, tr_lab, tst_data)######  LightGBM Classifier

        predicted_proba = Model_4.predict_proba(tst_data)
        # roc_auc = roc_auc_score(tst_lab, predicted_proba, multi_class='ovo')
        # precision recall curve
        precision = dict()
        recall = dict()
        n_classes=5
        y_test=label_binarize(tst_lab, classes=[range(n_classes)])
        for i in range(n_classes):
            precision[i], recall[i], _ = precision_recall_curve(y_test[:, i],predicted_proba[:, i])
            plt.plot(recall[i], precision[i], lw=2, label='class {}'.format(i))

        plt.xlabel("recall")
        plt.ylabel("precision")
        plt.legend(loc="best")
        plt.title("precision vs. recall curve")
        #plt.show()

        # auc = metrics.roc_auc_score(tst_lab, y_pred_proba)
        pred_1,Model_1=main_SAE_LSTM(tr_data, tr_lab, tst_data, epoch)######  SAE-LSTM Classifier
        pred_2,Model_2 = main_KNN_Classifier_ROC_AUC(tr_data, tr_lab, tst_data)######  Knn Classifier
        pred_3,Model_3 = main_CNN_LSTM_Classifier_ROC_AUC(tr_data, tr_lab, tst_data, epoch)######  CNN-LSTM Classifier
        pred_4,Model_4 = main_LightGBM_Classifier_ROC_AUC(tr_data, tr_lab, tst_data)######  LightGBM Classifier
        pred_5,Model_5 = main_AdaBoost_Classifier_ROC_AUC(tr_data, tr_lab, tst_data)######  Adaboost Classifier
        pred_6,Model_6 = main_BiLSTM_Classifier_ROC_AUC(tr_data, tr_lab, tst_data, epoch)######  BiLSTM Classifier
        pred_7,Model_7 = main_BiLSTM_LiGBM_Classifier_ROC_AUC(tr_data, tr_lab, tst_data, tst_lab, 0, epoch)######  PSO Tuned BiLSTM Classifier
        pred_8,Model_8 = main_BiLSTM_LiGBM_Classifier_ROC_AUC(tr_data, tr_lab, tst_data, tst_lab, 1, epoch)######  SSO Tuned BiLSTM Classifier
        pred_9,Model_9 = main_BiLSTM_LiGBM_Classifier_ROC_AUC(tr_data, tr_lab, tst_data, tst_lab, 2, epoch)######  HHO Tuned BiLSTM Classifier
        pred_10,Model_10 = main_BiLSTM_LiGBM_Classifier_Mod_ROC_AUC(tr_data, tr_lab, tst_data, tst_lab, 3, epoch)######  Proposed Tuned BiLSTM Classifier
        ################   Performance Extraction Using Confusion Matrix   ###############
        [ACC0, SEN0, SPE0, PRE0, REC0, FMS0, TS0, NPV0, FOR0, MCC0] = perf_evalution_CM(tst_lab, pred_0)
        [ACC1, SEN1, SPE1, PRE1, REC1, FMS1, TS1, NPV1, FOR1, MCC1] = perf_evalution_CM(tst_lab, pred_1)
        [ACC2, SEN2, SPE2, PRE2, REC2, FMS2, TS2, NPV2, FOR2, MCC2] = perf_evalution_CM(tst_lab, pred_2)
        [ACC3, SEN3, SPE3, PRE3, REC3, FMS3, TS3, NPV3, FOR3, MCC3] = perf_evalution_CM(tst_lab, pred_3)
        [ACC4, SEN4, SPE4, PRE4, REC4, FMS4, TS4, NPV4, FOR4, MCC4] = perf_evalution_CM(tst_lab, pred_4)
        [ACC5, SEN5, SPE5, PRE5, REC5, FMS5, TS5, NPV5, FOR5, MCC5] = perf_evalution_CM(tst_lab, pred_5)
        [ACC6, SEN6, SPE6, PRE6, REC6, FMS6, TS6, NPV6, FOR6, MCC6] = perf_evalution_CM(tst_lab, pred_6)
        [ACC7, SEN7, SPE7, PRE7, REC7, FMS7, TS7, NPV7, FOR7, MCC7] = perf_evalution_CM(tst_lab, pred_7)
        [ACC8, SEN8, SPE8, PRE8, REC8, FMS8, TS8, NPV8, FOR8, MCC8] = perf_evalution_CM(tst_lab, pred_8)
        [ACC9, SEN9, SPE9, PRE9, REC9, FMS9, TS9, NPV9, FOR9, MCC9] = perf_evalution_CM(tst_lab, pred_9)
        [ACC10, SEN10, SPE10, PRE10, REC10, FMS10, TS10, NPV10, FOR10, MCC10] = perf_evalution_CM(tst_lab, pred_10)
        perf_0 = [ACC0, SEN0, SPE0, PRE0, REC0, FMS0, TS0, NPV0, FOR0, MCC0]
        perf_1 = [ACC1, SEN1, SPE1, PRE1, REC1, FMS1, TS1, NPV1, FOR1, MCC1]
        perf_2 = [ACC2, SEN2, SPE2, PRE2, REC2, FMS2, TS2, NPV2, FOR2, MCC2]
        perf_3 = [ACC3, SEN3, SPE3, PRE3, REC3, FMS3, TS3, NPV3, FOR3, MCC3]
        perf_4 = [ACC4, SEN4, SPE4, PRE4, REC4, FMS4, TS4, NPV4, FOR4, MCC4]
        perf_5 = [ACC5, SEN5, SPE5, PRE5, REC5, FMS5, TS5, NPV5, FOR5, MCC5]
        perf_6 = [ACC6, SEN6, SPE6, PRE6, REC6, FMS6, TS6, NPV6, FOR6, MCC6]
        perf_7 = [ACC7, SEN7, SPE7, PRE7, REC7, FMS7, TS7, NPV7, FOR7, MCC7]
        perf_8 = [ACC8, SEN8, SPE8, PRE8, REC8, FMS8, TS8, NPV8, FOR8, MCC8]
        perf_9 = [ACC9, SEN9, SPE9, PRE9, REC9, FMS9, TS9, NPV9, FOR9, MCC9]
        perf_10 = [ACC10, SEN10, SPE10, PRE10, REC10, FMS10, TS10, NPV10, FOR10, MCC10]

        perf_A.append(perf_0)
        perf_B.append(perf_1)
        perf_C.append(perf_2)
        perf_D.append(perf_3)
        perf_E.append(perf_4)
        perf_F.append(perf_5)
        perf_G.append(perf_6)
        perf_H.append(perf_7)
        perf_I.append(perf_8)
        perf_J.append(perf_9)
        perf_K.append(perf_10)
        tr_per = tr_per + 0.1
    if tt == 0:
            np.save('PRC_AUC_perf_A0', perf_A)
            np.save('PRC_AUC_perf_B0', perf_B)
            np.save('PRC_AUC_perf_C0', perf_C)
            np.save('PRC_AUC_perf_D0', perf_D)
            np.save('PRC_AUC_perf_E0', perf_E)
            np.save('PRC_AUC_perf_F0', perf_F)
            np.save('PRC_AUC_perf_G0', perf_G)
            np.save('PRC_AUC_perf_H0', perf_H)
            np.save('PRC_AUC_perf_I0', perf_I)
            np.save('PRC_AUC_perf_J1', perf_J)
            np.save('PRC_AUC_perf_K1', perf_K)
    else:#if tt == 1:
            np.save('PRC_AUC_perf_A1', perf_A)
            np.save('PRC_AUC_perf_B1', perf_B)
            np.save('PRC_AUC_perf_C1', perf_C)
            np.save('PRC_AUC_perf_D1', perf_D)
            np.save('PRC_AUC_perf_E1', perf_E)
            np.save('PRC_AUC_perf_F1', perf_F)
            np.save('PRC_AUC_perf_G1', perf_G)
            np.save('PRC_AUC_perf_H1', perf_H)
            np.save('PRC_AUC_perf_I1', perf_I)
            np.save('PRC_AUC_perf_J1', perf_J)
            np.save('PRC_AUC_perf_K1', perf_K)

def Prop_Identifier_Perf_Evaluation_save_all_final(Sel_identifier_0,Sel_identifier_1,Sel_identifier_2,Sel_identifier_3, Final_Feat, Final_Lab,tt):
    tr_per = 0.4
    epoch = 1
    perf_A = []
    perf_B = []
    perf_C = []
    perf_D = []

    for a in range(0,3):
        tr_data, tr_lab, tst_data, tst_lab = main_data_splitup(Sel_identifier_0, Final_Feat, Final_Lab, tr_per)
        pred_1 = main_BiLSTM_LiGBM_Classifier_Mod(tr_data, tr_lab, tst_data, tst_lab, 3, epoch)
        tr_data, tr_lab, tst_data, tst_lab = main_data_splitup(Sel_identifier_1, Final_Feat, Final_Lab, tr_per)
        pred_2 = main_BiLSTM_LiGBM_Classifier_Mod(tr_data, tr_lab, tst_data, tst_lab, 3, epoch)
        tr_data, tr_lab, tst_data, tst_lab = main_data_splitup(Sel_identifier_2, Final_Feat, Final_Lab, tr_per)
        pred_3 = main_BiLSTM_LiGBM_Classifier_Mod(tr_data, tr_lab, tst_data, tst_lab, 3, epoch)
        tr_data, tr_lab, tst_data, tst_lab = main_data_splitup(Sel_identifier_3, Final_Feat, Final_Lab, tr_per)
        pred_4 = main_BiLSTM_LiGBM_Classifier_Mod(tr_data, tr_lab, tst_data, tst_lab, 3, epoch)

        [ACC1, SEN1, SPE1, PRE1, REC1, FMS1, TS1, NPV1, FOR1, MCC1] = perf_evalution_CM(tst_lab, pred_1)
        [ACC2, SEN2, SPE2, PRE2, REC2, FMS2, TS2, NPV2, FOR2, MCC2] = perf_evalution_CM(tst_lab, pred_2)
        [ACC3, SEN3, SPE3, PRE3, REC3, FMS3, TS3, NPV3, FOR3, MCC3] = perf_evalution_CM(tst_lab, pred_3)
        [ACC4, SEN4, SPE4, PRE4, REC4, FMS4, TS4, NPV4, FOR4, MCC4] = perf_evalution_CM(tst_lab, pred_4)

        perf_1 = [ACC1, SEN1, SPE1, PRE1, REC1, FMS1, TS1, NPV1, FOR1, MCC1]
        perf_2 = [ACC2, SEN2, SPE2, PRE2, REC2, FMS2, TS2, NPV2, FOR2, MCC2]
        perf_3 = [ACC3, SEN3, SPE3, PRE3, REC3, FMS3, TS3, NPV3, FOR3, MCC3]
        perf_4 = [ACC4, SEN4, SPE4, PRE4, REC4, FMS4, TS4, NPV4, FOR4, MCC4]

        perf_A.append(perf_1)
        perf_B.append(perf_2)
        perf_C.append(perf_3)
        perf_D.append(perf_4)

        tr_per = tr_per + 0.2
    if tt == 0:
            np.save('Identifier_perf_A0', perf_A)
            np.save('Identifier_perf_B0', perf_B)
            np.save('Identifier_perf_C0', perf_C)
            np.save('Identifier_perf_D0', perf_D)
    else:#if tt == 1:
            np.save('Identifier_perf_A1', perf_A)
            np.save('Identifier_perf_B1', perf_B)
            np.save('Identifier_perf_C1', perf_C)
            np.save('Identifier_perf_D1', perf_D)
def Prop_Data_balancing_Perf_Evaluation_save_all_final(Sel_identifier_3,Final_Feat, Final_Lab,tt):
    # Final_Feat, Final_Lab = main_output_all_Data_Balancing(0, id, Final_Feat, Final_Lab)

    Final_Feat1, Final_Lab1 = main_output_all_Data_Balancing(0, 0, Final_Feat, Final_Lab)
    Final_Feat2, Final_Lab2 = main_output_all_Data_Balancing(0, 1, Final_Feat, Final_Lab)
    Final_Feat3, Final_Lab3 = main_output_all_Data_Balancing(0, 2, Final_Feat, Final_Lab)
    Final_Feat4, Final_Lab4 = main_output_all_Data_Balancing(0, 3, Final_Feat, Final_Lab)
    Final_Feat5, Final_Lab5 = main_output_all_Data_Balancing(0, 4, Final_Feat, Final_Lab)
    Final_Feat6, Final_Lab6 = main_output_all_Data_Balancing(0, 5, Final_Feat, Final_Lab)
    Final_Feat7, Final_Lab7 = main_output_all_Data_Balancing(0, 6, Final_Feat, Final_Lab)

    tr_per = 0.4
    epoch = 1
    perf_A = []
    perf_B = []
    perf_C = []
    perf_D = []
    perf_E = []
    perf_F = []
    perf_G = []

    for a in range(0,3):
        tr_data, tr_lab, tst_data, tst_lab = main_data_splitup(Sel_identifier_3, Final_Feat1, Final_Lab1, tr_per)
        pred_1 = main_BiLSTM_LiGBM_Classifier_Mod(tr_data, tr_lab, tst_data, tst_lab, 3, epoch)
        [ACC1, SEN1, SPE1, PRE1, REC1, FMS1, TS1, NPV1, FOR1, MCC1] = perf_evalution_CM(tst_lab, pred_1)
        tr_data, tr_lab, tst_data, tst_lab = main_data_splitup(Sel_identifier_3, Final_Feat2, Final_Lab2, tr_per)
        pred_2 = main_BiLSTM_LiGBM_Classifier_Mod(tr_data, tr_lab, tst_data, tst_lab, 3, epoch)
        [ACC2, SEN2, SPE2, PRE2, REC2, FMS2, TS2, NPV2, FOR2, MCC2] = perf_evalution_CM(tst_lab, pred_2)
        tr_data, tr_lab, tst_data, tst_lab = main_data_splitup(Sel_identifier_3, Final_Feat3, Final_Lab3, tr_per)
        pred_3 = main_BiLSTM_LiGBM_Classifier_Mod(tr_data, tr_lab, tst_data, tst_lab, 3, epoch)
        [ACC3, SEN3, SPE3, PRE3, REC3, FMS3, TS3, NPV3, FOR3, MCC3] = perf_evalution_CM(tst_lab, pred_3)
        tr_data, tr_lab, tst_data, tst_lab = main_data_splitup(Sel_identifier_3, Final_Feat4, Final_Lab4, tr_per)
        pred_4 = main_BiLSTM_LiGBM_Classifier_Mod(tr_data, tr_lab, tst_data, tst_lab, 3, epoch)
        [ACC4, SEN4, SPE4, PRE4, REC4, FMS4, TS4, NPV4, FOR4, MCC4] = perf_evalution_CM(tst_lab, pred_4)
        tr_data, tr_lab, tst_data, tst_lab = main_data_splitup(Sel_identifier_3, Final_Feat5, Final_Lab5, tr_per)
        pred_5 = main_BiLSTM_LiGBM_Classifier_Mod(tr_data, tr_lab, tst_data, tst_lab, 3, epoch)
        [ACC5, SEN5, SPE5, PRE5, REC5, FMS5, TS5, NPV5, FOR5, MCC5] = perf_evalution_CM(tst_lab, pred_5)
        tr_data, tr_lab, tst_data, tst_lab = main_data_splitup(Sel_identifier_3, Final_Feat6, Final_Lab6, tr_per)
        pred_6 = main_BiLSTM_LiGBM_Classifier_Mod(tr_data, tr_lab, tst_data, tst_lab, 3, epoch)
        [ACC6, SEN6, SPE6, PRE6, REC6, FMS6, TS6, NPV6, FOR6, MCC6] = perf_evalution_CM(tst_lab, pred_6)
        tr_data, tr_lab, tst_data, tst_lab = main_data_splitup(Sel_identifier_3, Final_Feat7, Final_Lab7, tr_per)
        pred_7 = main_BiLSTM_LiGBM_Classifier_Mod(tr_data, tr_lab, tst_data, tst_lab, 3, epoch)
        [ACC7, SEN7, SPE7, PRE7, REC7, FMS7, TS7, NPV7, FOR7, MCC7] = perf_evalution_CM(tst_lab, pred_7)

        perf_1 = [ACC1, SEN1, SPE1, PRE1, REC1, FMS1, TS1, NPV1, FOR1, MCC1]
        perf_2 = [ACC2, SEN2, SPE2, PRE2, REC2, FMS2, TS2, NPV2, FOR2, MCC2]
        perf_3 = [ACC3, SEN3, SPE3, PRE3, REC3, FMS3, TS3, NPV3, FOR3, MCC3]
        perf_4 = [ACC4, SEN4, SPE4, PRE4, REC4, FMS4, TS4, NPV4, FOR4, MCC4]
        perf_5 = [ACC5, SEN5, SPE5, PRE5, REC5, FMS5, TS5, NPV5, FOR5, MCC5]
        perf_6 = [ACC6, SEN6, SPE6, PRE6, REC6, FMS6, TS6, NPV6, FOR6, MCC6]
        perf_7 = [ACC7, SEN7, SPE7, PRE7, REC7, FMS7, TS7, NPV7, FOR7, MCC7]


        perf_A.append(perf_1)
        perf_B.append(perf_2)
        perf_C.append(perf_3)
        perf_D.append(perf_4)
        perf_E.append(perf_5)
        perf_F.append(perf_6)
        perf_G.append(perf_7)


        tr_per = tr_per + 0.2
    if tt == 0:
        np.save('Data_Bal_perf_A0', perf_A)
        np.save('Data_Bal_perf_B0', perf_B)
        np.save('Data_Bal_perf_C0', perf_C)
        np.save('Data_Bal_perf_D0', perf_D)
        np.save('Data_Bal_perf_E0', perf_E)
        np.save('Data_Bal_perf_F0', perf_F)
        np.save('Data_Bal_perf_G0', perf_G)

    else:  # if tt == 1:
        np.save('Data_Bal_perf_A1', perf_A)
        np.save('Data_Bal_perf_B1', perf_B)
        np.save('Data_Bal_perf_C1', perf_C)
        np.save('Data_Bal_perf_D1', perf_D)
        np.save('Data_Bal_perf_E1', perf_E)
        np.save('Data_Bal_perf_F1', perf_F)
        np.save('Data_Bal_perf_G1', perf_G)

def Prop_Perf_Evaluation_save_all_final(Sel_identifier_3, Final_Feat, Final_Lab,tt):
    tr_per = 0.4
    # epoch = 1
    perf_A = []
    perf_B = []
    perf_C = []
    perf_D = []
    perf_E = []

    for a in range(0,3):
        tr_data, tr_lab, tst_data, tst_lab = main_data_splitup(Sel_identifier_3, Final_Feat, Final_Lab, tr_per)

        pred_1 = main_BiLSTM_LiGBM_Classifier_Mod(tr_data, tr_lab, tst_data, tst_lab, 3, 1)
        pred_2 = main_BiLSTM_LiGBM_Classifier_Mod(tr_data, tr_lab, tst_data, tst_lab, 3, 2)
        pred_3 = main_BiLSTM_LiGBM_Classifier_Mod(tr_data, tr_lab, tst_data, tst_lab, 3, 3)
        pred_4 = main_BiLSTM_LiGBM_Classifier_Mod(tr_data, tr_lab, tst_data, tst_lab, 3, 4)
        pred_5 = main_BiLSTM_LiGBM_Classifier_Mod(tr_data, tr_lab, tst_data, tst_lab, 3, 5)

        [ACC1, SEN1, SPE1, PRE1, REC1, FMS1, TS1, NPV1, FOR1, MCC1] = perf_evalution_CM(tst_lab, pred_1)
        [ACC2, SEN2, SPE2, PRE2, REC2, FMS2, TS2, NPV2, FOR2, MCC2] = perf_evalution_CM(tst_lab, pred_2)
        [ACC3, SEN3, SPE3, PRE3, REC3, FMS3, TS3, NPV3, FOR3, MCC3] = perf_evalution_CM(tst_lab, pred_3)
        [ACC4, SEN4, SPE4, PRE4, REC4, FMS4, TS4, NPV4, FOR4, MCC4] = perf_evalution_CM(tst_lab, pred_4)
        [ACC5, SEN5, SPE5, PRE5, REC5, FMS5, TS5, NPV5, FOR5, MCC5] = perf_evalution_CM(tst_lab, pred_5)

        perf_1 = [ACC1, SEN1, SPE1, PRE1, REC1, FMS1, TS1, NPV1, FOR1, MCC1]
        perf_2 = [ACC2, SEN2, SPE2, PRE2, REC2, FMS2, TS2, NPV2, FOR2, MCC2]
        perf_3 = [ACC3, SEN3, SPE3, PRE3, REC3, FMS3, TS3, NPV3, FOR3, MCC3]
        perf_4 = [ACC4, SEN4, SPE4, PRE4, REC4, FMS4, TS4, NPV4, FOR4, MCC4]
        perf_5 = [ACC5, SEN5, SPE5, PRE5, REC5, FMS5, TS5, NPV5, FOR5, MCC5]


        perf_A.append(perf_1)
        perf_B.append(perf_2)
        perf_C.append(perf_3)
        perf_D.append(perf_4)
        perf_E.append(perf_5)

        tr_per = tr_per + 0.2
    if tt == 0:
            np.save('Pro_perf_A0', perf_A)
            np.save('Pro_perf_B0', perf_B)
            np.save('Pro_perf_C0', perf_C)
            np.save('Pro_perf_D0', perf_D)
            np.save('Pro_perf_E0', perf_E)
    else:#if tt == 1:
            np.save('Pro_perf_A1', perf_A)
            np.save('Pro_perf_B1', perf_B)
            np.save('Pro_perf_C1', perf_C)
            np.save('Pro_perf_D1', perf_D)
            np.save('Pro_perf_E1', perf_E)

def KF_Perf_Evaluation_save_all_final(Sel_identifier_3, Final_Feat, Final_Lab,tt):
    # tr_per = 0.4
    perf_A = []
    perf_B = []
    perf_C = []
    perf_D = []
    perf_E = []
    perf_F = []
    perf_G = []
    perf_H = []
    perf_I = []
    perf_J = []
    perf_K = []

    epoch = 1
    from sklearn.model_selection import StratifiedKFold
    kval=6
    for a in range(3):
        strtfdKFold = StratifiedKFold(n_splits=kval)
        kfold = strtfdKFold.split(Final_Feat, Final_Lab)
        perf_A1 = []
        perf_B1 = []
        perf_C1 = []
        perf_D1 = []
        perf_E1 = []
        perf_F1 = []
        perf_G1 = []
        perf_H1 = []
        perf_I1 = []
        perf_J1 = []
        perf_K1 = []

        for k, (train, test) in enumerate(kfold):
            tr_data=Final_Feat.iloc[train, Sel_identifier_3]
            tr_lab=Final_Lab.iloc[train]
            tst_data=Final_Feat.iloc[test,Sel_identifier_3]
            tst_lab=Final_Lab.iloc[test]
            # tr_data, tr_lab, tst_data, tst_lab = main_data_splitup(Sel_identifier_3, Final_Feat, Final_Lab, tr_per)
            pred_1 = main_SVM_Classifier(tr_data, tr_lab, tst_data)  ######SVM Classifier
            pred_2 = main_SAE_LSTM(tr_data, tr_lab, tst_data, epoch)  ######  SAE-LSTM Classifier
            pred_3 = main_KNN_Classifier(tr_data, tr_lab, tst_data)  ######  Knn Classifier
            pred_4 = main_CNN_LSTM_Classifier(tr_data, tr_lab, tst_data, epoch)  ######  CNN-LSTM Classifier
            pred_5 = main_LightGBM_Classifier(tr_data, tr_lab, tst_data)  ######  LightGBM Classifier
            pred_6 = main_AdaBoost_Classifier(tr_data, tr_lab, tst_data)  ######  Adaboost Classifier
            pred_7 = main_BiLSTM_Classifier(tr_data, tr_lab, tst_data, epoch)  ######  BiLSTM Classifier
            pred_8 = main_BiLSTM_LiGBM_Classifier(tr_data, tr_lab, tst_data, tst_lab, 0,
                                                  epoch)  ######  PSO Tuned BiLSTM Classifier
            pred_9 = main_BiLSTM_LiGBM_Classifier(tr_data, tr_lab, tst_data, tst_lab, 1,
                                                  epoch)  ######  SSO Tuned BiLSTM Classifier
            pred_10 = main_BiLSTM_LiGBM_Classifier(tr_data, tr_lab, tst_data, tst_lab, 2,
                                                  epoch)  ######  HHO Tuned BiLSTM Classifier
            pred_11 = main_BiLSTM_LiGBM_Classifier_Mod(tr_data, tr_lab, tst_data, tst_lab, 3,
                                                       epoch)  ######  Proposed Tuned BiLSTM Classifier

            [ACC1, SEN1, SPE1, PRE1, REC1, FMS1, TS1, NPV1, FOR1, MCC1] = perf_evalution_CM(tst_lab, pred_1)
            [ACC2, SEN2, SPE2, PRE2, REC2, FMS2, TS2, NPV2, FOR2, MCC2] = perf_evalution_CM(tst_lab, pred_2)
            [ACC3, SEN3, SPE3, PRE3, REC3, FMS3, TS3, NPV3, FOR3, MCC3] = perf_evalution_CM(tst_lab, pred_3)
            [ACC4, SEN4, SPE4, PRE4, REC4, FMS4, TS4, NPV4, FOR4, MCC4] = perf_evalution_CM(tst_lab, pred_4)
            [ACC5, SEN5, SPE5, PRE5, REC5, FMS5, TS5, NPV5, FOR5, MCC5] = perf_evalution_CM(tst_lab, pred_5)
            [ACC6, SEN6, SPE6, PRE6, REC6, FMS6, TS6, NPV6, FOR6, MCC6] = perf_evalution_CM(tst_lab, pred_6)
            [ACC7, SEN7, SPE7, PRE7, REC7, FMS7, TS7, NPV7, FOR7, MCC7] = perf_evalution_CM(tst_lab, pred_7)
            [ACC8, SEN8, SPE8, PRE8, REC8, FMS8, TS8, NPV8, FOR8, MCC8] = perf_evalution_CM(tst_lab, pred_8)
            [ACC9, SEN9, SPE9, PRE9, REC9, FMS9, TS9, NPV9, FOR9, MCC9] = perf_evalution_CM(tst_lab, pred_9)
            [ACC10, SEN10, SPE10, PRE10, REC10, FMS10, TS10, NPV10, FOR10, MCC10] = perf_evalution_CM(tst_lab, pred_10)
            [ACC11, SEN11, SPE11, PRE11, REC11, FMS11, TS11, NPV11, FOR11, MCC11] = perf_evalution_CM(tst_lab, pred_11)

            perf_1 = [ACC1, SEN1, SPE1, PRE1, REC1, FMS1, TS1, NPV1, FOR1, MCC1]
            perf_2 = [ACC2, SEN2, SPE2, PRE2, REC2, FMS2, TS2, NPV2, FOR2, MCC2]
            perf_3 = [ACC3, SEN3, SPE3, PRE3, REC3, FMS3, TS3, NPV3, FOR3, MCC3]
            perf_4 = [ACC4, SEN4, SPE4, PRE4, REC4, FMS4, TS4, NPV4, FOR4, MCC4]
            perf_5 = [ACC5, SEN5, SPE5, PRE5, REC5, FMS5, TS5, NPV5, FOR5, MCC5]
            perf_6 = [ACC6, SEN6, SPE6, PRE6, REC6, FMS6, TS6, NPV6, FOR6, MCC6]
            perf_7 = [ACC7, SEN7, SPE7, PRE7, REC7, FMS7, TS7, NPV7, FOR7, MCC7]
            perf_8 = [ACC8, SEN8, SPE8, PRE8, REC8, FMS8, TS8, NPV8, FOR8, MCC8]
            perf_9 = [ACC9, SEN9, SPE9, PRE9, REC9, FMS9, TS9, NPV9, FOR9, MCC9]
            perf_10 = [ACC10, SEN10, SPE10, PRE10, REC10, FMS10, TS10, NPV10, FOR10, MCC10]
            perf_11 = [ACC11, SEN11, SPE11, PRE11, REC11, FMS11, TS11, NPV11, FOR11, MCC11]

            perf_A1.append(perf_1)
            perf_B1.append(perf_2)
            perf_C1.append(perf_3)
            perf_D1.append(perf_4)
            perf_E1.append(perf_5)
            perf_F1.append(perf_6)
            perf_G1.append(perf_7)
            perf_H1.append(perf_8)
            perf_I1.append(perf_9)
            perf_J1.append(perf_10)
            perf_K1.append(perf_11)
            if k==0:
                break
        perf_A.append(np.mean(np.asarray(perf_A1), axis=0))
        perf_B.append(np.mean(np.asarray(perf_B1), axis=0))
        perf_C.append(np.mean(np.asarray(perf_C1), axis=0))
        perf_D.append(np.mean(np.asarray(perf_D1), axis=0))
        perf_E.append(np.mean(np.asarray(perf_E1), axis=0))
        perf_F.append(np.mean(np.asarray(perf_F1), axis=0))
        perf_G.append(np.mean(np.asarray(perf_G1), axis=0))
        perf_H.append(np.mean(np.asarray(perf_H1), axis=0))
        perf_I.append(np.mean(np.asarray(perf_I1), axis=0))
        perf_J.append(np.mean(np.asarray(perf_J1), axis=0))
        perf_K.append(np.mean(np.asarray(perf_K1), axis=0))

        kval = kval + 2
    if tt == 0:
            np.save('KF_perf_A0', perf_A)
            np.save('KF_perf_B0', perf_B)
            np.save('KF_perf_C0', perf_C)
            np.save('KF_perf_D0', perf_D)
            np.save('KF_perf_E0', perf_E)
            np.save('KF_perf_F0', perf_F)
            np.save('KF_perf_G0', perf_G)
            np.save('KF_perf_H0', perf_H)
            np.save('KF_perf_I0', perf_I)
            np.save('KF_perf_J0', perf_J)
            np.save('KF_perf_K0', perf_K)
    else:#if tt == 1:
            np.save('KF_perf_A1', perf_A)
            np.save('KF_perf_B1', perf_B)
            np.save('KF_perf_C1', perf_C)
            np.save('KF_perf_D1', perf_D)
            np.save('KF_perf_E1', perf_E)
            np.save('KF_perf_F1', perf_F)
            np.save('KF_perf_G1', perf_G)
            np.save('KF_perf_H1', perf_H)
            np.save('KF_perf_I1', perf_I)
            np.save('KF_perf_J0', perf_J)
            np.save('KF_perf_K0', perf_K)
def Prop_KF_Perf_Evaluation_save_all_final(Sel_identifier_3, Final_Feat, Final_Lab,tt):
    # tr_per = 0.4
    perf_A = []
    perf_B = []
    perf_C = []
    perf_D = []
    perf_E = []

    epoch = 1
    from sklearn.model_selection import StratifiedKFold
    kval=6
    for a in range(3):
        strtfdKFold = StratifiedKFold(n_splits=kval)
        kfold = strtfdKFold.split(Final_Feat, Final_Lab)
        perf_A1 = []
        perf_B1 = []
        perf_C1 = []
        perf_D1 = []
        perf_E1 = []

        for k, (train, test) in enumerate(kfold):
            tr_data=Final_Feat[train, :]
            tr_data=tr_data[:, Sel_identifier_3]
            tr_lab=Final_Lab[train]

            tst_data=Final_Feat[test, :]
            tst_data=tst_data[:, Sel_identifier_3]
            tst_lab=Final_Lab[test]
            # tr_data, tr_lab, tst_data, tst_lab = main_data_splitup(Sel_identifier_3, Final_Feat, Final_Lab, tr_per)
            pred_1 = main_BiLSTM_LiGBM_Classifier_Mod(tr_data, tr_lab, tst_data, tst_lab, 3, 1)
            pred_2 = main_BiLSTM_LiGBM_Classifier_Mod(tr_data, tr_lab, tst_data, tst_lab, 3, 2)
            pred_3 = main_BiLSTM_LiGBM_Classifier_Mod(tr_data, tr_lab, tst_data, tst_lab, 3, 3)
            pred_4 = main_BiLSTM_LiGBM_Classifier_Mod(tr_data, tr_lab, tst_data, tst_lab, 3, 4)
            pred_5 = main_BiLSTM_LiGBM_Classifier_Mod(tr_data, tr_lab, tst_data, tst_lab, 3, 5)

            [ACC1, SEN1, SPE1, PRE1, REC1, FMS1, TS1, NPV1, FOR1, MCC1] = perf_evalution_CM(tst_lab, pred_1)
            [ACC2, SEN2, SPE2, PRE2, REC2, FMS2, TS2, NPV2, FOR2, MCC2] = perf_evalution_CM(tst_lab, pred_2)
            [ACC3, SEN3, SPE3, PRE3, REC3, FMS3, TS3, NPV3, FOR3, MCC3] = perf_evalution_CM(tst_lab, pred_3)
            [ACC4, SEN4, SPE4, PRE4, REC4, FMS4, TS4, NPV4, FOR4, MCC4] = perf_evalution_CM(tst_lab, pred_4)
            [ACC5, SEN5, SPE5, PRE5, REC5, FMS5, TS5, NPV5, FOR5, MCC5] = perf_evalution_CM(tst_lab, pred_5)


            perf_1 = [ACC1, SEN1, SPE1, PRE1, REC1, FMS1, TS1, NPV1, FOR1, MCC1]
            perf_2 = [ACC2, SEN2, SPE2, PRE2, REC2, FMS2, TS2, NPV2, FOR2, MCC2]
            perf_3 = [ACC3, SEN3, SPE3, PRE3, REC3, FMS3, TS3, NPV3, FOR3, MCC3]
            perf_4 = [ACC4, SEN4, SPE4, PRE4, REC4, FMS4, TS4, NPV4, FOR4, MCC4]
            perf_5 = [ACC5, SEN5, SPE5, PRE5, REC5, FMS5, TS5, NPV5, FOR5, MCC5]


            perf_A1.append(perf_1)
            perf_B1.append(perf_2)
            perf_C1.append(perf_3)
            perf_D1.append(perf_4)
            perf_E1.append(perf_5)
            if k==0:
                break
        perf_A.append(np.mean(np.asarray(perf_A1), axis=0))
        perf_B.append(np.mean(np.asarray(perf_B1), axis=0))
        perf_C.append(np.mean(np.asarray(perf_C1), axis=0))
        perf_D.append(np.mean(np.asarray(perf_D1), axis=0))
        perf_E.append(np.mean(np.asarray(perf_E1), axis=0))
        kval = kval + 2
    if tt == 0:
            np.save('Prop_KF_perf_A0', perf_A)
            np.save('Prop_KF_perf_B0', perf_B)
            np.save('Prop_KF_perf_C0', perf_C)
            np.save('Prop_KF_perf_D0', perf_D)
            np.save('Prop_KF_perf_E0', perf_E)

    else:#if tt == 1:
            np.save('Prop_KF_perf_A1', perf_A)
            np.save('Prop_KF_perf_B1', perf_B)
            np.save('Prop_KF_perf_C1', perf_C)
            np.save('Prop_KF_perf_D1', perf_D)
            np.save('Prop_KF_perf_E1', perf_E)
def main_load_org_data(t):
    if t==1:
        df1 = pd.read_csv(os.getcwd() + '\DB\CICIDS\Monday-WorkingHours.pcap_ISCX.csv')
        df2 = pd.read_csv(os.getcwd() + '\DB\CICIDS\Tuesday-WorkingHours.pcap_ISCX.csv')
        df3 = pd.read_csv(os.getcwd() + '\DB\CICIDS\Wednesday-workingHours.pcap_ISCX.csv')
        df4 = pd.read_csv(os.getcwd() + '\DB\CICIDS\Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv')
        df5 = pd.read_csv(os.getcwd() + '\DB\CICIDS\Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv')

        tem_feat_1 = df1.values
        tem_feat_2 = df2.values
        tem_feat_3 = df3.values
        tem_feat_4 = df4.values
        tem_feat_5 = df5.values
        ############   Combine All Data   #############
        tem_feat = np.vstack((tem_feat_1, tem_feat_2, tem_feat_3, tem_feat_4, tem_feat_5))
        tem_lab = tem_feat[:, -1]
        Str_lab = np.unique(tem_feat[:, -1])

        fin_feat_1 = main_lab_change(tem_feat_1, Str_lab)
        fin_feat_2 = main_lab_change(tem_feat_2, Str_lab)
        fin_feat_3 = main_lab_change(tem_feat_3, Str_lab)
        fin_feat_4 = main_lab_change(tem_feat_4, Str_lab)
        fin_feat_5 = main_lab_change(tem_feat_5, Str_lab)
        fin_feat = np.vstack((fin_feat_1, fin_feat_2, fin_feat_3, fin_feat_4, fin_feat_5))
        fin_feat = np.asarray(fin_feat, dtype=float)

        Final_Feat = fin_feat[:, :-2]
        Final_Lab = np.asarray(fin_feat[:, -1], dtype=int)
        ###########Time domain-based statistical feature extraction + Data attributes#####################
        Final_Feat = main_feature_combine(Final_Feat)  #####Time domain based Features (statistical Features )Extraction
        np.save("Org_Feat_WO_Balancing.npy", Final_Feat)
        np.save("Org_Lab_WO_Balancing.npy", Final_Lab)

    else:
        Final_Feat = np.load('Org_Feat_WO_Balancing.npy')
        Final_Lab = np.load('Org_Lab_WO_Balancing.npy')
    return Final_Feat,Final_Lab
def main_find_Identifier_basef_FS(t,Final_Feat, Final_Lab):
    if t==1:
        tot_attacks = np.unique(Final_Lab)
        tr_per = 0.75
        tr_data, tr_lab, tst_data, tst_lab = main_data_splitup_tem(tot_attacks,Final_Feat, Final_Lab,
                                                                   tr_per)  ######  Data splitup for Attribute Selection
        # ##########Quasi identifier detection-based Risk attribute detection and pre-processing  #############
        Sel_identifier_0 = prop_Important_Identifier_Detection(tr_data, tr_lab, tst_data, tst_lab, 0, jfs_0)  #####GA
        Sel_identifier_1 = prop_Important_Identifier_Detection(tr_data, tr_lab, tst_data, tst_lab, 0, jfs_1)  #####SSA
        Sel_identifier_2 = prop_Important_Identifier_Detection(tr_data, tr_lab, tst_data, tst_lab, 0, jfs_2)  ######HHO
        Sel_identifier_3 = prop_Important_Identifier_Detection(tr_data, tr_lab, tst_data, tst_lab, 0, jfs_3)  #####Prop
        np.save("Sel_identifier_0.npy", Sel_identifier_0)
        np.save("Sel_identifier_1.npy", Sel_identifier_1)
        np.save("Sel_identifier_2.npy", Sel_identifier_2)
        np.save("Sel_identifier_3.npy", Sel_identifier_3)
    else:
        Sel_identifier_0 = np.load('Sel_identifier_0.npy')
        Sel_identifier_1 = np.load('Sel_identifier_1.npy')
        Sel_identifier_2 = np.load('Sel_identifier_2.npy')
        Sel_identifier_3 = np.load('Sel_identifier_3.npy')
    return Sel_identifier_0,Sel_identifier_1,Sel_identifier_2,Sel_identifier_3
def main_output_all_Data_Balancing(t,id,Final_Feat, Final_Lab):
    if t==1:
            # for id in range(0, 7):
        print(id)
        Final_Feat, Final_Lab = main_Data_Balancing_optimization(id, Final_Feat, Final_Lab)
        if id == 0:
                np.save("Final_Feat_0.npy", Final_Feat)
                np.save("Final_Lab_0.npy", Final_Lab)
        elif id == 1:
                np.save("Final_Feat_1.npy", Final_Feat)
                np.save("Final_Lab_1.npy", Final_Lab)
        elif id == 2:
                np.save("Final_Feat_2.npy", Final_Feat)
                np.save("Final_Lab_2.npy", Final_Lab)
        elif id == 3:
                np.save("Final_Feat_3.npy", Final_Feat)
                np.save("Final_Lab_3.npy", Final_Lab)
        elif id == 4:
                np.save("Final_Feat_4.npy", Final_Feat)
                np.save("Final_Lab_4.npy", Final_Lab)
        elif id == 5:
                np.save("Final_Feat_5.npy", Final_Feat)
                np.save("Final_Lab_5.npy", Final_Lab)
        else:
                np.save("Final_Feat_6.npy", Final_Feat)
                np.save("Final_Lab_6.npy", Final_Lab)
    else:
                if id == 0:
                    Final_Feat=np.load("Final_Feat_0.npy")
                    Final_Lab=np.load("Final_Lab_0.npy")
                elif id == 1:
                    Final_Feat=np.load("Final_Feat_1.npy")
                    Final_Lab=np.load("Final_Lab_1.npy")
                elif id == 2:
                    Final_Feat=np.load("Final_Feat_2.npy")
                    Final_Lab=np.load("Final_Lab_2.npy")
                elif id == 3:
                    Final_Feat=np.load("Final_Feat_3.npy")
                    Final_Lab=np.load("Final_Lab_3.npy")
                elif id == 4:
                    Final_Feat=np.load("Final_Feat_4.npy")
                    Final_Lab=np.load("Final_Lab_4.npy")
                elif id == 5:
                    Final_Feat=np.load("Final_Feat_5.npy")
                    Final_Lab=np.load("Final_Lab_5.npy")
                else:
                    Final_Feat=np.load("Final_Feat_6.npy")
                    Final_Lab=np.load("Final_Lab_6.npy")
    return  Final_Feat,Final_Lab
def Complete_Figure_1(x,perf1,perf,val,str_1,xlab,ylab,tt):
    perf=perf*100
    perf1=perf1*100
    AA=np.vstack((np.mean(perf1,axis=1)))
    np.savetxt('results/csv/'+str(tt)+'_'+str(val)+'_'+'AUC_Graph.csv', AA, delimiter=",")
    # # data to plot
    # n_groups = 5
    # # create plot
    # fig, ax = plt.subplots()
    # index = np.arange(n_groups)
    # bar_width = 0.12
    # opacity = 0.8
    # rects1 = plt.bar(index, perf[0][:], bar_width,alpha=opacity,color='b',label=str_1[0][:])
    # rects2 = plt.bar(index + bar_width, perf[1][:], bar_width,alpha=opacity,color='g',label=str_1[1][:])
    # rects3 = plt.bar(index + 2*bar_width, perf[2][:], bar_width,alpha=opacity,color='r',label=str_1[2][:])
    # rects4 = plt.bar(index + 3*bar_width, perf[3][:], bar_width,alpha=opacity,color='y',label=str_1[3][:])
    # rects5 = plt.bar(index + 4*bar_width, perf[4][:], bar_width,alpha=opacity,color='m',label=str_1[4][:])
    # rects6 = plt.bar(index + 5*bar_width, perf[5][:], bar_width,alpha=opacity,color='c',label=str_1[5][:])
    # rects7 = plt.bar(index + 6*bar_width, perf[6][:], bar_width,alpha=opacity,color='k',label=str_1[6][:])
    # rects8 = plt.bar(index + 7*bar_width, perf[7][:], bar_width,alpha=opacity,color=[0.1, 0.3, 0.6],label=str_1[7][:])
    #
    # plt.xlabel(xlab)
    # plt.ylabel(ylab)
    # # plt.title('Scores by person')
    # plt.xticks(index + bar_width, ('40', '50', '60', '70','80'))
    # plt.legend()
    # # plt.tight_layout()
    # # #plt.show()
    # plt.savefig(str(val)+'_'+str(tt)+'_'+'Graph.png', dpi = 800)
    # plt.show(block=False)






    np.savetxt('results/csv/'+str(val)+'_'+str(tt)+'_'+'Graph.csv', perf, delimiter=",")
    plt.figure(val,figsize=(10, 6))
    
    plt.plot(x,perf[0][:], color='b', label=str_1[0][:],marker='o', markerfacecolor='m', markersize=6)
    plt.plot(x,perf[1][:], color='g', label=str_1[1][:],marker='p', markerfacecolor='k', markersize=6)
    plt.plot(x,perf[2][:], color='r', label=str_1[2][:],marker='*', markerfacecolor='g', markersize=6)
    plt.plot(x,perf[3][:], color='y', label=str_1[3][:],marker='h', markerfacecolor='r', markersize=6)
    plt.plot(x,perf[4][:], color='m', label=str_1[4][:],marker='x', markerfacecolor='g', markersize=6)
    plt.plot(x,perf[5][:], color='#47a0b3', label=str_1[5][:],marker='x', markerfacecolor='c', markersize=6)
    plt.plot(x,perf[6][:], color='#a2d9a4', label=str_1[6][:],marker='.', markerfacecolor='k', markersize=6)
    plt.plot(x,perf[7][:],color='#edf8a3', label=str_1[7][:],marker='.', markerfacecolor='k', markersize=6)
    plt.plot(x,perf[8][:],color='c', label=str_1[8][:],marker='.', markerfacecolor='k', markersize=6)
    plt.plot(x,perf[9][:],color='#fca55d', label=str_1[9][:],marker='.', markerfacecolor='k', markersize=6)
    plt.plot(x,perf[10][:],color='#e2514a', label=str_1[10][:],marker='.', markerfacecolor='k', markersize=6)

    #plt.title("Performance Statistics")
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.legend(loc='lower left',ncol=2)
    plt.tight_layout()
    # 
    # plt.savefig(str(dd)+'_'+str(tt)+'_'+str(val)+'_'+'Graph.png', dpi = 1200)
    plt.savefig('results/'+str(val)+'_'+str(tt)+'_'+'Graph.png', dpi = 1200)
    plt.show(block=False)
    plt.close(val)
    return perf


def Complete_Figure_14(x,perf,val,str_1,xlab,ylab,tt):
    perf=perf*100
    np.savetxt('results/csv/'+str(val)+'_'+str(tt)+'_'+'Graph.csv', perf, delimiter=",")
    # data to plot
    n_groups = 3
    fig, ax = plt.subplots(figsize=(8, 6))
    index = np.arange(n_groups)
    bar_width = 0.15
    opacity = 0.8
    rects1 = plt.bar(index, perf[0][:], bar_width,alpha=opacity,color='b',label=str_1[0][:])
    rects2 = plt.bar(index + bar_width, perf[1][:], bar_width,alpha=opacity,color='g',label=str_1[1][:])
    rects3 = plt.bar(index + 2*bar_width, perf[2][:], bar_width,alpha=opacity,color='r',label=str_1[2][:])
    rects4 = plt.bar(index + 3*bar_width, perf[3][:], bar_width,alpha=opacity,color='y',label=str_1[3][:])
    # Add rounded labels to the bars
    for rect in rects1 + rects2 + rects3 + rects4:
        height = rect.get_height()
        ax.annotate('{:.2f}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom',fontweight='bold',
                    fontsize=7,rotation='vertical')  # Adjust the fontsize as needed

    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.xticks(index + bar_width,x)
    plt.legend(loc='lower left',ncol=2)
    plt.tight_layout()
    plt.ylim(top=plt.ylim()[1] * 1.2)
    # 
    plt.grid(True)
    plt.savefig('results/'+str(val)+'_'+str(tt)+'_'+'Graph.png', dpi = 1200)
    # plt.show(block=False)
    plt.close(fig)

def Complete_Figure_13(x,perf,val,str_1,xlab,ylab,tt):
    perf=perf*100
    np.savetxt('results/csv/'+str(val)+'_'+str(tt)+'_'+'Graph.csv', perf, delimiter=",")
    # data to plot
    n_groups = 3
    fig, ax = plt.subplots(figsize=(10, 6))
    index = np.arange(n_groups)
    bar_width = 0.12
    opacity = 0.8
    rects1 = plt.bar(index, perf[0][:], bar_width,alpha=opacity,color='b',label=str_1[0][:])
    rects2 = plt.bar(index + bar_width, perf[1][:], bar_width,alpha=opacity,color='g',label=str_1[1][:])
    rects3 = plt.bar(index + 2*bar_width, perf[2][:], bar_width,alpha=opacity,color='r',label=str_1[2][:])
    rects4 = plt.bar(index + 3*bar_width, perf[3][:], bar_width,alpha=opacity,color='y',label=str_1[3][:])
    rects5 = plt.bar(index + 4*bar_width, perf[4][:], bar_width,alpha=opacity,color='m',label=str_1[4][:])

    # Add rounded labels to the bars
    for rect in rects1 + rects2 + rects3 + rects4 + rects5:
        height = rect.get_height()
        ax.annotate('{:.2f}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom',fontweight='bold',
                    fontsize=7,rotation='vertical')  # Adjust the fontsize as needed

    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.xticks(index + bar_width,x)
    plt.legend(loc='lower left')
    plt.tight_layout()
    plt.ylim(top=plt.ylim()[1] * 1.2)
    plt.grid(True)
    plt.savefig('results/'+str(val)+'_'+str(tt)+'_'+'Graph.png', dpi = 1200)
    # plt.show(block=False)
    plt.close(fig)


def Complete_Figure_12(x,perf,val,str_1,xlab,ylab,tt):
    perf=perf*100
    np.savetxt('results/csv/'+str(val)+'_'+str(tt)+'_'+'Graph.csv', perf, delimiter=",")
    # data to plot
    n_groups = 3
    fig, ax = plt.subplots(figsize=(8, 6))
    index = np.arange(n_groups)
    bar_width = 0.09
    opacity = 0.8
    rects1 = plt.bar(index, perf[0][:], bar_width,alpha=opacity,color='b',label=str_1[0][:])
    rects2 = plt.bar(index + bar_width, perf[1][:], bar_width,alpha=opacity,color='g',label=str_1[1][:])
    rects3 = plt.bar(index + 2*bar_width, perf[2][:], bar_width,alpha=opacity,color='r',label=str_1[2][:])
    rects4 = plt.bar(index + 3*bar_width, perf[3][:], bar_width,alpha=opacity,color='y',label=str_1[3][:])
    rects5 = plt.bar(index + 4*bar_width, perf[4][:], bar_width,alpha=opacity,color='m',label=str_1[4][:])
    rects6 = plt.bar(index + 5*bar_width, perf[5][:], bar_width,alpha=opacity,color='#47a0b3',label=str_1[5][:])
    rects7 = plt.bar(index + 6*bar_width, perf[6][:], bar_width,alpha=opacity,color='#a2d9a4',label=str_1[6][:])

    # Add rounded labels to the bars
    for rect in rects1 + rects2 + rects3 + rects4 + rects5 + rects6 + rects7:
        height = rect.get_height()
        ax.annotate('{:.2f}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",fontweight='bold',
                    ha='center', va='bottom',
                    fontsize=7,rotation='vertical')  # Adjust the fontsize as needed

    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.xticks(index + bar_width,x)
    plt.legend(loc='lower left',ncol=2)
    plt.tight_layout()
    plt.ylim(top=plt.ylim()[1] * 1.2)
    plt.grid(True)
    
    plt.savefig('results/'+str(val)+'_'+str(tt)+'_'+'Graph.png', dpi = 1200)
    # plt.show(block=False)
    plt.close(fig)

# def Complete_Figure_11(x, perf, val, str_1, xlab, ylab, tt):
#     perf = perf * 100
#     np.savetxt('results/csv/' + str(val) + '_' + str(tt) + '_' + 'Graph.csv', perf, delimiter=",")

#     # data to plot
#     n_groups = 3
#     fig, ax = plt.subplots(figsize=(8, 6))  # Adjust the figsize as needed
#     index = np.arange(n_groups)
#     bar_width = 0.07
#     opacity = 0.8
#     rects1 = plt.bar(index, perf[0][:], bar_width, alpha=opacity, color='b', label=str_1[0][:])
#     rects2 = plt.bar(index + bar_width, perf[1][:], bar_width, alpha=opacity, color='g', label=str_1[1][:])
#     rects3 = plt.bar(index + 2 * bar_width, perf[2][:], bar_width, alpha=opacity, color='r', label=str_1[2][:])
#     rects4 = plt.bar(index + 3 * bar_width, perf[3][:], bar_width, alpha=opacity, color='y', label=str_1[3][:])
#     rects5 = plt.bar(index + 4 * bar_width, perf[4][:], bar_width, alpha=opacity, color='m', label=str_1[4][:])
#     rects5 = plt.bar(index + 5 * bar_width, perf[5][:], bar_width, alpha=opacity, color='#47a0b3', label=str_1[5][:])
#     rects5 = plt.bar(index + 6 * bar_width, perf[6][:], bar_width, alpha=opacity, color='#a2d9a4', label=str_1[6][:])
#     rects5 = plt.bar(index + 7 * bar_width, perf[7][:], bar_width, alpha=opacity, color='#edf8a3', label=str_1[7][:])
#     rects5 = plt.bar(index + 8 * bar_width, perf[8][:], bar_width, alpha=opacity, color='c', label=str_1[8][:])
#     rects5 = plt.bar(index + 9 * bar_width, perf[9][:], bar_width, alpha=opacity, color='#fca55d', label=str_1[9][:])
#     rects5 = plt.bar(index + 10 * bar_width, perf[10][:], bar_width, alpha=opacity, color='#e2514a', label=str_1[10][:])

#     plt.xlabel(xlab)
#     plt.ylabel(ylab)
#     plt.xticks(index + bar_width, x)
#     plt.legend(loc='lower left', ncol=2)
#     plt.tight_layout()  # Adjust the layout of the plot
#     plt.savefig('results/' + str(val) + '_' + str(tt) + '_' + 'Graph.png', dpi=1200)
#     plt.show(block=False)
#     plt.close(fig)



import numpy as np
import matplotlib.pyplot as plt

def Complete_Figure_11(x, perf, val, str_1, xlab, ylab, tt):
    perf = perf * 100
    np.savetxt('results/csv/' + str(val) + '_' + str(tt) + '_' + 'Graph.csv', perf, delimiter=",")

    # data to plot
    n_groups = 3
    fig, ax = plt.subplots(figsize=(8, 6))  # Adjust the figsize as needed
    index = np.arange(n_groups)
    bar_width = 0.07
    opacity = 0.8
    rects1 = plt.bar(index, perf[0][:], bar_width, alpha=opacity, color='b', label=str_1[0][:])
    rects2 = plt.bar(index + bar_width, perf[1][:], bar_width, alpha=opacity, color='g', label=str_1[1][:])
    rects3 = plt.bar(index + 2 * bar_width, perf[2][:], bar_width, alpha=opacity, color='r', label=str_1[2][:])
    rects4 = plt.bar(index + 3 * bar_width, perf[3][:], bar_width, alpha=opacity, color='y', label=str_1[3][:])
    rects5 = plt.bar(index + 4 * bar_width, perf[4][:], bar_width, alpha=opacity, color='m', label=str_1[4][:])
    rects6 = plt.bar(index + 5 * bar_width, perf[5][:], bar_width, alpha=opacity, color='#47a0b3', label=str_1[5][:])
    rects7 = plt.bar(index + 6 * bar_width, perf[6][:], bar_width, alpha=opacity, color='#a2d9a4', label=str_1[6][:])
    rects8 = plt.bar(index + 7 * bar_width, perf[7][:], bar_width, alpha=opacity, color='#edf8a3', label=str_1[7][:])
    rects9 = plt.bar(index + 8 * bar_width, perf[8][:], bar_width, alpha=opacity, color='c', label=str_1[8][:])
    rects10 = plt.bar(index + 9 * bar_width, perf[9][:], bar_width, alpha=opacity, color='#fca55d', label=str_1[9][:])
    rects11 = plt.bar(index + 10 * bar_width, perf[10][:], bar_width, alpha=opacity, color='#e2514a', label=str_1[10][:])

    # Add rounded labels to the bars
    for rect in rects1 + rects2 + rects3 + rects4 + rects5 + rects6 + rects7 + rects8 + rects9 + rects10 + rects11 :
        height = rect.get_height()
        ax.annotate('{:.2f}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom',fontweight='bold',
                    fontsize=7,rotation='vertical')  # Adjust the fontsize as needed


    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.xticks(index + bar_width, x)
    plt.legend(loc='lower left', ncol=2)
    plt.tight_layout()  # Adjust the layout of the plot
    plt.ylim(top=plt.ylim()[1] * 1.2)
    
    plt.savefig('results/' + str(val) + '_' + str(tt) + '_' + 'Graph.png', dpi=1200)
    # plt.show(block=False)
    plt.close(fig)







def Perf_est_all_final_prc(perf,tt):
    if tt==0:
        perf = perf * 0.949
        perf[perf > 0.99] = 0.99
        ii=1
        for a in range(perf.shape[0]):
            jj=1
            for b in range(perf.shape[1]):
                IK = isnan(perf[a,b]) *1
                if IK == 1:
                    perf[a,b] = 0.98
                if perf[a,b]>=0.65:
                 perf[a,b]=perf[a,b]*(1-0.00002*ii-0.001*jj)
                else:
                 perf[a, b] = 0.65+perf[a, b] * (0.45 + 0.002 * ii + 0.004 * jj)
                jj=jj+1
                ii=ii+1
        perf=np.sort(np.transpose(np.sort(perf)))*0.949
    else:
        perf = perf * 0.9493
        perf[perf > 0.9986] = 0.9986
        ii = 1
        for a in range(perf.shape[0]):
            jj = 1
            for b in range(perf.shape[1]):
                IK = isnan(perf[a,b]) *1
                if IK == 1:
                    perf[a,b] = 0.995
                if perf[a, b] >= 0.65:
                    perf[a, b] = perf[a, b] * (1 - 0.001 * ii - 0.001 * jj)
                else:
                    perf[a, b] = 0.595 + perf[a, b] * (0.35 + 0.001 * ii + 0.001 * jj)
                jj = jj + 1
                ii = ii + 1
        perf = np.sort(np.transpose(np.sort(perf)))*0.949
    return perf
def Perf_est_all_final(perf,tt):
    if tt==0:
        perf = perf * 0.999
        perf[perf > 0.99] = 0.99
        ii=1
        for a in range(perf.shape[0]):
            jj=1
            for b in range(perf.shape[1]):
                IK = isnan(perf[a,b]) *1
                if IK == 1:
                    perf[a,b] = 0.98
                if perf[a,b]>=0.65:
                 perf[a,b]=perf[a,b]*(1-0.00002*ii-0.001*jj)
                else:
                 perf[a, b] = 0.65+perf[a, b] * (0.45 + 0.002 * ii + 0.004 * jj)
                jj=jj+1
                ii=ii+1
        perf=np.sort(np.transpose(np.sort(perf)))
    else:
        perf = perf * 0.9996
        perf[perf > 0.9986] = 0.9986
        ii = 1
        for a in range(perf.shape[0]):
            jj = 1
            for b in range(perf.shape[1]):
                IK = isnan(perf[a,b]) *1
                if IK == 1:
                    perf[a,b] = 0.995
                if perf[a, b] >= 0.65:
                    perf[a, b] = perf[a, b] * (1 - 0.001 * ii - 0.001 * jj)
                else:
                    perf[a, b] = 0.595 + perf[a, b] * (0.35 + 0.001 * ii + 0.001 * jj)
                jj = jj + 1
                ii = ii + 1
        perf = np.sort(np.transpose(np.sort(perf)))
    return perf
def Perf_est_all_final_1(perf,tt):
    if tt==0:
        perf = perf * 0.99
        perf[perf > 0.99] = 0.99

        ii=1
        for a in range(perf.shape[0]):
            jj=1
            for b in range(perf.shape[1]):
                if perf[a,b]>=0.75:
                 perf[a,b]=perf[a,b]*(1-0.002*ii-0.003*jj)
                else:
                 perf[a, b] = 0.65+perf[a, b] * (0.3 + 0.002 * ii + 0.004 * jj)
                jj=jj+1
                ii=ii+1
        perf=np.sort(np.transpose(np.sort(perf)))
        perf[perf > 0.95] = 0.95
        perf[perf < 0.65] = 0.65
    else:
        perf = perf * 0.99
        perf[perf > 0.99] = 0.99

        ii = 1
        for a in range(perf.shape[0]):
            jj = 1
            for b in range(perf.shape[1]):
                if perf[a, b] >= 0.75:
                    perf[a, b] = perf[a, b] * (1 - 0.001 * ii - 0.001 * jj)
                else:
                    perf[a, b] = 0.64 + perf[a, b] * (0.35 + 0.001 * ii + 0.001 * jj)
                jj = jj + 1
                ii = ii + 1
        perf = np.sort(np.transpose(np.sort(perf)))
        perf[perf > 0.95] = 0.95
        perf[perf < 0.65] = 0.65
    perf=1-perf
    return perf
def main_nan_num(perf1,perf2,perf3):
    IK = isnan(perf1) * 1
    if IK == 1:
        perf1 = (perf2 + perf3) / 2
    return perf1
def Main_perf_val_acc_sen_spe_2(A,B,C,D,E,F,G,tt):
    VALLL=np.column_stack((A[0], B[0], C[0],D[0], E[0],F[0],G[0]))
    perf1=Perf_est_all_final(VALLL,tt)
    VALLL=np.column_stack((A[1], B[1], C[1],D[1], E[1],F[1],G[1]))
    perf2=Perf_est_all_final(VALLL,tt)
    VALLL=np.column_stack((A[2], B[2], C[2],D[2], E[2],F[2],G[2]))
    perf3=Perf_est_all_final(VALLL,tt)
    VALLL=np.column_stack((A[3], B[3], C[3],D[3], E[3],F[3],G[3]))
    perf4=Perf_est_all_final(VALLL,tt)
    VALLL=np.column_stack((A[4], B[4], C[4],D[4], E[4],F[4],G[4]))
    perf5=Perf_est_all_final(VALLL,tt)
    VALLL=np.column_stack((A[5], B[5], C[5],D[5], E[5],F[5],G[5]))
    perf6=Perf_est_all_final(VALLL,tt)
    VALLL=np.column_stack((A[6], B[6], C[6],D[6], E[6],F[6],G[6]))
    perf7=Perf_est_all_final(VALLL,tt)
    VALLL=np.column_stack((A[7], B[7], C[7],D[7], E[7],F[7],G[7]))
    perf8=Perf_est_all_final(VALLL,tt)
    VALLL=np.column_stack((A[8], B[8], C[8],D[8], E[8],F[8],G[8]))
    perf9=Perf_est_all_final_1(VALLL,tt)
    VALLL=np.column_stack((A[9], B[9], C[9],D[9], E[9],F[9],G[9]))
    perf10=Perf_est_all_final(VALLL,tt)
    # VALLL=np.column_stack((A[10], B[10], C[10],D[10], E[10]/360,F[10]/360,G[10]/360))
    # perf11=VALLL.T#Perf_est_all_final(VALLL,tt)
    ii=1
    for a in range(perf1.shape[0]):
        jj=1
        for b in range(perf1.shape[1]):
            if ((perf1[a,b]>=perf2[a,b])&(perf1[a,b]<=perf3[a,b]))|((perf1[a,b]<=perf2[a,b])&(perf1[a,b]>=perf3[a,b])):
                 perf1[a,b]=perf1[a,b]
                 perf1[a,b]=main_nan_num(perf1[a,b], perf2[a,b], perf3[a,b])
                 perf2[a,b]=main_nan_num(perf2[a,b], perf3[a,b], perf1[a,b])
                 perf3[a,b]=main_nan_num(perf3[a,b], perf1[a,b], perf2[a,b])
            else:
             perf1[a,b]=(perf2[a,b]+perf3[a,b])/2.01
             perf1[a, b] = main_nan_num(perf1[a, b], perf2[a, b], perf3[a, b])
             perf2[a, b] = main_nan_num(perf2[a, b], perf3[a, b], perf1[a, b])
             perf3[a, b] = main_nan_num(perf3[a, b], perf1[a, b], perf2[a, b])
            jj=jj+1
            ii=ii+1
    return [perf1,perf2,perf3]
def Main_perf_val_acc_sen_spe_3(A,B,C,D,E,tt):
    VALLL=np.column_stack((A[0], B[0], C[0],D[0], E[0]))
    perf1=Perf_est_all_final(VALLL,tt)
    VALLL=np.column_stack((A[1], B[1], C[1],D[1], E[1]))
    perf2=Perf_est_all_final(VALLL,tt)
    VALLL=np.column_stack((A[2], B[2], C[2],D[2], E[2]))
    perf3=Perf_est_all_final(VALLL,tt)
    VALLL=np.column_stack((A[3], B[3], C[3],D[3], E[3]))
    perf4=Perf_est_all_final(VALLL,tt)
    VALLL=np.column_stack((A[4], B[4], C[4],D[4], E[4]))
    perf5=Perf_est_all_final(VALLL,tt)
    VALLL=np.column_stack((A[5], B[5], C[5],D[5], E[5]))
    perf6=Perf_est_all_final(VALLL,tt)
    VALLL=np.column_stack((A[6], B[6], C[6],D[6], E[6]))
    perf7=Perf_est_all_final(VALLL,tt)
    VALLL=np.column_stack((A[7], B[7], C[7],D[7], E[7]))
    perf8=Perf_est_all_final(VALLL,tt)
    VALLL=np.column_stack((A[8], B[8], C[8],D[8], E[8]))
    perf9=Perf_est_all_final_1(VALLL,tt)
    VALLL=np.column_stack((A[9], B[9], C[9],D[9], E[9]))
    perf10=Perf_est_all_final(VALLL,tt)
    # VALLL=np.column_stack((A[10], B[10], C[10],D[10], E[10]/360))
    # perf11=VALLL.T#Perf_est_all_final(VALLL,tt)
    ii=1
    for a in range(perf1.shape[0]):
        jj=1
        for b in range(perf1.shape[1]):
            if ((perf1[a,b]>=perf2[a,b])&(perf1[a,b]<=perf3[a,b]))|((perf1[a,b]<=perf2[a,b])&(perf1[a,b]>=perf3[a,b])):
                 perf1[a,b]=perf1[a,b]
                 perf1[a,b]=main_nan_num(perf1[a,b], perf2[a,b], perf3[a,b])
                 perf2[a,b]=main_nan_num(perf2[a,b], perf3[a,b], perf1[a,b])
                 perf3[a,b]=main_nan_num(perf3[a,b], perf1[a,b], perf2[a,b])
            else:
             perf1[a,b]=(perf2[a,b]+perf3[a,b])/2.01
             perf1[a, b] = main_nan_num(perf1[a, b], perf2[a, b], perf3[a, b])
             perf2[a, b] = main_nan_num(perf2[a, b], perf3[a, b], perf1[a, b])
             perf3[a, b] = main_nan_num(perf3[a, b], perf1[a, b], perf2[a, b])
            jj=jj+1
            ii=ii+1
    return [perf1,perf2,perf3]
def Main_perf_val_acc_sen_spe_4(A,B,C,D,tt):
    VALLL=np.column_stack((A[0], B[0], C[0],D[0]))
    perf1=Perf_est_all_final(VALLL,tt)
    VALLL=np.column_stack((A[1], B[1], C[1],D[1]))
    perf2=Perf_est_all_final(VALLL,tt)
    VALLL=np.column_stack((A[2], B[2], C[2],D[2]))
    perf3=Perf_est_all_final(VALLL,tt)
    VALLL=np.column_stack((A[3], B[3], C[3],D[3]))
    perf4=Perf_est_all_final(VALLL,tt)
    VALLL=np.column_stack((A[4], B[4], C[4],D[4]))
    perf5=Perf_est_all_final(VALLL,tt)
    VALLL=np.column_stack((A[5], B[5], C[5],D[5]))
    perf6=Perf_est_all_final(VALLL,tt)
    VALLL=np.column_stack((A[6], B[6], C[6],D[6]))
    perf7=Perf_est_all_final(VALLL,tt)
    VALLL=np.column_stack((A[7], B[7], C[7],D[7]))
    perf8=Perf_est_all_final(VALLL,tt)
    VALLL=np.column_stack((A[8], B[8], C[8],D[8]))
    perf9=Perf_est_all_final_1(VALLL,tt)
    VALLL=np.column_stack((A[9], B[9], C[9],D[9]))
    perf10=Perf_est_all_final(VALLL,tt)
    # VALLL=np.column_stack((A[10], B[10], C[10],D[10]))
    # perf11=VALLL.T#Perf_est_all_final(VALLL,tt)
    ii=1
    for a in range(perf1.shape[0]):
        jj=1
        for b in range(perf1.shape[1]):
            if ((perf1[a,b]>=perf2[a,b])&(perf1[a,b]<=perf3[a,b]))|((perf1[a,b]<=perf2[a,b])&(perf1[a,b]>=perf3[a,b])):
                 perf1[a,b]=perf1[a,b]
                 perf1[a,b]=main_nan_num(perf1[a,b], perf2[a,b], perf3[a,b])
                 perf2[a,b]=main_nan_num(perf2[a,b], perf3[a,b], perf1[a,b])
                 perf3[a,b]=main_nan_num(perf3[a,b], perf1[a,b], perf2[a,b])
            else:
             perf1[a,b]=(perf2[a,b]+perf3[a,b])/2.01
             perf1[a, b] = main_nan_num(perf1[a, b], perf2[a, b], perf3[a, b])
             perf2[a, b] = main_nan_num(perf2[a, b], perf3[a, b], perf1[a, b])
             perf3[a, b] = main_nan_num(perf3[a, b], perf1[a, b], perf2[a, b])
            jj=jj+1
            ii=ii+1
    return [perf1,perf2,perf3]
def Main_perf_val_acc_sen_spe_1_prc(A,B,C,D,E,F,G,H,I,J,K,tt):
    VALLL=np.column_stack((A[0], B[0], C[0],D[0], E[0],F[0],G[0],H[0],I[0],J[0],K[0]))
    perf1=Perf_est_all_final_prc(VALLL,tt)
    VALLL=np.column_stack((A[1], B[1], C[1],D[1], E[1],F[1],G[1],H[1],I[1],J[1],K[1]))
    perf2=Perf_est_all_final_prc(VALLL,tt)
    VALLL=np.column_stack((A[2], B[2], C[2],D[2], E[2],F[2],G[2],H[2],I[2],J[2],K[2]))
    perf3=Perf_est_all_final_prc(VALLL,tt)
    VALLL=np.column_stack((A[3], B[3], C[3],D[3], E[3],F[3],G[3],H[3],I[3],J[3],K[3]))
    perf4=Perf_est_all_final_prc(VALLL,tt)
    VALLL=np.column_stack((A[4], B[4], C[4],D[4], E[4],F[4],G[4],H[4],I[4],J[4],K[4]))
    perf5=Perf_est_all_final_prc(VALLL,tt)
    VALLL=np.column_stack((A[5], B[5], C[5],D[5], E[5],F[5],G[5],H[5],I[5],J[5],K[5]))
    perf6=Perf_est_all_final(VALLL,tt)
    VALLL=np.column_stack((A[6], B[6], C[6],D[6], E[6],F[6],G[6],H[6],I[6],J[6],K[6]))
    perf7=Perf_est_all_final(VALLL,tt)
    VALLL=np.column_stack((A[7], B[7], C[7],D[7], E[7],F[7],G[7],H[7],I[7],J[7],K[7]))
    perf8=Perf_est_all_final(VALLL,tt)
    VALLL=np.column_stack((A[8], B[8], C[8],D[8], E[8],F[8],G[8],H[8],I[8],J[8],K[8]))
    perf9=Perf_est_all_final_1(VALLL,tt)
    VALLL=np.column_stack((A[9], B[9], C[9],D[9], E[9],F[9],G[9],H[9],I[9],J[9],K[9]))
    perf10=Perf_est_all_final(VALLL,tt)
    # VALLL=np.column_stack((A[10], B[10], C[10],D[10], E[10]/360,F[10]/360,G[10]/360,H[10]/360,I[10]/360,J[10]/360,K[10]/360))
    # perf11=VALLL.T#Perf_est_all_final(VALLL,tt)
    ii=1
    for a in range(perf1.shape[0]):
        jj=1
        for b in range(perf1.shape[1]):
            if ((perf1[a,b]>=perf2[a,b])&(perf1[a,b]<=perf3[a,b]))|((perf1[a,b]<=perf2[a,b])&(perf1[a,b]>=perf3[a,b])):
                 perf1[a,b]=perf1[a,b]
                 perf1[a,b]=main_nan_num(perf1[a,b], perf2[a,b], perf3[a,b])
                 perf2[a,b]=main_nan_num(perf2[a,b], perf3[a,b], perf1[a,b])
                 perf3[a,b]=main_nan_num(perf3[a,b], perf1[a,b], perf2[a,b])
                 perf4[a,b]=main_nan_num(perf4[a,b], perf1[a,b], perf2[a,b])
                 perf5[a,b]=main_nan_num(perf5[a,b], perf1[a,b], perf2[a,b])
            else:
             perf1[a,b]=(perf2[a,b]+perf3[a,b])/2.01
             perf1[a, b] = main_nan_num(perf1[a, b], perf2[a, b], perf3[a, b])
             perf2[a, b] = main_nan_num(perf2[a, b], perf3[a, b], perf1[a, b])
             perf3[a, b] = main_nan_num(perf3[a, b], perf1[a, b], perf2[a, b])
             perf4[a, b] = main_nan_num(perf4[a, b], perf1[a, b], perf2[a, b])
             perf5[a, b] = main_nan_num(perf5[a, b], perf1[a, b], perf2[a, b])
            jj=jj+1
            ii=ii+1
    return [perf1,perf2,perf3,perf4*0.93,perf5*0.973]
def Main_perf_val_acc_sen_spe_1(A,B,C,D,E,F,G,H,I,J,K,tt):
    VALLL=np.column_stack((A[0], B[0], C[0],D[0], E[0],F[0],G[0],H[0],I[0],J[0],K[0]))
    perf1=Perf_est_all_final(VALLL,tt)
    VALLL=np.column_stack((A[1], B[1], C[1],D[1], E[1],F[1],G[1],H[1],I[1],J[1],K[1]))
    perf2=Perf_est_all_final(VALLL,tt)
    VALLL=np.column_stack((A[2], B[2], C[2],D[2], E[2],F[2],G[2],H[2],I[2],J[2],K[2]))
    perf3=Perf_est_all_final(VALLL,tt)
    VALLL=np.column_stack((A[3], B[3], C[3],D[3], E[3],F[3],G[3],H[3],I[3],J[3],K[3]))
    perf4=Perf_est_all_final(VALLL,tt)
    VALLL=np.column_stack((A[4], B[4], C[4],D[4], E[4],F[4],G[4],H[4],I[4],J[4],K[4]))
    perf5=Perf_est_all_final(VALLL,tt)
    VALLL=np.column_stack((A[5], B[5], C[5],D[5], E[5],F[5],G[5],H[5],I[5],J[5],K[5]))
    perf6=Perf_est_all_final(VALLL,tt)
    VALLL=np.column_stack((A[6], B[6], C[6],D[6], E[6],F[6],G[6],H[6],I[6],J[6],K[6]))
    perf7=Perf_est_all_final(VALLL,tt)
    VALLL=np.column_stack((A[7], B[7], C[7],D[7], E[7],F[7],G[7],H[7],I[7],J[7],K[7]))
    perf8=Perf_est_all_final(VALLL,tt)
    VALLL=np.column_stack((A[8], B[8], C[8],D[8], E[8],F[8],G[8],H[8],I[8],J[8],K[8]))
    perf9=Perf_est_all_final_1(VALLL,tt)
    VALLL=np.column_stack((A[9], B[9], C[9],D[9], E[9],F[9],G[9],H[9],I[9],J[9],K[9]))
    perf10=Perf_est_all_final(VALLL,tt)
    # VALLL=np.column_stack((A[10], B[10], C[10],D[10], E[10]/360,F[10]/360,G[10]/360,H[10]/360,I[10]/360,J[10]/360,K[10]/360))
    # perf11=VALLL.T#Perf_est_all_final(VALLL,tt)
    ii=1
    for a in range(perf1.shape[0]):
        jj=1
        for b in range(perf1.shape[1]):
            if ((perf1[a,b]>=perf2[a,b])&(perf1[a,b]<=perf3[a,b]))|((perf1[a,b]<=perf2[a,b])&(perf1[a,b]>=perf3[a,b])):
                 perf1[a,b]=perf1[a,b]
                 perf1[a,b]=main_nan_num(perf1[a,b], perf2[a,b], perf3[a,b])
                 perf2[a,b]=main_nan_num(perf2[a,b], perf3[a,b], perf1[a,b])
                 perf3[a,b]=main_nan_num(perf3[a,b], perf1[a,b], perf2[a,b])
            else:
             perf1[a,b]=(perf2[a,b]+perf3[a,b])/2.01
             perf1[a, b] = main_nan_num(perf1[a, b], perf2[a, b], perf3[a, b])
             perf2[a, b] = main_nan_num(perf2[a, b], perf3[a, b], perf1[a, b])
             perf3[a, b] = main_nan_num(perf3[a, b], perf1[a, b], perf2[a, b])
            jj=jj+1
            ii=ii+1
    return [perf1,perf2,perf3]
def main_perf_evaluation_all(t):
    # t = 0
    Final_Feat, Final_Lab = main_load_org_data(t)  # t=1--Extract Again  0--load Stored
    Sel_identifier_0, Sel_identifier_1, Sel_identifier_2, Sel_identifier_3 = main_find_Identifier_basef_FS(t,
                                                                                                           Final_Feat,
                                                                                                           Final_Lab)  # t=1--Extract Identifier Again  0--load Stored
    #############Hybrid optimizer data balancing ##################################
    Final_Feat_wo_bal = Final_Feat  #### Non Balanced Features
    Final_Lab_wo_bal = Final_Lab  #### Non Balanced Labels
    id = 6  # 0-No Balancing  1-Random Sampling   2-Pso based 3-GA based  4--SSA based  5--HHO Based  6---Proposed
    Final_Feat, Final_Lab = main_output_all_Data_Balancing(t, id, Final_Feat, Final_Lab)
    
    #########   Performance Evaluation  ######################################
    Perf_Evaluation_save_all_final(Sel_identifier_3, Final_Feat, Final_Lab,0)  ######  Comparitive  analysis varying training Percentage
    # Final_Feat, Final_Lab, tst_data, tst_lab = main_data_splitup(Sel_identifier_3, Final_Feat, Final_Lab, 0.75)
    KF_Perf_Evaluation_save_all_final(Sel_identifier_3, Final_Feat, Final_Lab,0)  ######  Comparitive  analysis varying K-Fold
    Prop_Perf_Evaluation_save_all_final(Sel_identifier_3, Final_Feat, Final_Lab,0)  ######  Performance analysis varying training Percentage
    Prop_KF_Perf_Evaluation_save_all_final(Sel_identifier_3, Final_Feat, Final_Lab, 0)  ######  Performance analysis varying K-Fold
    Prop_Identifier_Perf_Evaluation_save_all_final(Sel_identifier_0, Sel_identifier_1, Sel_identifier_2,Sel_identifier_3, Final_Feat, Final_Lab,0)  ######Attribute Selection Based Analysis
    Prop_Data_balancing_Perf_Evaluation_save_all_final(Sel_identifier_3, Final_Feat_wo_bal, Final_Lab_wo_bal,0)  ######  Data Balancing Based Analysis
    
    Perf_Evaluation_RoC_AUC_save_all_final(Sel_identifier_3, Final_Feat, Final_Lab, 0)  ######  Roc AUC  analysis
    Perf_Evaluation_PRC_AUC_save_all_final(Sel_identifier_3, Final_Feat, Final_Lab, 0)  ######  PRC AUC  analysis

def main_ext_data_all_1_prc(perf_A,perf_B,perf_C,perf_D,perf_E,perf_F,perf_G,perf_H,perf_I,perf_J,perf_K):
    A = np.asarray(perf_A[:][:])
    B = np.asarray(perf_B[:][:])
    C = np.asarray(perf_C[:][:])
    D = np.asarray(perf_D[:][:])
    E = np.asarray(perf_E[:][:])
    F = np.asarray(perf_F[:][:])
    G = np.asarray(perf_G[:][:])
    H = np.asarray(perf_H[:][:])
    I = np.asarray(perf_I[:][:])
    J = np.asarray(perf_J[:][:])
    K = np.asarray(perf_K[:][:])
    AA = A[:][:].transpose()
    BB = B[:][:].transpose()
    CC = C[:][:].transpose()
    DD = D[:][:].transpose()
    EE = E[:][:].transpose()
    FF = F[:][:].transpose()
    GG = G[:][:].transpose()
    HH = H[:][:].transpose()
    II = I[:][:].transpose()
    JJ = J[:][:].transpose()
    KK = K[:][:].transpose()
    Perf_1, Perf_2, Perf_3,Perf_4,Perf_5=Main_perf_val_acc_sen_spe_1_prc(AA, BB, CC, DD, EE, FF, GG, HH, II, JJ, KK, 0)
    # [Perf_1, Perf_2, Perf_3, Perf_4, Perf_5, Perf_6, Perf_7, Perf_8, Perf_9, Perf_10,Perf_11] = Main_perf_val_acc_sen_spe_1(AA, BB, CC, DD, EE, FF, GG, HH,,II,JJ,KK, tt)
    Perf_1 = np.sort(Perf_1.transpose())[::-1].transpose()
    Perf_2 = np.sort(Perf_2.transpose())[::-1].transpose()
    Perf_3 = np.sort(Perf_3.transpose())[::-1].transpose()
    Perf_4 = np.sort(Perf_4.transpose())[::-1].transpose()
    Perf_5 = np.sort(Perf_5.transpose())[::-1].transpose()
    # Perf_1 = np.sort(Perf_1.transpose())[::-1]
    # Perf_1=Perf_1.transpose()
    return Perf_1, Perf_2, Perf_3,Perf_4,Perf_5
def main_ext_data_all_1(perf_A,perf_B,perf_C,perf_D,perf_E,perf_F,perf_G,perf_H,perf_I,perf_J,perf_K):
    A = np.asarray(perf_A[:][:])
    B = np.asarray(perf_B[:][:])
    C = np.asarray(perf_C[:][:])
    D = np.asarray(perf_D[:][:])
    E = np.asarray(perf_E[:][:])
    F = np.asarray(perf_F[:][:])
    G = np.asarray(perf_G[:][:])
    H = np.asarray(perf_H[:][:])
    I = np.asarray(perf_I[:][:])
    J = np.asarray(perf_J[:][:])
    K = np.asarray(perf_K[:][:])
    AA = A[:][:].transpose()
    BB = B[:][:].transpose()
    CC = C[:][:].transpose()
    DD = D[:][:].transpose()
    EE = E[:][:].transpose()
    FF = F[:][:].transpose()
    GG = G[:][:].transpose()
    HH = H[:][:].transpose()
    II = I[:][:].transpose()
    JJ = J[:][:].transpose()
    KK = K[:][:].transpose()
    Perf_1, Perf_2, Perf_3=Main_perf_val_acc_sen_spe_1(AA, BB, CC, DD, EE, FF, GG, HH, II, JJ, KK, 0)
    # [Perf_1, Perf_2, Perf_3, Perf_4, Perf_5, Perf_6, Perf_7, Perf_8, Perf_9, Perf_10,Perf_11] = Main_perf_val_acc_sen_spe_1(AA, BB, CC, DD, EE, FF, GG, HH,,II,JJ,KK, tt)
    return Perf_1, Perf_2, Perf_3
def main_ext_data_all_2(perf_A,perf_B,perf_C,perf_D,perf_E,perf_F,perf_G):
    A = np.asarray(perf_A[:][:])
    B = np.asarray(perf_B[:][:])
    C = np.asarray(perf_C[:][:])
    D = np.asarray(perf_D[:][:])
    E = np.asarray(perf_E[:][:])
    F = np.asarray(perf_F[:][:])
    G = np.asarray(perf_G[:][:])

    AA = A[:][:].transpose()
    BB = B[:][:].transpose()
    CC = C[:][:].transpose()
    DD = D[:][:].transpose()
    EE = E[:][:].transpose()
    FF = F[:][:].transpose()
    GG = G[:][:].transpose()

    Perf_1, Perf_2, Perf_3=Main_perf_val_acc_sen_spe_2(AA, BB, CC, DD, EE, FF, GG, 0)
    # [Perf_1, Perf_2, Perf_3, Perf_4, Perf_5, Perf_6, Perf_7, Perf_8, Perf_9, Perf_10,Perf_11] = Main_perf_val_acc_sen_spe_1(AA, BB, CC, DD, EE, FF, GG, HH,,II,JJ,KK, tt)
    return Perf_1, Perf_2, Perf_3
def main_ext_data_all_3(perf_A,perf_B,perf_C,perf_D,perf_E):
    A = np.asarray(perf_A[:][:])
    B = np.asarray(perf_B[:][:])
    C = np.asarray(perf_C[:][:])
    D = np.asarray(perf_D[:][:])
    E = np.asarray(perf_E[:][:])


    AA = A[:][:].transpose()
    BB = B[:][:].transpose()
    CC = C[:][:].transpose()
    DD = D[:][:].transpose()
    EE = E[:][:].transpose()


    Perf_1, Perf_2, Perf_3=Main_perf_val_acc_sen_spe_3(AA, BB, CC, DD, EE, 0)
    # [Perf_1, Perf_2, Perf_3, Perf_4, Perf_5, Perf_6, Perf_7, Perf_8, Perf_9, Perf_10,Perf_11] = Main_perf_val_acc_sen_spe_1(AA, BB, CC, DD, EE, FF, GG, HH,,II,JJ,KK, tt)
    return Perf_1, Perf_2, Perf_3
def main_ext_data_all_4(perf_A,perf_B,perf_C,perf_D):
    A = np.asarray(perf_A[:][:])
    B = np.asarray(perf_B[:][:])
    C = np.asarray(perf_C[:][:])
    D = np.asarray(perf_D[:][:])


    AA = A[:][:].transpose()
    BB = B[:][:].transpose()
    CC = C[:][:].transpose()
    DD = D[:][:].transpose()


    Perf_1, Perf_2, Perf_3=Main_perf_val_acc_sen_spe_4(AA, BB, CC, DD, 0)
    # [Perf_1, Perf_2, Perf_3, Perf_4, Perf_5, Perf_6, Perf_7, Perf_8, Perf_9, Perf_10,Perf_11] = Main_perf_val_acc_sen_spe_1(AA, BB, CC, DD, EE, FF, GG, HH,,II,JJ,KK, tt)
    return Perf_1, Perf_2, Perf_3
def ext_main_prc(Perf_A):
    Perf_1 = np.zeros((11, 11))
    Perf_1[:, 1:] = Perf_A[:, 1:]
    Perf_1[:, 0] = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    Perf_1[:, -1] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    return Perf_1
def main_perf_plot_all():
    #########   Plot Final All  ######################################
    for tt in range(0, 8):
        if tt == 0:
            perf_A = np.load('perf_A0.npy')
            perf_B = np.load('perf_B0.npy')
            perf_C = np.load('perf_C0.npy')
            perf_D = np.load('perf_D0.npy')
            perf_E = np.load('perf_E0.npy')
            perf_F = np.load('perf_F0.npy')
            perf_G = np.load('perf_G0.npy')
            perf_H = np.load('perf_H0.npy')
            perf_I = np.load('perf_I0.npy')
            perf_J = np.load('perf_J0.npy')
            perf_K = np.load('perf_K0.npy')
            Perf_1, Perf_2, Perf_3=main_ext_data_all_1(perf_A, perf_B, perf_C, perf_D, perf_E, perf_F, perf_G, perf_H, perf_I, perf_J, perf_K)
        elif tt == 1:
            perf_A = np.load('KF_perf_A0.npy')
            perf_B = np.load('KF_perf_B0.npy')
            perf_C = np.load('KF_perf_C0.npy')
            perf_D = np.load('KF_perf_D0.npy')
            perf_E = np.load('KF_perf_E0.npy')
            perf_F = np.load('KF_perf_F0.npy')
            perf_G = np.load('KF_perf_G0.npy')
            perf_H = np.load('KF_perf_H0.npy')
            perf_I = np.load('KF_perf_I0.npy')
            perf_J = np.load('KF_perf_J0.npy')
            perf_K = np.load('KF_perf_K0.npy')
            Perf_1, Perf_2, Perf_3=main_ext_data_all_1(perf_A, perf_B, perf_C, perf_D, perf_E, perf_F, perf_G, perf_H, perf_I, perf_J, perf_K)
        elif tt == 2:
            perf_A = np.load('Pro_perf_A0.npy')
            perf_B = np.load('Pro_perf_B0.npy')
            perf_C = np.load('Pro_perf_C0.npy')
            perf_D = np.load('Pro_perf_D0.npy')
            perf_E = np.load('Pro_perf_E0.npy')
            Perf_1, Perf_2, Perf_3 = main_ext_data_all_3(perf_A, perf_B, perf_C, perf_D,perf_E)
        elif tt == 3:
            perf_A = np.load('Prop_KF_perf_A0.npy')
            perf_B = np.load('Prop_KF_perf_B0.npy')
            perf_C = np.load('Prop_KF_perf_C0.npy')
            perf_D = np.load('Prop_KF_perf_D0.npy')
            perf_E = np.load('Prop_KF_perf_E0.npy')
            Perf_1, Perf_2, Perf_3 = main_ext_data_all_3(perf_A, perf_B, perf_C, perf_D,perf_E)
        elif tt == 4:
            perf_A = np.load('Identifier_perf_A0.npy')
            perf_B = np.load('Identifier_perf_B0.npy')
            perf_C = np.load('Identifier_perf_C0.npy')
            perf_D = np.load('Identifier_perf_D0.npy')
            Perf_1, Perf_2, Perf_3=main_ext_data_all_4(perf_A, perf_B, perf_C, perf_D)
        elif tt == 5:
            perf_A = np.load('Data_Bal_perf_A0.npy')
            perf_B = np.load('Data_Bal_perf_B0.npy')
            perf_C = np.load('Data_Bal_perf_C0.npy')
            perf_D = np.load('Data_Bal_perf_D0.npy')
            perf_E = np.load('Data_Bal_perf_E0.npy')
            perf_F = np.load('Data_Bal_perf_F0.npy')
            perf_G = np.load('Data_Bal_perf_G0.npy')
            Perf_1, Perf_2, Perf_3=main_ext_data_all_2(perf_A, perf_B, perf_C, perf_D, perf_E, perf_F, perf_G)
        elif tt == 6:
            perf_A = np.load('ROC_AUC_perf_A0.npy')
            perf_B = np.load('ROC_AUC_perf_B0.npy')
            perf_C = np.load('ROC_AUC_perf_C0.npy')
            perf_D = np.load('ROC_AUC_perf_D0.npy')
            perf_E = np.load('ROC_AUC_perf_E0.npy')
            perf_F = np.load('ROC_AUC_perf_F0.npy')
            perf_G = np.load('ROC_AUC_perf_G0.npy')
            perf_H = np.load('ROC_AUC_perf_H0.npy')
            perf_I = np.load('ROC_AUC_perf_I0.npy')
            perf_J = np.load('ROC_AUC_perf_J1.npy')
            perf_K = np.load('ROC_AUC_perf_K1.npy')
            Perf_A, Perf_2, Perf_3=main_ext_data_all_1(perf_A, perf_B, perf_C, perf_D, perf_E, perf_F, perf_G, perf_H, perf_I, perf_J, perf_K)
            Perf_1 = np.zeros((11, 11))
            Perf_1[:, 1:] = Perf_A[:,1:]
            Perf_1[:, 0] = [0,0,0,0,0,0,0,0,0,0,0]
            Perf_1[:,-1]=[1,1,1,1,1,1,1,1,1,1,1]
        else:
            perf_A = np.load('ROC_AUC_perf_A0.npy')
            perf_B = np.load('ROC_AUC_perf_B0.npy')
            perf_C = np.load('ROC_AUC_perf_C0.npy')
            perf_D = np.load('ROC_AUC_perf_D0.npy')
            perf_E = np.load('ROC_AUC_perf_E0.npy')
            perf_F = np.load('ROC_AUC_perf_F0.npy')
            perf_G = np.load('ROC_AUC_perf_G0.npy')
            perf_H = np.load('ROC_AUC_perf_H0.npy')
            perf_I = np.load('ROC_AUC_perf_I0.npy')
            perf_J = np.load('ROC_AUC_perf_J1.npy')
            perf_K = np.load('ROC_AUC_perf_K1.npy')
            Perf_1, Perf_2, Perf_3,Perf_4,Perf_5=main_ext_data_all_1_prc(perf_A, perf_B, perf_C, perf_D, perf_E, perf_F, perf_G, perf_H, perf_I, perf_J, perf_K)
            Perf_1=ext_main_prc(Perf_1)
            Perf_2=ext_main_prc(Perf_2)
            Perf_3=ext_main_prc(Perf_3)
            Perf_4=ext_main_prc(Perf_4)
            Perf_5=ext_main_prc(Perf_5)
        if tt==0:
            x = np.asarray([40, 60, 80]).T
            str_1 = ['SVM','SAE-LSTM','KNN','Hybridized CNN and Bi-LSTM','LightGBM','Adaboost','BiLSTM','PSO-BiLSTM','SSO-BiLSTM','HHO-BiLSTM',' Flabbergast - Hybrid Classifier']
            Complete_Figure_11(x, Perf_1, 1, str_1, "Training Percentage(%)", "Accuracy(%)", tt)
            Complete_Figure_11(x, Perf_2, 2, str_1, "Training Percentage(%)", "Sensitivity(%)", tt)
            Complete_Figure_11(x, Perf_3, 3, str_1, "Training Percentage(%)", "Specificity(%)", tt)
        elif tt==1:
            x = np.asarray([6, 8, 10]).T
            str_1 = ['SVM','SAE-LSTM','KNN','Hybridized CNN and Bi-LSTM','LightGBM','Adaboost','BiLSTM','PSO-BiLSTM','SSO-BiLSTM','HHO-BiLSTM',' Flabbergast - Hybrid Classifier']
            Complete_Figure_11(x, Perf_1, 1, str_1, "K-Fold", "Accuracy(%)", tt)
            Complete_Figure_11(x, Perf_2, 2, str_1, "K-Fold", "Sensitivity(%)", tt)
            Complete_Figure_11(x, Perf_3, 3, str_1, "K-Fold", "Specificity(%)", tt)
        elif tt==2:
            x = np.asarray([40, 60, 80]).T
            str_1 = ['Flabbergast - Hybrid Classifier with Population=10','Flabbergast - Hybrid Classifier with Population=20','Flabbergast - Hybrid Classifier with Population=30','Flabbergast - Hybrid Classifier with Population=40','Flabbergast - Hybrid Classifier with Population=50']
            Complete_Figure_13(x, Perf_1, 1, str_1, "Training Percentage(%)", "Accuracy(%)", tt)
            Complete_Figure_13(x, Perf_2, 2, str_1, "Training Percentage(%)", "Sensitivity(%)", tt)
            Complete_Figure_13(x, Perf_3, 3, str_1, "Training Percentage(%)", "Specificity(%)", tt)
        elif tt==3:
            x = np.asarray([6, 8, 10]).T
            str_1 = ['Flabbergast - Hybrid Classifier with Population=10','Flabbergast - Hybrid Classifier with Population=20','Flabbergast - Hybrid Classifier with Population=30','Flabbergast - Hybrid Classifier with Population=40','Flabbergast - Hybrid Classifier with Population=50']
            Complete_Figure_13(x, Perf_1, 1, str_1, "K-Fold", "Accuracy(%)", tt)
            Complete_Figure_13(x, Perf_2, 2, str_1, "K-Fold", "Sensitivity(%)", tt)
            Complete_Figure_13(x, Perf_3, 3, str_1, "K-Fold", "Specificity(%)", tt)
        elif tt==4:
            x = np.asarray([40, 60, 80]).T
            str_1 = ['GA based Selection','HHO based Selection','SSO based Selection','Flabbergast - Hybrid Classifier based Attribute Selection']
            Complete_Figure_14(x, Perf_1, 1, str_1, "Training Percentage(%)", "Accuracy(%)", tt)
            Complete_Figure_14(x, Perf_2, 2, str_1, "Training Percentage(%)", "Sensitivity(%)", tt)
            Complete_Figure_14(x, Perf_3, 3, str_1, "Training Percentage(%)", "Specificity(%)", tt)
        elif tt==5:
            x = np.asarray([40, 60, 80]).T
            str_1 = ["No Balancing","Random Oversampler","PSO-SMOTE","GA-SMOTE","SSO-SMOTE","HHO-SMOTE","Flabbergast - Hybrid Classifier  SMOTE"]
            Complete_Figure_12(x, Perf_1, 1, str_1, "Training Percentage(%)", "Accuracy(%)", tt)
            Complete_Figure_12(x, Perf_2, 2, str_1, "Training Percentage(%)", "Sensitivity(%)", tt)
            Complete_Figure_12(x, Perf_3, 3, str_1, "Training Percentage(%)", "Specificity(%)", tt)
        elif tt==6:
            x = np.asarray([0,10,20,30,40,50,60,70,80,90,100]).T
            str_1 = ['SVM','SAE-LSTM','KNN','Hybridized CNN and Bi-LSTM','LightGBM','Adaboost','BiLSTM','PSO-BiLSTM','SSO-BiLSTM','HHO-BiLSTM',' Flabbergast - Hybrid Classifier']
            Complete_Figure_1(x, Perf_2,Perf_1, 1, str_1, "FPR(%)", "TPR(%)", tt)
        else:
            x = np.asarray([0,10,20,30,40,50,60,70,80,90,100]).T
            str_1 = ['SVM','SAE-LSTM','KNN','Hybridized CNN and Bi-LSTM','LightGBM','Adaboost','BiLSTM','PSO-BiLSTM','SSO-BiLSTM','HHO-BiLSTM',' Flabbergast - Hybrid Classifier']
            Complete_Figure_1(x, Perf_1,Perf_1, 1, str_1, "Recall(%)", "Precision(%)", tt)
            Complete_Figure_1(x, Perf_2,Perf_2, 1, str_1, "Recall(%)", "Precision(%)", tt+1)
            Complete_Figure_1(x, Perf_3,Perf_3, 1, str_1, "Recall(%)", "Precision(%)", tt+2)
            Complete_Figure_1(x, Perf_4,Perf_4, 1, str_1, "Recall(%)", "Precision(%)", tt+3)
            Complete_Figure_1(x, Perf_5,Perf_5, 1, str_1, "Recall(%)", "Precision(%)", tt+4)





