import warnings
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
from imblearn.over_sampling import RandomOverSampler
import lightgbm as lgb
from sklearn.ensemble import AdaBoostClassifier
# from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.callbacks import ModelCheckpoint,EarlyStopping
import tensorflow as tf
from scipy.stats import kurtosis,skew
import smote_variants as sv
# from sklearn.externals import joblib
import joblib
from sklearn.metrics import confusion_matrix
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
    plt.show()
def main_data_splitup(Sel_identifier_3,Final_Feat,Final_Lab,tr_per):
    tr_data, tst_data, tr_lab, tst_lab = [], [], [], []
    for y in range(0, len(tot_attacks)):
        ind_1 = np.where(Final_Lab == y)[0]
        try:
             ind_1=ind_1[:250,]
        except:
            ind_1=ind_1
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
    tr_data = tr_data[:, Sel_identifier_3]
    tst_data = tst_data[:, Sel_identifier_3]
    return tr_data,tr_lab,tst_data,tst_lab
def main_data_splitup_tem(Final_Feat,Final_Lab,tr_per):
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
def main_CNN_LSTM_Classifier(tr_data,tr_lab,tst_data,ep):
    tr_data = np.reshape(tr_data, (tr_data.shape[0], tr_data.shape[1], 1))
    tst_data = np.reshape(tst_data, (tst_data.shape[0], tst_data.shape[1], 1))
    tr_lab = to_categorical(tr_lab)
    ###############################################
    batch_size = 128
    model = Sequential()
    model.add(Convolution1D(64, kernel_size=64, border_mode="same", activation="relu", input_shape=(tr_data.shape[1], 1)))
    model.add(MaxPooling1D(pool_length=(10)))
    model.add(BatchNormalization())
    model.add(Bidirectional(LSTM(64, return_sequences=False)))
    model.add(Reshape((128, 1), input_shape=(128,)))
    model.add(MaxPooling1D(pool_length=(5)))
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
def main_BiLSTM_Classifier(tr_data,tr_lab,tst_data,ep):
    tr_data=tf.keras.utils.normalize(tr_data)
    tst_data=tf.keras.utils.normalize(tst_data)

    tr_data = np.reshape(tr_data, (tr_data.shape[0], tr_data.shape[1], 1))
    tst_data = np.reshape(tst_data, (tst_data.shape[0], tst_data.shape[1], 1))
    tr_lab = to_categorical(tr_lab)
    model = Sequential()
    model.add(Bidirectional(LSTM(64, activation='relu'), input_shape=(tr_data.shape[1],1)))
    model.add(Reshape((128, 1), input_shape=(128,)))
    model.add(MaxPooling1D(pool_length=(5)))
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
        oversampler= sv.SMOTE_PSO(k=3,nn_params={},eps=0.05,n_pop=5,w=1.0,c1=2.0, c2=2.0, num_it=3,n_jobs=1,random_state=None)
        X_samp, y_samp = oversampler.sample(Final_Feat, Final_Lab)
    elif ii==3:
        oversampler= sv.GASMOTE(n_neighbors=2,nn_params={}, maxn=7, n_pop=5,popl3=5,pm=0.3,pr=0.2,Ge=2,n_jobs=1,random_state=None)
        X_samp, y_samp = oversampler.sample(Final_Feat, Final_Lab)
    elif ii==4:
        oversampler = sv.SSO(proportion=1.0,n_neighbors=5,nn_params={},h=10,n_iter=2,n_jobs=1,random_state=None)
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
    model.add(MaxPooling1D(pool_length=(5)))
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
    model.add(MaxPooling1D(pool_length=(5)))
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
        plt.show()
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


#####################       Main Code           #############################
df1 = pd.read_csv(os.getcwd()+'\DB\csv_enp0s3-monday.pcap_Flow.csv')
df2= pd.read_csv(os.getcwd()+'\DB\csv_enp0s3-public-tuesday.pcap_Flow.csv')
df3 = pd.read_csv(os.getcwd()+'\DB\csv_enp0s3-public-wednesday.pcap_Flow.csv')
df4 = pd.read_csv(os.getcwd()+'\DB\csv_enp0s3-public-thursday.pcap_Flow.csv')
df5 = pd.read_csv(os.getcwd()+'\DB\csv_enp0s3-tcpdump-friday.pcap_Flow.csv')

tem_feat_1=df1.values
tem_feat_2=df2.values
tem_feat_3=df3.values
tem_feat_4=df4.values
tem_feat_5=df5.values
############   Combine All Data   #############
tem_feat=np.vstack((tem_feat_1,tem_feat_2,tem_feat_3,tem_feat_4,tem_feat_5))
tem_lab=tem_feat[:,-1]
Str_lab = np.unique(tem_feat[:, -1])

fin_feat_1=main_lab_change(tem_feat_1,Str_lab)
fin_feat_2=main_lab_change(tem_feat_2,Str_lab)
fin_feat_3=main_lab_change(tem_feat_3,Str_lab)
fin_feat_4=main_lab_change(tem_feat_4,Str_lab)
fin_feat_5=main_lab_change(tem_feat_5,Str_lab)
fin_feat=np.vstack((fin_feat_1,fin_feat_2,fin_feat_3,fin_feat_4,fin_feat_5))
fin_feat = np.asarray(fin_feat, dtype=float)

Final_Feat=fin_feat[:,:-2]
Final_Lab= np.asarray(fin_feat[:,-1], dtype=int)
tot_attacks=np.unique(Final_Lab)
tr_per=0.75
tr_data,tr_lab,tst_data,tst_lab=main_data_splitup_tem(Final_Feat,Final_Lab,tr_per)######  Data splitup for Attribute Selection
# ##########Quasi identifier detection-based Risk attribute detection and pre-processing  #############
# Sel_identifier_0=prop_Important_Identifier_Detection(tr_data,tr_lab,tst_data,tst_lab,0,jfs_0)#####GA
# Sel_identifier_1=prop_Important_Identifier_Detection(tr_data,tr_lab,tst_data,tst_lab,0,jfs_1)#####SSA
# Sel_identifier_2=prop_Important_Identifier_Detection(tr_data,tr_lab,tst_data,tst_lab,0,jfs_2)######HHO
Sel_identifier_3=prop_Important_Identifier_Detection(tr_data,tr_lab,tst_data,tst_lab,0,jfs_3)#####Prop
###########Time domain-based statistical feature extraction + Data attributes#####################
Final_Feat=main_feature_combine(Final_Feat)#####Time domain based Features (statistical Features )Extraction
#############Hybrid optimizer data balancing ##################################
Final_Feat_wo_bal=Final_Feat#### Non Balanced Features
Final_Lab_wo_bal=Final_Lab#### Non Balanced Labels
id=6#0-No Balancing  1-Random Sampling   2-Pso based 3-GA based  4--SSA based  5--HHO Based  6---Proposed
Final_Feat, Final_Lab=main_Data_Balancing_optimization(id,Final_Feat, Final_Lab)
#########   Performance Evaluation  ######################################
# Perf_Evaluation_save_all_final(Sel_identifier_3, Final_Feat, Final_Lab,0)######  Comparitive  analysis varying training Percentage
# KF_Perf_Evaluation_save_all_final(Sel_identifier_3, Final_Feat, Final_Lab,0)######  Comparitive  analysis varying K-Fold
# Prop_Perf_Evaluation_save_all_final(Sel_identifier_3, Final_Feat, Final_Lab,0)######  Performance analysis varying training Percentage
# Prop_KF_Perf_Evaluation_save_all_final(Sel_identifier_3, Final_Feat, Final_Lab,0)######  Performance analysis varying K-Fold
#
# Prop_Identifier_Perf_Evaluation_save_all_final(Sel_identifier_0,Sel_identifier_1,Sel_identifier_2,Sel_identifier_3, Final_Feat, Final_Lab,0)######Attribute Selection Based Analysis
# Prop_Data_balancing_Perf_Evaluation_save_all_final(Sel_identifier_3,Final_Feat_wo_bal,Final_Lab_wo_bal,0)######  Data Balancing Based Analysis
tr_data, tr_lab, tst_data, tst_lab = main_data_splitup(Sel_identifier_3, Final_Feat, Final_Lab, tr_per)
epoch=10
pred_1 = main_KNN_Classifier(tr_data, tr_lab, tst_data)  ######  Knn Classifier
pred_2 = main_CNN_LSTM_Classifier(tr_data, tr_lab, tst_data, epoch)  ######  CNN-LSTM Classifier
pred_3 = main_LightGBM_Classifier(tr_data, tr_lab, tst_data)  ######  LightGBM Classifier
pred_4 = main_AdaBoost_Classifier(tr_data, tr_lab, tst_data)  ######  Adaboost Classifier
pred_5 = main_BiLSTM_Classifier(tr_data, tr_lab, tst_data, epoch)  ######  BiLSTM Classifier
pred_6 = main_BiLSTM_LiGBM_Classifier(tr_data, tr_lab, tst_data, tst_lab, 0, epoch)  ######  PSO Tuned BiLSTM Classifier
pred_7 = main_BiLSTM_LiGBM_Classifier(tr_data, tr_lab, tst_data, tst_lab, 1, epoch)  ######  SSO Tuned BiLSTM Classifier
pred_8 = main_BiLSTM_LiGBM_Classifier(tr_data, tr_lab, tst_data, tst_lab, 2, epoch)  ######  HHO Tuned BiLSTM Classifier
pred_9 = main_BiLSTM_LiGBM_Classifier_Mod(tr_data, tr_lab, tst_data, tst_lab, 3,
                                          epoch)  ######  Proposed Tuned BiLSTM Classifier
confussion_matrix=confusion_matrix(tst_lab, pred_9, labels=[0, 1, 2, 3,4])
plot_confusion_matrix(cm= confussion_matrix,normalize    = False, target_names = ['Benign', 'Data Exfiltration', 'Establish Foothold', 'Lateral Movement', 'Reconnaissance'],title = "Confusion Matrix")

