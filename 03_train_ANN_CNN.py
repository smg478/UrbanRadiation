#=======================================================================================================================
# This file traines a Convolutional neural network (CNN) and a Multi layer perceptron (MLP) neural network
# using training features generated in script 02
# Model weights will be saved in weights/ folder
# @ Shaikat Galib / smg478@mst.edu / 25/04/2019
#=======================================================================================================================
from __future__ import print_function, division
import sys
import numpy as np
import pandas as pd
from keras.models import Model
np.random.seed(203)
from keras.utils import to_categorical
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.metrics import f1_score
from sklearn.externals import joblib
from keras.layers import concatenate
from sklearn.preprocessing import RobustScaler
from keras.layers import Input, Dense, Dropout, Conv1D, Flatten, BatchNormalization, GlobalMaxPool1D
import os
#=======================================================================================================================
# bash train.sh /data/training/ /data/trainingAnswers.csv

########################################################################################################################
expt_name = 'ANN_CNN'
train_folder = '/data/training/'
train_answr = '/data/trainingAnswers.csv'
wdata_dir = '/wdata/'
if len(sys.argv) > 1:
    train_folder = sys.argv[1]
    train_answr = sys.argv[2]
########################################################################################################################

df_train = pd.read_csv(wdata_dir + 'train_feature_bin_30_slice.csv')
#=======================================================================================================================
# Make weight directories
weight_dir ='weights/' + expt_name + '/'
if not os.path.exists(weight_dir):
    os.makedirs(weight_dir)

#=======================================================================================================================
target = df_train.iloc[:, -1]
y= to_categorical(target, num_classes=len(np.unique(target)))
x_trn = df_train.iloc[:,1:-1]

# scale train ==========================================================================================================
X = x_trn.values
where_are_NaNs = np.isnan(X)
where_are_infs = np.isinf(X)
X[where_are_NaNs] = 0
X[where_are_infs] = 0

scaler = RobustScaler()
scaler.fit(X)
# scaler_filename = "scaler.save"
# joblib.dump(scaler, scaler_filename)

scaled_train_X = scaler.transform(X)
X = scaled_train_X
X = X.reshape(len(df_train), len(X[0]), 1)

#========================================================================================================================

def init_model():
    inp = Input(shape=(len(X[0]), 1))

    a = Conv1D(64, 5, activation="relu", kernel_initializer="uniform", )(inp)
    a = BatchNormalization()(a)
    a = Conv1D(64, 5, activation="relu", kernel_initializer="uniform", )(a)
    a = BatchNormalization()(a)
    max_pool = GlobalMaxPool1D()(a)

    b = Flatten()(inp)
    ab = concatenate([ max_pool, b])

    a = Dense(128, activation="relu", kernel_initializer="uniform")(ab)
    a = Dropout(0.5)(a)
    a = Dense(128, activation="relu", kernel_initializer="uniform")(a)

    output = Dense(7, activation="softmax", kernel_initializer="uniform")(a)
    model = Model(inp, output)

    return model

#======================================================================================================================
num_folds = 5
for i in range (7):
    _ids = df_train.index[df_train['152'] == i].tolist()
    all_length = len(_ids)
    fold_len = int(all_length / num_folds)

    init_idx = 0
    for j in range (num_folds):

        _train_idx = _ids[init_idx: init_idx + fold_len]
        init_idx = init_idx + fold_len
        df_train.loc[_train_idx, 'fold'] = j
df_train = df_train.fillna(0)

#=======================================================================================================================
oof = np.zeros(shape = (len(df_train), 7))

for fold_ in range(num_folds):
    trn_idx = df_train.index[df_train['fold'] != fold_].tolist()
    val_idx = df_train.index[df_train['fold'] == fold_].tolist()

    X_train, X_test = X[trn_idx], X[val_idx]
    y_train, y_test = y[trn_idx], y[val_idx]

    #===================================================================================================================
    callbacks = [EarlyStopping(monitor='val_acc',
                               patience=100,
                               verbose=2,
                               min_delta=1e-4,
                               mode='max'),
                 ReduceLROnPlateau(monitor='val_acc',
                                   factor=0.1,
                                   patience=50,
                                   cooldown=2,
                                   verbose=1,
                                   min_delta=1e-4,
                                   mode='max'),
                 ModelCheckpoint(monitor='val_acc',
                                 filepath=weight_dir + 'model_{}.hdf5'.format(fold_),
                                 save_best_only=True,
                                 save_weights_only=False,
                                 mode='max'),
                 #TensorBoard(log_dir="logs/" + expt_name + '/'),
                 #SWA(weight_dir + 'model_swa_{}.hdf5'.format(fold_), 15)
                 ]

    # model training ===================================================================================================
    model = init_model()
    model.compile(loss="categorical_crossentropy", optimizer=Adam(lr=0.001), metrics=["accuracy"])
    # https://github.com/umbertogriffo/focal-loss-keras
    #model.compile(loss = [categorical_focal_loss(alpha=.25, gamma=0)], optimizer=Adam(lr=0.001), metrics=["accuracy"])

    epochs = 30
    model.fit(X_train, y_train,
              validation_data=(X_test, y_test),
              epochs=epochs,
              batch_size=128,
              shuffle=True,
              verbose = 2,
              callbacks=callbacks)

    #===================================================================================================================
    model.load_weights(weight_dir + 'model_{}.hdf5'.format(fold_))
    pred_valid = model.predict(X_test)
    f1_err = f1_score(np.argmax(y_test, axis=1), np.argmax(pred_valid, axis=1), average='macro')
    print('F1 score on validation set:', f1_err)


print('training complete.')
