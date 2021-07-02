import numpy as np
import os
import keras
from keras.layers import Conv2D, Conv3D, Flatten, Dense, Reshape, BatchNormalization
from keras.layers import Dropout, Input, AveragePooling3D, Activation, add, AveragePooling2D
from keras import regularizers
from keras.models import Model
from keras.optimizers import Adam, RMSprop

from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.utils import np_utils
from keras.utils import np_utils
from plotly.offline import init_notebook_mode
init_notebook_mode(connected=True)
PATH = os.getcwd()
print(PATH)

for i in range(7, 8):
    windowSize = 7
    numPCAcomponents = 103
    testRatio = 0.98
    X_train = np.load("./dataset/XtrainWindowSize" + str(windowSize) + "testRatio" + str(testRatio) + ".npy")
    y_train = np.load("./dataset/ytrainWindowSize" + str(windowSize) + "testRatio" + str(testRatio) + ".npy")
    X_test = np.load("./dataset/XtestWindowSize" + str(windowSize) + "testRatio" + str(testRatio) + ".npy")
    y_test = np.load("./dataset/ytestWindowSize" + str(windowSize) + "testRatio" + str(testRatio) + ".npy")


    X_train = X_train.reshape(-1, X_train.shape[1], X_train.shape[2], X_train.shape[3], 1)
    X_test = X_test.reshape(-1, X_test.shape[1], X_test.shape[2], X_test.shape[3], 1)
    y_test = np_utils.to_categorical(y_test)
    y_train = np_utils.to_categorical(y_train)

    input_layer = Input(shape=(7, 7, 103, 1), name='PaviaU_Input')

    conv_layer11 = Conv3D(filters=24, kernel_size=(1, 1, 7), strides=(1, 1, 2))(input_layer)
    BN11 = BatchNormalization()(conv_layer11)
    dp11 = Dropout(rate=0.5)(BN11)
    relu11 = Activation('relu')(dp11)
    #dp11 = Dropout(rate=0.5)(relu11)
    conv_layer12 = Conv3D(filters=24, kernel_size=(1, 1, 7), padding='same')(relu11)
    BN12 = BatchNormalization()(conv_layer12)
    dp12 = Dropout(rate=0.5)(BN12)
    relu12 = Activation('relu')(dp12)
    conv_layer13 = Conv3D(filters=24, kernel_size=(1, 1, 7), padding='same')(relu12)
    BN13 = BatchNormalization()(conv_layer13)
    dp13 = Dropout(rate=0.5)(BN13)
    #dp12 = Dropout(rate=0.5)(relu12)
    relu13 = Activation('relu')(dp13)
    Add11 = add([relu11, relu13])

    conv_layer14 = Conv3D(filters=24, kernel_size=(1, 1, 7), padding='same')(Add11)
    BN14 = BatchNormalization()(conv_layer14)
    dp14 = Dropout(rate=0.5)(BN14)
    relu14 = Activation('relu')(dp14)
    conv_layer15 = Conv3D(filters=24, kernel_size=(1, 1, 7), padding='same')(relu14)
    BN15 = BatchNormalization()(conv_layer15)
    dp15 = Dropout(rate=0.5)(BN15)
    relu15 = Activation('relu')(dp15)
    Add12 = add([Add11, relu15])


    conv_layer = Conv3D(filters=128, kernel_size=(1, 1, 49))(Add12)
    conv3d_shape = conv_layer._keras_shape
    conv_layer3 = Reshape((conv3d_shape[1], conv3d_shape[2], conv3d_shape[3] * conv3d_shape[4], 1))(conv_layer)
    BN = BatchNormalization()(conv_layer3)
    dp = Dropout(rate=0.5)(BN)
    relu = Activation('relu')(dp)
    #dp = Dropout(rate=0.5)(relu)

    #Conv_layer = Reshape((conv_layer3[0], conv_layer3[1], conv_layer3[2], conv_layer3))

    conv_layer21 = Conv3D(filters=24, kernel_size=(3, 3, 128))(relu)
    BN21 = BatchNormalization()(conv_layer21)
    dp21 = Dropout(rate=0.5)(BN21)
    relu21 = Activation('relu')(dp21)
    conv_layer22 = Conv3D(filters=24, kernel_size=(3, 3, 128), padding='same')(relu21)
    BN22 = BatchNormalization()(conv_layer22)
    dp22 = Dropout(rate=0.5)(BN22)
    relu22 = Activation('relu')(dp22)
    #dp22 = Dropout(rate=0.5)(BN22)
    conv_layer23 = Conv3D(filters=24, kernel_size=(3, 3, 128), padding='same')(relu22)
    BN23 = BatchNormalization()(conv_layer23)
    dp23 = Dropout(rate=0.5)(BN23)
    relu23 = Activation('relu')(dp23)
    Add21 = add([relu21, relu23])

    conv_layer24 = Conv3D(filters=24, kernel_size=(3, 3, 128), padding='same')(Add21)
    BN24 = BatchNormalization()(conv_layer24)
    dp24 = Dropout(rate=0.5)(BN24)
    relu24 = Activation('relu')(dp24)
    conv_layer25 = Conv3D(filters=24, kernel_size=(3, 3, 128), padding='same')(relu24)
    BN25 = BatchNormalization()(conv_layer25)
    dp25 = Dropout(rate=0.5)(BN25)
    relu25 = Activation('relu')(dp25)
    Add22 = add([Add21, relu25])


    averagepooling = AveragePooling3D(pool_size=(5, 5, 1))(Add22)
    flatten_layer = Flatten()(averagepooling)
    output_layer = Dense(units=9, activation='softmax')(flatten_layer)

    model = Model(inputs=input_layer, outputs=output_layer)
    model.summary()

    RMSProp1 = RMSprop(lr=0.0003)
    model.compile(loss='categorical_crossentropy', optimizer=RMSProp1, metrics=['accuracy'])

    #earlyStopping = EarlyStopping(monitor='val_loss', patience=100, verbose=1, mode='auto')

    filepath = 'SSRN测试-3D' + '尺寸' +str(windowSize) + '测试集占比' + str(testRatio) + str(i) + '.hdf5'
    checkpoint = ModelCheckpoint(filepath, monitor='accuracy', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]
    history = model.fit(x=X_train, y=y_train, verbose=1, batch_size=16, epochs=200, callbacks=callbacks_list)






