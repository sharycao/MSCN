import numpy as np
import os
import keras
from keras.layers import Conv2D, MaxPooling2D, Deconvolution2D, Flatten, Dense, add, BatchNormalization, concatenate
from keras.layers import Input, UpSampling2D, Activation, Concatenate, Conv2DTranspose, Dropout, GlobalAvgPool2D
from keras.models import Model
from keras import initializers
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
from plotly.offline import init_notebook_mode
from keras.optimizers import Adam
import time
init_notebook_mode(connected=True)
PATH = os.getcwd()
print(PATH)

for i in range(1,11):
    windowSize = 27
    numPCAcomponents = 1
    testRatio = 0.955
    X_train = np.load("./dataset/XtrainWindowSize" + str(windowSize) + "PCA" + str(numPCAcomponents) +
                      "testRatio" + str(testRatio) + ".npy")
    y_train = np.load("./dataset/ytrainWindowSize" + str(windowSize) + "PCA" + str(numPCAcomponents) +
                      "testRatio" + str(testRatio) + ".npy")

    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], X_train.shape[2], X_train.shape[3]))
    #X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], X_test.shape[2], X_test.shape[3]))

    y_train = np_utils.to_categorical(y_train)
    #y_test = np_utils.to_categorical(y_test)

    input_layer = Input(shape=(27, 27, 1), name='PaviaU_Input')
    Conv11 = Conv2D(filters=32, kernel_size=(4, 4))(input_layer)
    BN11 = BatchNormalization()(Conv11)
    relu11 = Activation('relu')(BN11)
    maxpooling11 = MaxPooling2D(pool_size=(2, 2))(relu11)

    Conv12 = Conv2D(filters=64, kernel_size=(5, 5))(maxpooling11)
    BN12 = BatchNormalization()(Conv12)
    relu12 = Activation('relu')(BN12)
    dropout1 = Dropout(rate=0.5)(relu12)
    maxpooling11 = MaxPooling2D(pool_size=(2,2))(dropout1)

    Conv13 = Conv2D(filters=128, kernel_size=(4, 4))(maxpooling11)
    BN13 = BatchNormalization()(Conv13)
    relu13 = Activation('relu')(BN13)

    flatten_layer = Flatten()(relu13)
    output_layer = Dense(units=9, activation='softmax')(flatten_layer)





    model = Model(inputs=[input_layer], outputs=[output_layer])
    model.summary()

    #opt = Adam(lr=0.003)
    SGD = keras.optimizers.SGD(lr=0.01)
    model.compile(loss='categorical_crossentropy', optimizer=SGD, metrics=['accuracy'])

    filepath = '2D-CNN-LR' + '尺寸' + str(windowSize) + '测试集占比' + str(testRatio) + str(i) + '.hdf5'
    checkpoint = ModelCheckpoint(filepath, monitor='accuracy', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]
    start = time.time()
    history = model.fit(x=X_train, y=y_train, verbose=1, batch_size=100, epochs=200,callbacks=callbacks_list)
    end = time.time()
    run_time = (start - end)
    print(run_time)