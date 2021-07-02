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
init_notebook_mode(connected=True)
PATH = os.getcwd()
print(PATH)

for i in range(4,5):
    windowSize = 5
    numPCAcomponents = 103
    testRatio = 0.97
    X_train = np.load("./dataset/XtrainWindowSize" + str(windowSize)+ "testRatio" + str(testRatio) + ".npy")
    y_train = np.load("./dataset/ytrainWindowSize" + str(windowSize)+ "testRatio" + str(testRatio) + ".npy")

    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], X_train.shape[2], X_train.shape[3]))
    #X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], X_test.shape[2], X_test.shape[3]))

    y_train = np_utils.to_categorical(y_train)
    #y_test = np_utils.to_categorical(y_test)

    input_layer = Input(shape=(5, 5, 103), name='Indian_Pines_Input')
    Conv11 = Conv2D(filters=128, kernel_size=(5, 5))(input_layer)
    #BN11 = BatchNormalization()(Conv11)
    #relu11 = Activation('relu')(BN11)
    #maxpooling11 = MaxPooling2D(pool_size=(5, 5))(Conv11)
    Conv12 = Conv2D(filters=128, kernel_size=(3, 3))(input_layer)
    #BN12 = BatchNormalization()(Conv12)
    #relu12 = Activation('relu')(BN12)
    maxpooling11 = MaxPooling2D(pool_size=(3,3))(Conv12)
    Conv13 = Conv2D(filters=128, kernel_size=(1, 1))(input_layer)
    #BN13 = BatchNormalization()(Conv13)
    #relu13 = Activation('relu')(BN13)
    maxpooling12 = MaxPooling2D(pool_size=(5, 5))(Conv13)
    Conc1 = concatenate([maxpooling11, maxpooling12, maxpooling12])
    BN1 = BatchNormalization()(Conc1)
    Relu11 = Activation('relu')(BN1)


    conv2 = Conv2D(filters=128, kernel_size=(1,1))(Relu11)
    BN2 = BatchNormalization()(conv2)
    relu2 = Activation('relu')(BN2)
    conv3 = Conv2D(filters=128, kernel_size=(1,1),activation='relu')(relu2)
    conv4 = Conv2D(filters=128, kernel_size=(1,1))(conv3)
    add1 = add([relu2,conv4])
    relu_add = Activation('relu')(add1)

    conv5 = Conv2D(filters=128, kernel_size=(1,1),activation='relu')(relu_add)
    conv6 = Conv2D(filters=128, kernel_size=(1,1))(conv5)
    add2 = add([relu_add, conv6])
    relu_add2 = Activation('relu')(add2)

    conv7 = Conv2D(filters=128, kernel_size=(1,1))(relu_add2)
    dropout1 = Dropout(rate=0.5)(conv7)
    relu1 = Activation('relu')(dropout1)
    conv8 = Conv2D(filters=128, kernel_size=(1, 1))(relu1)
    dropout2 = Dropout(rate=0.5)(conv8)
    relu2 = Activation('relu')(dropout2)
    conv9 = Conv2D(filters=128, kernel_size=(1, 1))(relu2)

    flatten_layer = Flatten()(conv9)
    output_layer = Dense(units=9, activation='softmax')(flatten_layer)





    model = Model(inputs=[input_layer], outputs=[output_layer])
    model.summary()

    #opt = Adam(lr=0.003)
    SGD = keras.optimizers.SGD(lr=0.01, momentum=0.9)
    model.compile(loss='categorical_crossentropy', optimizer=SGD, metrics=['accuracy'])

    filepath = 'DCNN-16' + '尺寸' + str(windowSize) + '测试集占比' + str(testRatio) + str(i) + '.hdf5'
    checkpoint = ModelCheckpoint(filepath, monitor='accuracy', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]
    history = model.fit(x=X_train, y=y_train, verbose=1, batch_size=16, epochs=300,callbacks=callbacks_list)

