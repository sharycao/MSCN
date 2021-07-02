import numpy as np
import os
import keras
import time
from keras.layers import Conv3D, MaxPooling2D, Deconvolution2D, Flatten, Dense, add, BatchNormalization
from keras.layers import Input, UpSampling2D, Activation, Concatenate, Conv2DTranspose, Dropout
from keras.models import Model
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, cohen_kappa_score
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from keras.optimizers import Adam, SGD
from sklearn import preprocessing
import matplotlib.pyplot as plt
#from keras import backend as K
#K.set_image_dim_ordering('th')
from operator import truediv
import spectral
from keras.utils import np_utils
from plotly.offline import init_notebook_mode
init_notebook_mode(connected=True)
PATH = os.getcwd()
print(PATH)

for i in range(1, 15):
    windowSize = 5
    numPCAcomponents = 103
    testRatio = 0.98

    X_train = np.load("./dataset/XtrainWindowSize"
                  + str(windowSize) + "PCA" + str(numPCAcomponents) + "testRatio" + str(testRatio)  + ".npy")
    y_train = np.load("./dataset/ytrainWindowSize"
                  + str(windowSize) + "PCA" + str(numPCAcomponents) + "testRatio" + str(testRatio) + ".npy")

    X_test = np.load("./dataset/XtrainWindowSize"
                  + str(windowSize) + "PCA" + str(numPCAcomponents) + "testRatio" + str(testRatio)  + ".npy")
    y_test = np.load("./dataset/ytrainWindowSize"
                  + str(windowSize) + "PCA" + str(numPCAcomponents) + "testRatio" + str(testRatio) + ".npy")



    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], X_train.shape[2], X_train.shape[3], 1))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], X_test.shape[2], X_test.shape[3], 1 ))

    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)


    input_layer = Input(shape=(5, 5, 103, 1), name='PaviaU_Input')

#第一层（27X27X128   变为   27X27X288)
    Conv1 = Conv3D(filters=2, kernel_size=(3,3,7))(input_layer)
    BN1= BatchNormalization()(Conv1)
    Relu1 = Activation('relu')(BN1)
    Conv2 = Conv3D(filters=4, kernel_size=(3,3,3))(Conv1)
    BN2 = BatchNormalization()(Conv2)
    Relu2 = Activation('relu')(BN2)

    flatten_layer = Flatten()(Relu2)
    F1 = Dense(units=144, activation='relu')(flatten_layer)
    Dropout1 = Dropout(rate=0.5)(F1)


    output_layer = Dense(units=9, activation='softmax')(Dropout1)



#全连接层
    model = Model(inputs=[input_layer], outputs=[output_layer])
    model.summary()

#反向传播
    #opt = Adam(lr=0.0003)
    SGD = keras.optimizers.SGD(lr=0.1, momentum=0.9, decay=0.0005)
    model.compile(loss='categorical_crossentropy', optimizer=SGD, metrics=['accuracy'])

#检查

    filepath = '3DCNN-windowSize' + str(windowSize) + 'PCA' + str(numPCAcomponents) + "testRatio" + str(testRatio) + str(i) +'.h5'
    #reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', factor=0.1, patience=1, min_lr=0.00001, verbose=1)
    checkpoint = ModelCheckpoint(filepath, monitor='accuracy', verbose=1, save_best_only=True, mode='max')
    #tensorboard = TensorBoard()
    callbacks_list = [checkpoint]
    start = time.time()
    history = model.fit(x=X_train, y=y_train, verbose=1, batch_size=128, epochs=1000,
                        callbacks=callbacks_list)
    end = time.time()
    print(start - end)


    #plt.plot(history.history['acc'])
    #plt.plot(history.history['val_acc'])
    #plt.title('model accuracy')
    #plt.ylabel('accuracy')
    #plt.xlabel('epoch')
    #plt.legend(['train', 'test'], loc='upper left')
    #plt.savefig(str(i)+'acc0.001'+".jpg")
    #plt.show()


    #plt.xlim((0, 30))

    #plt.ylim((0, 1.6))
    #my_x_ticks = np.arange(0, 30, 5)
    #my_y_ticks = np.arange(0, 1.6, 0.15)
    #plt.xticks(my_x_ticks)
    #plt.yticks(my_y_ticks)
    #plt.plot(history.history['loss'])
    #plt.plot(history.history['val_loss'])
    #plt.title('model loss')
    #plt.ylabel('loss')
    #plt.xlabel('epoch')
    #plt.grid(True)
    #plt.legend(['train', 'test'], loc='upper left')
    #plt.savefig(str(i)+'loss0.01'+".jpg")
    #plt.show()









