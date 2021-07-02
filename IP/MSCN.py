import numpy as np
import os
import keras
from keras.layers import Conv3D,  Deconvolution3D, Flatten, Dense, add, BatchNormalization
from keras.layers import Input,  Activation, Concatenate, Dropout, MaxPooling3D, Reshape
from keras.models import Model
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, cohen_kappa_score
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam, SGD
from sklearn import preprocessing
import matplotlib.pyplot as plt
#from keras import backend as K
#K.set_image_dim_ordering('th')
from operator import truediv
import spectral
from keras.utils import np_utils
import time
from plotly.offline import init_notebook_mode
init_notebook_mode(connected=True)
PATH = os.getcwd()
print(PATH)

for i in range(1, 11):
    windowSize = 25
    numPCAcomponents = 25
    testRatio = 0.70

    #X_train = np.load("./dataset/XtrainWindowSize" + str(windowSize) + "testRatio" + str(testRatio) + ".npy")
    #y_train = np.load("./dataset/ytrainWindowSize" + str(windowSize) + "testRatio" + str(testRatio) + ".npy")
    X_train = np.load("./dataset/XtrainWindowSize" + str(windowSize) + "PCA" + str(numPCAcomponents) + "testRatio" + str(testRatio) + ".npy")
    y_train = np.load("./dataset/ytrainWindowSize" + str(windowSize) + "PCA" + str(numPCAcomponents) + "testRatio" + str(testRatio) + ".npy")

    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], X_train.shape[2], X_train.shape[3], 1))
    y_train = np_utils.to_categorical(y_train)


    input_layer = Input(shape=(25, 25, 25, 1), name='Indian_Pines_Input')



#第一层
    #光谱
    conv_layer11 = Conv3D(filters=8, kernel_size=(1, 1, 3),  padding='same')(input_layer)
    BN11 = BatchNormalization()(conv_layer11)
    relu11 = Activation('relu')(BN11)

    pool1 = MaxPooling3D(pool_size=(3, 3, 1), strides=(1, 1, 1), padding='same')(relu11)

    conv_layer12 = Conv3D(filters=8, kernel_size=(3, 3, 1), padding='same')(pool1)
    BN12 = BatchNormalization()(conv_layer12)
    relu12 = Activation('relu')(BN12)

    conv_layer13 = Conv3D(filters=16, kernel_size=(3, 3, 25))(relu12)
    conv3d_shape13 = conv_layer13._keras_shape
    cl13 = Reshape((conv3d_shape13[1], conv3d_shape13[2], conv3d_shape13[3] * conv3d_shape13[4], 1))(conv_layer13)
    BN13 = BatchNormalization()(cl13)
    relu13 = Activation('relu')(BN13)




    flatten_layer = Flatten()(relu13)


#全连接层
    #output_layer1 = GlobalAveragePooling3D()(flatten_layer)
    Dense1 = Dense(256, activation='relu')(flatten_layer)
    dropout1 = Dropout(rate=0.3)(Dense1)
    Dense2 = Dense(128, activation='relu')(dropout1)
    dropout2 = Dropout(rate=0.3)(Dense2)
    output_layer = Dense(16, activation='softmax')(dropout2)

    model = Model(inputs=[input_layer], outputs=[output_layer])
    model.summary()

#反向传播
    SGD = keras.optimizers.SGD(lr=0.01)
    #Adam = keras.optimizers.Adam(lr=0.0001)
    model.compile(loss='categorical_crossentropy', optimizer=SGD, metrics=['accuracy'])


    #检查

    filepath = 'TKSN-25x25x25-0.70' + "Window Size" + str(windowSize) + "num" + str(numPCAcomponents)\
               + str(i) + '.hdf5'
    #reduce_lr = ReduceLROnPlateau(monitor=*'val_acc', factor=0.5, patience=2, min_lr=0.0001, verbose=1)
    checkpoint = ModelCheckpoint(filepath, monitor='accuracy', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]
    start = time.time()
    history = model.fit(x=X_train, y=y_train, verbose=1, batch_size=16, epochs=50,
                        callbacks=callbacks_list)
    end = time.time()
    print(start - end)
'''
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(str(i)+'acc0.001'+".jpg")
    plt.show()


    plt.xlim((0, 30))

    plt.ylim((0, 1.6))
    my_x_ticks = np.arange(0, 30, 5)
    my_y_ticks = np.arange(0, 1.6, 0.15)
    plt.xticks(my_x_ticks)
    plt.yticks(my_y_ticks)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.grid(True)
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(str(i)+'loss0.01'+".jpg")
    plt.show()
'''








