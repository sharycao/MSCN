import numpy as np
import os
import keras
from keras.layers import Conv2D, MaxPooling2D, Deconvolution2D, Flatten, Dense, add, BatchNormalization, AveragePooling2D
from keras.layers import Input, UpSampling2D, Activation, Concatenate, Conv2DTranspose, Dropout, GlobalAvgPool2D
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
from plotly.offline import init_notebook_mode
from keras.optimizers import Adam
import time
init_notebook_mode(connected=True)
PATH = os.getcwd()
print(PATH)

for i in range(1, 11):
    windowSize = 15
    numPCAcomponents = 15
    testRatio = 0.995

    X_train = np.load("./dataset/XtrainWindowSize" + str(windowSize) + "testRatio" + str(testRatio) + ".npy")
    y_train = np.load("./dataset/ytrainWindowSize" + str(windowSize) + "testRatio" + str(testRatio) + ".npy")



    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], X_train.shape[2], X_train.shape[3]))


    y_train = np_utils.to_categorical(y_train)



    input_layer = Input(shape=(15, 15, 15), name='PaviaU_Input')
#8
    #Conv1 = Conv2D(filters=8, kernel_size=(1, 1), padding='same')(input_layer)
    Conv11 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', kernel_initializer='he_normal')(input_layer)
    BN11 = BatchNormalization()(Conv11)
    relu11 = Activation('relu')(BN11)
    Conv12 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', kernel_initializer='he_normal')(relu11)
    BN12 = BatchNormalization()(Conv12)
    Add11 = add([Conv11, BN12])
    relu12 = Activation('relu')(Add11)

    Conv21 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', kernel_initializer='he_normal')(relu12)
    BN21 = BatchNormalization()(Conv21)
    relu21 = Activation('relu')(BN21)
    Conv22 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', kernel_initializer='he_normal')(relu21)
    BN22 = BatchNormalization()(Conv22)
    Add21 = add([relu12, BN22])
    relu22 = Activation('relu')(Add21)

#16
    #Conv2 = Conv2D(filters=16, kernel_size=(1, 1), padding='same')(relu22)
    #mp2 = MaxPooling2D(pool_size=(2,2))(Conv2)
    Conv31 = Conv2D(filters=128, kernel_size=(3, 3), strides=(2, 2), kernel_initializer='he_normal')(relu22)
    BN31 = BatchNormalization()(Conv31)
    relu31 = Activation('relu')(BN31)
    Conv32 = Conv2D(filters=128, kernel_size=(3, 3), padding='same', kernel_initializer='he_normal')(relu31)
    BN32 = BatchNormalization()(Conv32)
    Add31 = add([Conv31, BN32])
    relu32 = Activation('relu')(Add31)

    Conv41 = Conv2D(filters=128, kernel_size=(3, 3), padding='same', kernel_initializer='he_normal')(relu32)
    BN41 = BatchNormalization()(Conv41)
    relu41 = Activation('relu')(BN41)
    Conv42 = Conv2D(filters=128, kernel_size=(3, 3), padding='same', kernel_initializer='he_normal')(relu41)
    BN42 = BatchNormalization()(Conv42)
    Add41 = add([relu32, BN42])
    relu42 = Activation('relu')(Add41)

# 32
    #Conv3 = Conv2D(filters=32, kernel_size=(1, 1), padding='same')(relu42)
    #mp3 = MaxPooling2D(pool_size=(2,2))(Conv3)
    Conv51 = Conv2D(filters=256, kernel_size=(3, 3), strides=(2, 2), kernel_initializer='he_normal')(relu42)
    BN51 = BatchNormalization()(Conv51)
    relu51 = Activation('relu')(BN51)
    Conv52 = Conv2D(filters=256, kernel_size=(3, 3), padding='same', kernel_initializer='he_normal')(relu51)
    BN52 = BatchNormalization()(Conv52)
    Add51 = add([Conv51, BN52])
    relu52 = Activation('relu')(Add51)

    Conv61 = Conv2D(filters=256, kernel_size=(3, 3), padding='same', kernel_initializer='he_normal')(relu52)
    BN61 = BatchNormalization()(Conv61)
    relu61 = Activation('relu')(BN61)
    Conv62 = Conv2D(filters=256, kernel_size=(3, 3), padding='same', kernel_initializer='he_normal')(relu61)
    BN62 = BatchNormalization()(Conv62)
    Add61 = add([relu52, BN62])
    relu62 = Activation('relu')(Add61)

# 512
    #Conv4 = Conv2D(filters=64, kernel_size=(1, 1), padding='same')(relu62)
    #mp4 = MaxPooling2D(pool_size=(2, 2))(Conv4)
    Conv71 = Conv2D(filters=512, kernel_size=(3, 3), strides=(2, 2), kernel_initializer='he_normal')(relu62)
    BN71 = BatchNormalization()(Conv71)
    relu71 = Activation('relu')(BN71)
    Conv72 = Conv2D(filters=512, kernel_size=(3, 3), padding='same', kernel_initializer='he_normal')(relu71)
    BN72 = BatchNormalization()(Conv72)
    Add71 = add([Conv71, BN72])
    relu72 = Activation('relu')(Add71)

    Conv81 = Conv2D(filters=512, kernel_size=(3, 3), padding='same', kernel_initializer='he_normal')(relu72)
    BN81 = BatchNormalization()(Conv81)
    relu81 = Activation('relu')(BN81)
    Conv82 = Conv2D(filters=512, kernel_size=(3, 3), padding='same', kernel_initializer='he_normal')(relu81)
    BN82 = BatchNormalization()(Conv82)
    Add81 = add([relu72, BN82])
    relu82 = Activation('relu')(Add81)

    AveragePooling = AveragePooling2D(pool_size=(1, 1))(relu82)
    flatten_layer = Flatten()(AveragePooling)

    output_layer = Dense(units=9, activation='softmax')(flatten_layer)

    model = Model(inputs=[input_layer], outputs=[output_layer])
    model.summary()
#反向传播
    opt = Adam(learning_rate=0.001)
    #SGD = keras.optimizers.SGD(lr=0.1, decay=0.0001, momentum=0.9)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    filepath = 'DRN+windowsize' + str(windowSize) + "PCA" + str(numPCAcomponents) \
               + "testRatio" + str(testRatio) + str(i) + '.hdf5'
    checkpoint = ModelCheckpoint(filepath, monitor='accuracy', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]
    start = time.time()
    history = model.fit(x=X_train, y=y_train, verbose=1, batch_size=16, epochs=50,callbacks=callbacks_list)
    end = time.time()
    time1 = start - end
    print(time1)