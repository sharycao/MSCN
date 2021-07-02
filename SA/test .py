from sklearn.decomposition import PCA
import os
import scipy.io as sio
import numpy as np
from keras.models import load_model
from keras.utils import np_utils
from sklearn.metrics import classification_report, confusion_matrix
import itertools

import spectral
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, cohen_kappa_score
from operator import truediv

windowSize = 9
numPCAcomponents = 20
testRatio = 0.99

numComponents = 20

for i in range(1, 11):
    def loadIndianPinesData():
        data_path = os.path.join(os.getcwd(), 'data')
        data = sio.loadmat(os.path.join(data_path, 'Salinas_corrected.mat'))['salinas_corrected']
        labels = sio.loadmat(os.path.join(data_path, 'Salinas_gt.mat'))['salinas_gt']
        return data, labels

    def AA_andEachClassAccuracy(confusion_matrix):
        counter = confusion_matrix.shape[0]
        list_diag = np.diag(confusion_matrix)
        list_raw_sum = np.sum(confusion_matrix, axis=1)
        each_acc = np.nan_to_num(truediv(list_diag, list_raw_sum))
        average_acc = np.mean(each_acc)
        return each_acc, average_acc

    def reports (X_test,y_test):

        Y_pred = model.predict(X_test)
        y_pred = np.argmax(Y_pred, axis=1)
        target_names = ['Brocoli_green_weeds_1 ', 'Brocoli_green_weeds_2 ', 'Fallow', 'Fallow_rough_plow',
                        'Fallow_smooth', 'Stubble', 'Celery', 'Grapes_untrained', 'Soil_vinyard_develop',
                        'Corn_senesced_green_weeds ', 'Lettuce_romaine_4wk', 'Lettuce_romaine_5wk',
                        'Lettuce_romaine_6wk', 'Lettuce_romaine_7wk', 'Vinyard_untrained ',
                        'Vinyard_vertical_trellis']

        classification = classification_report(np.argmax(y_test, axis=1), y_pred, target_names=target_names)
        oa = accuracy_score(np.argmax(y_test, axis=1), y_pred)
        confusion = confusion_matrix(np.argmax(y_test, axis=1), y_pred)
        each_acc, aa = AA_andEachClassAccuracy(confusion)
        kappa = cohen_kappa_score(np.argmax(y_test, axis=1), y_pred)
        score = model.evaluate(X_test, y_test, batch_size=32)
        Test_Loss = score[0] * 100
        Test_accuracy = score[1] * 100

        return classification, confusion, Test_Loss, Test_accuracy, oa*100, each_acc*100, aa*100, kappa*100

    def applyPCA(X, numComponents=75):
        newX = np.reshape(X, (-1, X.shape[2]))
        pca = PCA(n_components=numComponents, whiten=True)
        newX = pca.fit_transform(newX)
        newX = np.reshape(newX, (X.shape[0],X.shape[1], numComponents))
        return newX, pca


    def Patch(data, height_index, width_index):
        # transpose_array = data.transpose((2,0,1))
        # print transpose_array.shape
        height_slice = slice(height_index, height_index + PATCH_SIZE)
        width_slice = slice(width_index, width_index + PATCH_SIZE)
        patch = data[height_slice, width_slice, :]
        return patch

    def padWithZeros(X, margin=2):
        newX = np.zeros((X.shape[0] + 2 * margin, X.shape[1] + 2 * margin, X.shape[2]))
        x_offset = margin
        y_offset = margin
        newX[x_offset:X.shape[0] + x_offset, y_offset:X.shape[1] + y_offset, :] = X
        return newX

    X_test = np.load("./dataset/XtestWindowSize" + str(windowSize) + "PCA" + str(numPCAcomponents) + "testRatio" + str(testRatio) + ".npy")
    y_test = np.load("./dataset/ytestWindowSize" + str(windowSize) + "PCA" + str(numPCAcomponents) + "testRatio" + str(testRatio) + ".npy")

    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], X_test.shape[2], X_test.shape[3], 1))
    y_test = np_utils.to_categorical(y_test)

    #model = load_model('池化+信息融合' + '尺寸' + str(windowSize) + '测试集占比' + str(testRatio) + str(i) + '.hdf5')
    model = load_model('TKSN-9x9x20 - 1%' + "Window Size" + str(windowSize) + "num" + str(numPCAcomponents)\
               + str(i) + '.hdf5')

    classification, confusion, Test_loss, Test_accuracy, oa, each_acc, aa, kappa = reports(X_test, y_test)
    classification = str(classification)
    confusion = str(confusion)
    file_name = 'TKSN-9x9x20 - 1%' + "Window Size" + str(windowSize) + "num" + str(numPCAcomponents)\
               + str(i) + '.txt'
    with open(file_name, 'w') as x_file:
        x_file.write('{} Test loss (%)'.format(Test_loss))
        x_file.write('\n')
        x_file.write('{} Test accuracy (%)'.format(Test_accuracy))
        x_file.write('\n')
        x_file.write('\n')
        x_file.write('{} Kappa accuracy (%)'.format(kappa))
        x_file.write('\n')
        x_file.write('{} Overall accuracy (%)'.format(oa))
        x_file.write('\n')
        x_file.write('{} Average accuracy (%)'.format(aa))
        x_file.write('\n')
        x_file.write('{} each_acc (%)'.format(each_acc))
        x_file.write('\n')
        x_file.write('\n')
        x_file.write('{}'.format(classification))
        x_file.write('\n')
        x_file.write('{}'.format(confusion))
'''
    X, y = loadIndianPinesData()

    X,pca = applyPCA(X,numComponents=numComponents)

    height = y.shape[0]
    width = y.shape[1]
    PATCH_SIZE = 25

    X = padWithZeros(X, PATCH_SIZE//2)

    outputs = np.zeros((height, width))
    for i in range(height):
       for j in range(width):
            target = int(y[i, j])
            if target == 0:
                continue
            else:
                image_patch = Patch(X, i, j)
                #print (image_patch.shape)
                X_test_image = image_patch.reshape(1, image_patch.shape[0], image_patch.shape[1], image_patch.shape[2], 1).astype('float32')
                prediction = (model.predict(X_test_image))
                prediction = np.argmax(prediction, axis=1)
                outputs[i][j] = prediction+1

    ground_truth = spectral.imshow(classes = y, figsize =(15,15))

    predict_image = spectral.imshow(classes = outputs.astype(int), figsize =(15,15))

    spectral.save_rgb("TKRN-SA.jpg", outputs.astype(int), colors=spectral.spy_colors)

'''
