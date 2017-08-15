import numpy as np
import cv2
import os
from random import shuffle
from tqdm import tqdm


TrainSet = 'C:/Kaggle/train/train'
TestSet = 'C:/Kaggle/test/test'
ImgSize = 50
LearnRate = 1e-3
ModelName = 'DogsCats-{}-{}.model'.format(LearnRate,'6conv') #save model to know what is going on
#load in images. have feataures and labels
#convert label to 2d array

def labelImg(img):
    #dog.93.png so -3 makes it go to dog. is the word label
    word_label = img.split('.')[-3]
    if word_label == 'cat':
        return [1,0]
    elif word_label == 'dog':
        return [0,1]

def createTrainData():        #LABELLED IMAGES. SUPERVISED!
    train_data = []
    for img in tqdm(os.listdir(TrainSet)):
        label = labelImg(img)
        path = os.path.join(TrainSet, img)
        img = cv2.resize(cv2.imread(path, cv2.IMREAD_GRAYSCALE),(ImgSize,ImgSize))
        train_data.append([np.array(img),np.array(label)])
    shuffle(train_data)
    np.save('train_data.py',train_data)
    return train_data

#for testing data
def processTestData():
    test_data = []
    for img in tqdm(os.listdir(TestSet)):
        path = os.path.join(TestSet, img)
        img_ID = img.split('.')[0]  #
        img = cv2.resize(cv2.imread(path, cv2.IMREAD_GRAYSCALE),(ImgSize,ImgSize))
        test_data.append([np.array(img),img_ID])
    np.save('test_data.py', test_data)
    return test_data

train_data = np.load('train_data.py.npy')


import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import tensorflow as tf

tf.reset_default_graph()

convnet = input_data(shape=[None, ImgSize, ImgSize, 1], name='input')

convnet = conv_2d(convnet, 32, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 64, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 32, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 64, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 32, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 64, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = fully_connected(convnet, 1024, activation='relu')
convnet = dropout(convnet, 0.8)

convnet = fully_connected(convnet, 2, activation='softmax')    #2 because dog/cat.
convnet = regression(convnet, optimizer='adam', learning_rate= LearnRate, loss='categorical_crossentropy', name='targets')

model = tflearn.DNN(convnet, tensorboard_dir='log')  #have to do this on windows. Can leave out on linux/ubuntu

if os.path.exists('{}.meta'.format(ModelName)):
    model.load(ModelName)
    print('success')

train = train_data[:-500]  #all but last 500
test = train_data[-500:] #

X = np.array([i[0] for i in train]).reshape(-1, ImgSize, ImgSize, 1) #has data and label. 0 element is image data
Y = [i[1] for i in train]

test_X =np.array([i[0] for i in test]).reshape(-1, ImgSize, ImgSize, 1)
test_Y = [i[1] for i in test]

model.fit({'input': X}, {'targets': Y}, n_epoch=5, validation_set=({'input': test_X}, {'targets': test_Y}),
    snapshot_step=500, show_metric=True, run_id=ModelName)

model.save(ModelName)

import matplotlib.pyplot as plt

test_data = processTestData()
fig = plt.figure()
for num, data in enumerate(test_data[:12]):
    img_num = data[1]
    img_data = data[0]
    y = fig.add_subplot(3,4,num+1)
    orig = img_data
    data = img_data.reshape(ImgSize,ImgSize,1)
    model_out = model.predict([data])[0]
    if np.argmax(model_out) == 1:
        str_label = 'Dog'
    else:
        str_label = 'Cat'
    y.imshow(orig, cmap = 'gray')
    plt.title(str_label)
    y.axes.get_xaxis().set_visible(False)
    y.axes.get_yaxis().set_visible(False)
plt.show()