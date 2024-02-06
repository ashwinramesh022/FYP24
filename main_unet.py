import PIL
from PIL import Image
import matplotlib.pyplot as plt
import tifffile as tiff
from skimage.transform import resize
import numpy as np
import math
import glob
import cv2
import os
import skimage.io as io
import skimage.transform as trans
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as keras
import re
import tensorflow as tf
from tensorflow.keras.metrics import MeanIoU
from keras import backend as K

def iou(y_true, y_pred, smooth=100):
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    union = K.sum(y_true, -1) + K.sum(y_pred, -1) - intersection
    iou_acc = (intersection + smooth) / (union + smooth)
    return iou_acc

def as_keras_metric(method):
    import functools
    from keras import backend as K
    import tensorflow as tf
    @functools.wraps(method)
    def wrapper(self, args, **kwargs):
        """ Wrapper for turning tensorflow metrics into keras metrics """
        value, update_op = method(self, args, **kwargs)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([update_op]):
            value = tf.identity(value)
        return value
    return wrapper

def tf_mean_iou(y_true, y_pred, num_classes=6):
    return tf.metrics.mean_iou(y_true, y_pred, num_classes)

mean_iou = as_keras_metric(tf_mean_iou)

# To read the images in numerical order
import re
numbers = re.compile(r'(\d+)')
def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

def padding(img, w, h, c, crop_size, stride, n_h, n_w):
    
    w_extra = w - ((n_w-1)*stride)
    w_toadd = crop_size - w_extra
    
    h_extra = h - ((n_h-1)*stride)
    h_toadd = crop_size - h_extra
    
    img_pad = np.zeros(((h+h_toadd), (w+w_toadd), c))
    #img_pad[:h, :w,:] = img
    #img_pad = img_pad+img
    img_pad = np.pad(img, [(0, h_toadd), (0, w_toadd), (0,0)], mode='constant')
    
    return img_pad

mean_iou = as_keras_metric(tf_mean_iou)

def crops(a, crop_size = 128):
    
    stride = int(crop_size/2)
    stride = 32

    croped_images = []
    h, w, c = a.shape
    
    n_h = int(int(h/stride))
    n_w = int(int(w/stride))
    
    # Padding using the padding function we wrote
    a = padding(a, w, h, c, crop_size, stride, n_h, n_w)
    
    #Resizing as required
    #a = resized(a, stride, n_h, n_w)
    
    # Adding pixals as required
    #a = add_pixals(a, h, w, c, n_h, n_w, crop_size, stride)
    
    # Slicing the image into 128*128 crops with a stride of 64
    for i in range(n_h-1):
        for j in range(n_w-1):
            crop_x = a[(i*stride):((i*stride)+crop_size), (j*stride):((j*stride)+crop_size), :]
            croped_images.append(crop_x)
    return croped_images

filelist_trainx = sorted(glob.glob('eye-in-the-sky\Inter-IIT-CSRE\The-Eye-in-the-Sky-dataset\sat\*.tif'), key=numericalSort)
filelist_trainy = sorted(glob.glob('eye-in-the-sky\Inter-IIT-CSRE\The-Eye-in-the-Sky-dataset\gt\*.tif'), key=numericalSort)
filelist_testx = sorted(glob.glob('eye-in-the-sky\Inter-IIT-CSRE\The-Eye-in-the-Sky-test-data\sat_test\*.tif'), key=numericalSort)

print(len(filelist_testx),len(filelist_trainx),len(filelist_trainy))
trainx_list = []

for fname in filelist_trainx[:13]:
    image = tiff.imread(fname)
    crops_list = crops(image)
    trainx_list = trainx_list + crops_list

trainx = np.asarray(trainx_list)

trainy_list = []

for fname in filelist_trainy[:13]:
    image = tiff.imread(fname)
    crops_list = crops(image)
    trainy_list = trainy_list + crops_list

trainy = np.asarray(trainy_list)

testx_list = []

tif = tiff.TiffFile(filelist_trainx[13])
image = tif.asarray()
crops_list = crops(image)
testx_list = testx_list + crops_list

testx = np.asarray(testx_list)

testy_list = []

tif = tiff.TiffFile(filelist_trainy[13])
image = tif.asarray()
crops_list = crops(image)
testy_list = testy_list + crops_list

testy = np.asarray(testy_list)

xtrain_list = []


for fname in filelist_trainx:
    tif = tiff.TiffFile(fname)
    image = tif.asarray()
    crop_size = 128
    stride = 64
    h, w, c = image.shape
    n_h = int(int(h/stride))
    n_w = int(int(w/stride))
    image = padding(image, w, h, c, crop_size, stride, n_h, n_w)
    xtrain_list.append(image)

# Assuming target shape is (128, 128, 4), adjust it according to your needs
target_shape = (128, 128, 4)

# Resizing each image in xtrain_list to the target shape
resized_xtrain_list = [resize(image, target_shape) for image in xtrain_list]

# Convert the resized list to a NumPy array
x_train = np.asarray(resized_xtrain_list)

ytrain_list = []

for fname in filelist_trainy:
    tif = tiff.TiffFile(fname)
    image = tif.asarray()
    crop_size = 128
    stride = 64
    h, w, c = image.shape
    n_h = int(int(h/stride))
    n_w = int(int(w/stride))
    image = padding(image, w, h, c, crop_size, stride, n_h, n_w)
    ytrain_list.append(image)

# Assuming target shape is (128, 128, 4), adjust it according to your needs
target_shape = (128, 128, 4)

# Resizing each image in xtrain_list to the target shape
resized_ytrain_list = [resize(image, target_shape) for image in ytrain_list]
y_train = np.asarray(resized_ytrain_list)

def unet(shape=(None, None, 4)):
    inputs = Input(shape)
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='random_normal')(inputs)
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='random_normal')(conv1)
    conv1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='random_normal')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='random_normal')(conv2)
    conv2 = BatchNormalization()(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='random_normal')(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='random_normal')(conv3)
    conv3 = BatchNormalization()(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='random_normal')(pool3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='random_normal')(conv4)
    conv4 = BatchNormalization()(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='random_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='random_normal')(conv5)
    conv5 = BatchNormalization()(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='random_normal')(
        UpSampling2D(size=(2, 2))(drop5))
    merge6 = concatenate([drop4, up6], axis=3)
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='random_normal')(merge6)
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='random_normal')(conv6)
    conv6 = BatchNormalization()(conv6)

    up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='random_normal')(
        UpSampling2D(size=(2, 2))(conv6))
    merge7 = concatenate([conv3, up7], axis=3)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='random_normal')(merge7)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='random_normal')(conv7)
    conv7 = BatchNormalization()(conv7)

    up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='random_normal')(
        UpSampling2D(size=(2, 2))(conv7))
    merge8 = concatenate([conv2, up8], axis=3)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='random_normal')(merge8)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='random_normal')(conv8)
    conv8 = BatchNormalization()(conv8)

    up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='random_normal')(
        UpSampling2D(size=(2, 2))(conv8))
    merge9 = concatenate([conv1, up9], axis=3)
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='random_normal')(merge9)
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='random_normal')(conv9)
    conv9 = Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='random_normal')(conv9)
    conv9 = BatchNormalization()(conv9)

    conv10 = Conv2D(4, 1, activation='softmax')(conv9)

    model = Model(inputs=inputs, outputs=conv10)

    model.compile(optimizer=Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])


    model.summary()

    return model

model = unet()

model.fit(x_train, y_train, batch_size=2, epochs=100, validation_split=0.1)




color_dict = {0: (0, 0, 0),
              1: (0, 125, 0),
              2: (150, 80, 0),
              3: (255, 255, 0),
              4: (100, 100, 100),
              5: (0, 255, 0),
              6: (0, 0, 150),
              7: (150, 150, 255),
              8: (255, 255, 255)}

def rgb_to_onehot(rgb_arr, color_dict):
    num_classes = len(color_dict)
    shape = rgb_arr.shape[:2]+(num_classes,)
    #print(shape)
    arr = np.zeros( shape, dtype=np.int8 )
    for i, cls in enumerate(color_dict):
        arr[:,:,i] = np.all(rgb_arr.reshape( (-1,3) ) == color_dict[i], axis=1).reshape(shape[:2])
    return arr

def onehot_to_rgb(onehot, color_dict):
    single_layer = np.argmax(onehot, axis=-1)
    output = np.zeros( onehot.shape[:2]+(3,) )
    for k in color_dict.keys():
        output[single_layer==k] = color_dict[k]
    return np.uint8(output)


# Convert trainy and testy into one hot encode

trainy_hot = []

for i in range(trainy.shape[0]):
    
    hot_img = rgb_to_onehot(trainy[i], color_dict)
    
    trainy_hot.append(hot_img)
    
trainy_hot = np.asarray(trainy_hot)

testy_hot = []

for i in range(testy.shape[0]):
    
    hot_img = rgb_to_onehot(testy[i], color_dict)
    
    testy_hot.append(hot_img)
    
testy_hot = np.asarray(testy_hot)


'''#trainx = trainx/np.max(trainx)
trainy = trainy/np.max(trainy)

#testx = testx/np.max(testx)
testy = testy/np.max(testy)

# Data Augmentation

datagen_args = dict(rotation_range=45.,
                         width_shift_range=0.1,
                         height_shift_range=0.1,
                         shear_range=0.2,
                         zoom_range=0.2,
                         horizontal_flip=True,
                         vertical_flip=True,
                         fill_mode='reflect')

x_datagen = ImageDataGenerator(**datagen_args)
y_datagen = ImageDataGenerator(**datagen_args)

seed = 1
batch_size = 16
x_datagen.fit(trainx, augment=True, seed = seed)
y_datagen.fit(trainy, augment=True, seed = seed)

x_generator = x_datagen.flow(trainx, batch_size = 16, seed=seed)

y_generator = y_datagen.flow(trainy, batch_size = 16, seed=seed)

train_generator = zip(x_generator, y_generator)

X_datagen_val = ImageDataGenerator()
Y_datagen_val = ImageDataGenerator()
X_datagen_val.fit(testx, augment=True, seed=seed)
Y_datagen_val.fit(testy, augment=True, seed=seed)
X_test_augmented = X_datagen_val.flow(testx, batch_size=batch_size, seed=seed)
Y_test_augmented = Y_datagen_val.flow(testy, batch_size=batch_size, seed=seed)

test_generator = zip(X_test_augmented, Y_test_augmented)

model.fit_generator(train_generator, validation_data=test_generator, validation_steps=batch_size/2, epochs = 10, steps_per_epoch=len(x_generator))
model.save("model_augment.h5")
'''

#trainx = trainx/np.max(trainx)
#trainy = trainy/np.max(trainy)

#testx = testx/np.max(testx)
#testy = testy/np.max(testy)

history = model.fit(trainx, trainy_hot, epochs=20, validation_data = (testx, testy_hot),batch_size=64, verbose=1)
model.save("model_onehot.h5")


# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.savefig('acc_plot.png')
plt.show()
plt.close()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['train', 'val'], loc='upper right')
plt.savefig('loss_plot.png')
plt.show()
plt.close()

#epochs = 20
#for e in range(epochs):
#        print("epoch %d" % e)
#        #for X_train, Y_train in zip(x_train, y_train): # these are chunks of ~10k pictures
#        h,w,c = x_train.shape
#        X_train = np.reshape(x_train,(1,h,w,c))
#        h,w,c = y_train.shape
#        Y_train = np.reshape(y_train,(1,h,w,c))
#        model.fit(X_train, Y_train, batch_size=1, nb_epoch=1)
	
#        model.save("model_nocropping.h5")        
#print(X_train.shape, Y_train.shape)


#model.save("model_nocropping.h5")

#epochs = 10
#for e in range(epochs):
#	print("epoch %d" % e)
#	for X_train, Y_train in zip(x_train, y_train): # these are chunks of ~10k pictures
#		h,w,c = X_train.shape
#		X_train = np.reshape(X_train,(1,h,w,c))
#		h,w,c = Y_train.shape
#		Y_train = np.reshape(Y_train,(1,h,w,c))
#		model.fit(X_train, Y_train, batch_size=1, nb_epoch=1)
        #print(X_train.shape, Y_train.shape)


#model.save("model_nocropping.h5")

#accuracy = model.evaluate(x=x_test,y=y_test,batch_size=16)
#print("Accuracy: ",accuracy[1])

