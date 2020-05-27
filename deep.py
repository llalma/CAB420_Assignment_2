import cv2
import matplotlib.pyplot as plt
import numpy as np

import keras
import keras.layers as layers
from keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical

#Helper functions
from helper_funcs.load_data import load,show_splits
from helper_funcs.show_image import show_image
from helper_funcs.evaluate import Top_N,CMC

#Convert to float16 for faster execution. allows for traiing of the resenet
from keras.backend.common import set_floatx
set_floatx('float16')

def data_augmentation():
    ''' This section will augment the images to increase the size of the data set
        taking one image and changing its view by applying muliple transformations. '''    
    
    # Reference: Question2 Week 5 solution
    datagen = ImageDataGenerator(
                            # Brightness of the image 20%  brightness value > 1 = lighter, < =  1 darker
                            brightness_range=[0.5,1.0],
                            # rotate between -10, +10 degrees
                            rotation_range=10,
                            # horiziontal shift by +/- 10% of the image width
                            width_shift_range=0.10,
                            # vertical shift by +/- 5% of the image width
                            height_shift_range=0.10,
                            # range for zooming
                            zoom_range=0.05,
                            # allow horizontal flips of data
                            horizontal_flip=True,
                            # Boolean. Set input mean to 0 over the dataset, feature-wise.
                            # featurewise_center=True,
                            # Boolean. Divide inputs by std of the dataset, feature-wise.
                            # featurewise_std_normalization=True,
                            # what value to place in new pixels
                            # fill_mode='constant', cval=.5)
                            fill_mode='nearest',
                            )

    return datagen
#end

def fc_block(inputs, size, dropout):
    x = layers.Dense(size, activation=None)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    if (dropout > 0.0):
        x = layers.Dropout(dropout)(x)
    
    return x
#end

def create_network(input_size,resnet_train):
    base_network = keras.applications.resnet.ResNet50(include_top=False, weights='imagenet', input_shape=(input_size[0],input_size[1],3), pooling=None)
    for layer in base_network.layers:
        layer.trainable = resnet_train
    #end

    inputs = keras.Input((input_size[0],input_size[1],3))

    #base network is Resnet50
    x = layers.MaxPooling2D()(base_network(inputs))
    x = layers.Conv2D(filters=8,kernel_size=(1,1))(x)
    x = layers.Conv2D(filters=16,kernel_size=(1,1))(x)
    x = layers.Conv2D(filters=32,kernel_size=(1,1))(x)
    x = fc_block(x,192,0.2)

    x = layers.Flatten()(x)
    output = layers.Dense(24,activation='softmax')(x)

    model = keras.Model(inputs, output, name='Embedding')

    model.compile(loss='categorical_crossentropy',optimizer=keras.optimizers.Adam(),metrics=['accuracy'])

    return model
#end

def get_accuracy(truths,preds):
    accuracy = 0
    for true,pred in zip(truths,preds):
        if true == pred:
            accuracy+=1
        #end
    #end

    return accuracy/len(truths)
#end



if __name__ == "__main__":
    img_size = (100,100)
    embedding_size = 150
    resnet_train = True
    batch_size = 32
    epochs = 1
    colour = cv2.COLOR_BGR2RGB
    split = 0.7
    
    
    #Load training and testing set
    train,test = load("grayscale_frames",img_size,colour,split)
    show_splits(train[1],test[1])
    train[1] = to_categorical(train[1])
    test[1] = to_categorical(test[1])

    
    checkpoint = ModelCheckpoint("temp.h5", verbose=2, monitor='val_loss', save_best_only=True, mode='auto')
    #Create model
    # deep_model = create_network(input_size=img_size,resnet_train=resnet_train)
    deep_model = keras.models.load_model("temp.h5")

    datagen = data_augmentation()
    history = deep_model.fit_generator(datagen.flow(train[0], train[1], batch_size=batch_size), epochs=epochs,validation_data = (test[0], test[1]),shuffle=True, callbacks=[checkpoint])
    preds = deep_model.predict(test[0])

    print(Top_N(test[1],preds,2))
    CMC(test[1],preds)
#end