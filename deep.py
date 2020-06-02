import cv2
import matplotlib.pyplot as plt
import numpy as np

import keras
import keras.layers as layers
from keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical

#Helper functions
from helper_funcs.load_data import load
# from helper_funcs.show_image import show_image
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

    inputs = keras.Input((input_size[0],input_size[1],3))

    x = layers.AveragePooling2D(name='sub1')(inputs)
    x = layers.Conv2D(filters=16,kernel_size=(3,3),activation='relu',name='conv1')(x)
    x = layers.AveragePooling2D(name='sub2')(x)
    x = layers.Flatten()(x)
    x = layers.Dense(units=120,name='dense1', activation='relu')(x)
    x = layers.Dense(units=84,name='dense2', activation='relu')(x)

    output = layers.Dense(24,activation='softmax',name='output')(x)

    model = keras.Model(inputs, output, name='LeNet')

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

    # show_splits(train[1],test[1])

    #Convert the lables to categorial, e.g. 2 > [0,0,1,0] 
    train[1] = to_categorical(train[1])
    test[1] = to_categorical(test[1])

    # fig = plt.figure()
    # fig.add_subplot(3,1,1)
    # plt.imshow(X_train[0])
    # fig.add_subplot(3,1,2)
    # plt.imshow(X_val[0])
    # fig.add_subplot(3,1,3)
    # plt.imshow(X_test[0])
    # plt.show()
    
    ######## - Load Model - ########
    deep_model = keras.models.load_model("temp.h5")
    history=np.load('history.npy',allow_pickle='TRUE').item()
    print(history.history.keys())
    
    ######## - Create Model - ########
    # deep_model = create_network(input_size=img_size,resnet_train=resnet_train)
    # deep_model.summary()

    # ####### - Trainining - ########
    # ## Values of test and train np.array(images), np.array(labels), np.array(person), np.array(file_name) ###
    # datagen = data_augmentation()
    # checkpoint = ModelCheckpoint("temp.h5", verbose=2, monitor='val_loss', save_best_only=True, mode='auto')
    # history = deep_model.fit_generator(datagen.flow(X_train, Y_train, batch_size=batch_size), epochs=epochs,validation_data = (X_val, Y_val),shuffle=True, callbacks=[checkpoint]) 
    # np.save('history.npy',history) # Save history
    # deep_model.save("temp.h5") # Save Model
    # deep_model.load_weights("temp.h5")

    # Top N acccuracy and CMC Curve
    preds = deep_model.predict(X_test)
    Top5 = Top_N(Y_test,preds,1)
    print('Top 5 Acc')
    print(Top5)
    CMC(Y_test,preds)

    # Model Evaluation  eval_model(model, X_train, Y_train, X_test, Y_test, history)
    test_accuracy, history, best_position, results = eval_model(deep_model, X_train, Y_train, X_test, Y_test, history)


    best_accuracy = 0
    if test_accuracy > best_accuracy:
        best_accuracy = test_accuracy
    
    # Write model details to txt file!
    f= open(os.getcwd() + "\\Model_Summary.txt","w+")
    f.write(' Val_loss did not improve from: ' + str(history.history['val_loss'][best_position]) + ' At epoch No.' + str(best_position+1))
    f.write(' \nFinal Loss:' + str(history.history['loss'][best_position]))
    f.write(' \nFinal accuracy:' + str(history.history['accuracy'][best_position]))
    f.write(' \nFinal val_accuracy:' + str(history.history['val_accuracy'][best_position]))
    f.write(' \n Total epochs:' + str(len(history.history['val_loss'])))
    f.write(' \n Top 5:' + str(Top5))
    f.write("\n\n test loss and test acc:" + str(results))
    deep_model.summary(print_fn = lambda x: f.write('\n\n'+ str(x)))
    f.close() 
    #end

#end