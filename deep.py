# Os, OpenCV, matplotlib and numpy
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np

# Keras
import keras
import keras.layers as layers
from keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical

# Pandas
import pandas as pd

# Sklean 
from sklearn.metrics import plot_confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import multilabel_confusion_matrix, confusion_matrix, precision_score, accuracy_score
from sklearn.model_selection import train_test_split



#Helper functions
from helper_funcs.load_data import load,show_splits
from helper_funcs.show_image import show_image
from helper_funcs.evaluate import Top_N,CMC

#Convert to float16 for faster execution. allows for traiing of the resenet
from keras.backend.common import set_floatx
set_floatx('float16')



def eval_model(model, X_train, Y_train, X_test, Y_test, history):

    print('\n Evaluate on test data') # from tensorflow .com
    results = model.evaluate(X_test, Y_test, verbose=1)
    print('test loss and test acc:', results)

    # Print to screen test Accuracy
    pred = model.predict(X_test)
    # print('Test Accuracy: ' + str(sum(pred == Y_test)/len(Y_test)))

    # Round precicted data - this does what model.predict_classes does which is not available
    # for this model because it is not a sequential model. 
    proba = model.predict(X_test, verbose=1)
    
    if proba.shape[-1] > 1:
        rounded_pred = proba.argmax(axis=-1)
    else:
        rounded_pred = (proba > 0.5).astype('int32')

    y_pred = np.argmax(proba, axis=1)
    Y_test_labels = np.argmax(Y_test, axis=1)


    # Show Confusion Matrix and Perdicted Performance hystogram 
    cm = confusion_matrix(Y_test_labels, y_pred)

    fig = plt.figure(figsize=[16, 6])
    ax = fig.add_subplot(1, 2, 1)
    c = ConfusionMatrixDisplay(cm, display_labels=range(24))
    ax.set_title('Training Set Performance')
    c.plot(ax = ax)

    ax = fig.add_subplot(1, 2, 2)
    cm = confusion_matrix(Y_test_labels, rounded_pred)
    ax.hist(Y_test, bins=len(np.diagonal(cm)), rwidth=0.95)
    ax.set_title('Prediction Performance')
    ax.set_xlabel('Number (Label)')
    ax.set_ylabel('Correct Predictions')
    ax.plot(np.diagonal(cm))
    plt.show() # Show plots to screen


    # plot loss during training REF: https://keras.io/visualization/
    fig = plt.figure(figsize=[16, 6])
    ax = fig.add_subplot(1, 2, 1)
    ax.plot(history.history['accuracy'])
    ax.plot(history.history['val_accuracy'])
    ax.set_title('Model accuracy')
    ax.set_ylabel('Accuracy')
    ax.set_xlabel('Epoch')
    ax.legend(['Train', 'Test'], loc='upper left')

    # # Plot training & validation loss values REF:https://keras.io/visualization/
    ax = fig.add_subplot(1, 2, 2)
    ax.plot(history.history['loss'])
    ax.plot(history.history['val_loss'])
    ax.set_title('Model loss')
    ax.set_ylabel('Loss')
    ax.set_xlabel('Epoch')
    ax.legend(['Train', 'Test'], loc='upper left')
    plt.show()
    
    best_position = 0
    best_val = 100
    for idx, val in enumerate(history.history['val_loss']):
        if val < best_val:
            best_val = val
            best_position = idx
    print('Best Validation Loss and epoch:')
    print(best_val)
    print(best_position)


    return results[1], history, best_position, results
    

#end
#  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def data_augmentation():
    ''' This section will augment the images to increase the size of the data set
        taking one image and changing its view by applying muliple transformations. '''    
    
    # Reference: Question2 Week 5 solution
    datagen = ImageDataGenerator(
                            # Brightness of the image 20%  brightness value > 1 = lighter, < =  1 darker
                            brightness_range=[0.5,1.0],
                            # rotate between -10, +10 degrees
                            rotation_range=15,
                            # horiziontal shift by +/- 5% of the image width
                            width_shift_range=0.05,
                            # vertical shift by +/- 5% of the image width
                            height_shift_range=0.05,
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
                            # fill_mode='nearest',
                            )

    return datagen
#end

def fc_block(inputs, size, dropout):
    x = layers.Dense(size, activation=None)(inputs)
    x = layers.BatchNormalization()(x)
    # x = layers.Activation('relu')(x)
    if (dropout > 0.0):
        x = layers.Dropout(dropout)(x)

    return x
#end

def create_network(input_size,resnet_train):
    # base_network = keras.applications.InceptionV3(include_top=False, weights='imagenet', input_shape=(input_size[0],input_size[1],3), pooling=None)
    base_network = keras.applications.resnet.ResNet50(include_top=False, weights='imagenet', input_shape=(input_size[0],input_size[1],3), pooling=None)
    for layer in base_network.layers:
        layer.trainable = resnet_train
    #end

    inputs = keras.Input((input_size[0],input_size[1],3))

    #base network is Resnet50
    x = base_network(inputs)
    x = layers.MaxPooling2D()(x)
    # x = layers.MaxPooling2D(pool_size=(2,2), data_format="channels_first")(x)
    x = layers.Conv2D(filters=8,kernel_size=(1,1))(x)
    x = layers.Conv2D(filters=16,kernel_size=(1,1))(x)
    x = layers.Conv2D(filters=32,kernel_size=(1,1))(x)
    x = fc_block(x,128,0.5)

    x = layers.Flatten()(x)
    # output = layers.Dense(24,activation=None)(x)
    output = layers.Dense(24,activation='softmax', kernel_regularizer=keras.regularizers.l1(0.0000001))(x)
    # # output = layers.Dense(24,activation='relu')(x)

    model = keras.Model(inputs, output, name='Embedding')

    # optimizer=keras.optimizers.RMSprop()
    # optimizer=keras.optimizers.Adam()

    #loss='categorical_crossentropy'
    #loss=keras.losses.SparseCategoricalCrossentropy(from_logits = True)

    # model.compile(loss='categorical_crossentropy',optimizer=keras.optimizers.RMSprop(lr=0.0000001),metrics=['accuracy'])
    model.compile(loss='categorical_crossentropy',optimizer=keras.optimizers.RMSprop(lr=0.00001),metrics=['accuracy'])

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
    # Data size and details for training
    img_size = (100,100)
    embedding_size = 150
    resnet_train = True
    batch_size = 100
    epochs = 150
    colour = cv2.COLOR_BGR2RGB
    split = 0.8
    
    #Load training and testing set
    train,test = load("original_frames",img_size,colour,split)
    # print(len(train))
    # train_F,test_F = load("grayscale_frames",img_size,colour,1)
    # train_B,test_B = load("binary_frames",img_size,colour,1)
    # train = train + train_F + train_B

    print(len(train))  
    # show_splits(train[1],test[1])

    # Get Y
    #Convert the lables to categorial, e.g. 2 = [0,0,1,0]
    Y_train = to_categorical(train[1])
    Y_test = to_categorical(test[1])
    # Y_train = train[1]
    # Y_test = test[1]

    # Get X 
    X_train = train[0]
    X_test = test[0]

    # Get validation set
    X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.125, random_state=1) # 0.125 x 0.8 = 0.1 ref: https://datascience.stackexchange.com/questions/15135/train-test-validation-set-splitting-in-sklearn

    
    ######## - Load Model - ########
    deep_model = keras.models.load_model("temp.h5")
    history=np.load('history.npy',allow_pickle='TRUE').item()
    print(history.history.keys())
    
    ######## - Create Model - ########
    # deep_model = create_network(input_size=img_size,resnet_train=resnet_train)

    ######## - Trainining - ########
    ### Values of test and train np.array(images), np.array(labels), np.array(person), np.array(file_name) ###
    # datagen = data_augmentation()
    # checkpoint = ModelCheckpoint("temp.h5", verbose=2, monitor='val_loss', save_best_only=True, mode='auto')
    # history = deep_model.fit_generator(datagen.flow(X_train, Y_train, batch_size=batch_size), epochs=epochs,validation_data = (X_val, Y_val),shuffle=True, callbacks=[checkpoint]) 
    # np.save('history.npy',history) # Save history
    # deep_model.save("temp.h5") # Save Model
    # deep_model.load_weights("temp.h5")

    # Top N acccuracy and CMC Curve
    preds = deep_model.predict(X_test)
    Top5 = Top_N(Y_test,preds,5)
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