from tensorflow.keras.preprocessing.image import ImageDataGenerator

def augmentation():
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