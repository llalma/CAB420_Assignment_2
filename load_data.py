#First char == gesture
#Second char == person performing it.
#Third and Fourth == i think just to prevent have the same names.

import cv2
import os
import numpy as np

def load_images(dirpath,image_list,size,colour):
    images = []
    labels = []
    person = []
    file_name = []

    for filename in image_list:
        #Save images,resized and to the colour specified
        images.append(cv2.resize(cv2.cvtColor(cv2.imread(dirpath + "\\" + filename), colour ),size))
        labels.append(ord(filename[0]) - ord('a'))
        person.append(int(filename[2]))   
        file_name.append(filename)
    #end

    return [np.array(images),np.array(labels),np.array(person),np.array(file_name)]
#end

def load(path,size,colour,split):
    #Load data with input values.
    #The split value is the percentage of data that will be a testing and training set. given 0.7m, 70% will be training.
    path = os.getcwd() + "\\data\\" + path

    for dirpath, dirnames, filenames in os.walk(path):
        #Sort file names to ensure a correct order
        filenames.sort()
        training_index = round(len(filenames)*split)
        train = load_images(dirpath,filenames[0:training_index],size,colour)
        test = load_images(dirpath,filenames[training_index:],size,colour)
    #end

    return train,test
#end