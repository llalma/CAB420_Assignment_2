import numpy as np
from sklearn import svm 
import cv2
import skimage.feature
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV 
from sklearn.metrics import classification_report

from skimage.feature import hog
from skimage import data, exposure


#Helper functions
from helper_funcs.load_data import load
from helper_funcs.show_image import show_image, compare_images
from helper_funcs.evaluate import Top_N,CMC

def hog_self(images):
    #Converts a list of images to a hog representation. features is the feature vector and imgs_out is the visualsation of the hog.

    features = []
    imgs_out = []
    for img in images:
        fd, hog_image = hog(img, orientations=10, pixels_per_cell=(8,8),cells_per_block=(4,4), visualize=True, multichannel=True)
        features.append(fd)
        imgs_out.append(hog_image)
    #end
    
    return features,imgs_out
#end

if __name__ == "__main__":
    img_size = (200,200)
    colour = cv2.COLOR_BGR2RGB
    split = 0.7
    
    #Load training and testing set
    train,test = load("original_frames",img_size,colour,split)
    
    #HOG
    features_train,hog_train = hog_self(train[0])
    features_test,hog_test = hog_self(test[0])
    
    #Hyper tuning.
    param_grid = {'C': [0.1, 1, 100, 1000],  
                'gamma': [1, 0.1, 0.001, 0.0001], 
                'kernel': ['rbf']}  
    
    #Non optimised model
    # grid = svm.SVC()
    # grid.fit(features_train,train[1])

    #Optimised model
    grid = svm.SVC(C=100,gamma=0.0001,kernel='rbf',probability=True)

    #Find optimal model
    # grid = GridSearchCV(svm.SVC(), param_grid, refit = True, verbose = 3) 

    grid.fit(features_train,train[1])  
    # print(grid.best_params_)
    # print(grid.best_estimator_)

    #Predictions
    predicitons = grid.predict(features_test)

    #Own Accuracy, for sanity check
    accuracy = 0
    for i in range(0,len(predicitons)):
        # print("Predicted: " + str(predicitons[i]) + " Acutal: " + str(test[1][i]))
        accuracy += predicitons[i] == test[1][i]
    #end
    print("Accuracy: " + str(accuracy/len(predicitons)))

    #Top N Accuracy and CMC
    probs = grid.predict_proba(features_test)
    print("Accuracy for top 1: " + str(Top_N(test[1],probs,1)))
    print("Accuracy for top 3: " + str(Top_N(test[1],probs,3)))
    print("Accuracy for top 5: " + str(Top_N(test[1],probs,5)))
    CMC(test[1],probs)

    #Show the edge detected images with the actual and predicted values.
    compare_images(test[0],test[1],predicitons,test[3])
#end