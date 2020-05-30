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

def edge_detection(images,sigma,low_threshold,high_threshold):
    #Detects edges for a list of images. returns a list of images with detected edges
    edges = []
    for image in images:
        edges.append(np.array(skimage.feature.canny(image=image,sigma=sigma,low_threshold=low_threshold,high_threshold=high_threshold)).flatten())
    #end
    return edges
#end

if __name__ == "__main__":
    img_size = (200,200)
    colour = cv2.COLOR_BGR2RGB
    split = 0.7
    
    #Load training and testing set
    train,test = load("original_frames",img_size,colour,split)
    
    # show_image(train[0],train[1],train[2],train[3],num_images=20)
    
    #HOC
    features_train = []
    for img in train[0]:
        # fd, hog_image = hog(img, orientations=9, pixels_per_cell=(4,4),cells_per_block=(3,3), visualize=True, multichannel=True)
        features_train.append(hog(img, orientations=10, pixels_per_cell=(8,8),cells_per_block=(4,4), visualize=False, multichannel=True))
    #end

    features_test = []
    for img in test[0]:
        # fd, hog_image = hog(img, orientations=9, pixels_per_cell=(4,4),cells_per_block=(3,3), visualize=True, multichannel=True)
        features_test.append(hog(img, orientations=10, pixels_per_cell=(8,8),cells_per_block=(4,4), visualize=False, multichannel=True))
    #end

    #Hyper tuning.
    param_grid = {'C': [0.1, 1, 10, 100, 1000],  
                'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 
                'kernel': ['poly', 'rbf', 'sigmoid']}  
    
    grid = GridSearchCV(svm.SVC(), param_grid, refit = True, verbose = 3) 
    grid.fit(features_train,train[1])
    
    print(grid.best_params_)
    print(grid.best_estimator_)

    #Predictions
    predicitons = grid.predict(features_test)

    #Own Accuracy, for sanity check
    accuracy = 0
    for i in range(0,len(predicitons)):
        # print("Predicted: " + str(predicitons[i]) + " Acutal: " + str(test[1][i]))
        accuracy += predicitons[i] == test[1][i]
    #end
    print("Accuracy: " + str(accuracy/len(predicitons)))

    #Show the edge detected images with the actual and predicted values.
    compare_images(test[0],test[1],predicitons,test[3])
#end