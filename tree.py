import numpy as np
from sklearn.ensemble import RandomForestClassifier
import cv2
import skimage.feature
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV

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
    
    # Create the parameter grid based on the results of random search 
    param_grid = {
        'bootstrap': [True],
        'max_depth': [2,200,1000],
        'max_features': [2, 3,6],
        'min_samples_leaf': [3, 4, 5],
        'min_samples_split': [8, 10, 12],
        'n_estimators': [100, 300, 1000]
    }
    print(param_grid)

    #Optimised model
    grid_search = RandomForestClassifier(bootstrap=True,max_depth=200,max_features=6,min_samples_leaf=3,min_samples_split=10,n_estimators=1000)

    #Find optimal model
    # grid_search = GridSearchCV(estimator = RandomForestClassifier(), param_grid = param_grid,cv = 3, n_jobs = -1, verbose = 2)
    
    
    grid_search.fit(features_train, train[1])
    # print(grid_search.best_params_)
    # print(grid_search.best_estimator_)


    #Predictions
    predictions = grid_search.predict(features_test)

    #Imported accuracy.
    print(classification_report(test[1], predictions)) 

    #Own Accuracy, for sanity check
    accuracy = 0
    for i in range(0,len(predictions)):
        # print("Predicted: " + str(predicitons[i]) + " Acutal: " + str(test[1][i]))
        accuracy += predictions[i] == test[1][i]
    #end
    print("Accuracy: " + str(accuracy/len(predictions)))

    #Top N Accuracy and CMC
    probs = grid_search.predict_proba(features_test)
    print("Accuracy for top 1: " + str(Top_N(test[1],probs,1)))
    print("Accuracy for top 3: " + str(Top_N(test[1],probs,3)))
    print("Accuracy for top 5: " + str(Top_N(test[1],probs,5)))
    CMC(test[1],probs)


    # Show the edge detected images with the actual and predicted values.
    compare_images(test[0],test[1],predictions,test[3])
#end