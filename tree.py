import numpy as np
from sklearn.ensemble import RandomForestClassifier
import cv2
import skimage.feature
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV

#Helper functions
from helper_funcs.load_data import load
from helper_funcs.show_image import show_image, compare_images

def edge_detection(images,sigma,low_threshold,high_threshold):
    #Detects edges for a list of images. returns a lsit of images with detected edges
    edges = []
    for image in images:
        edges.append(np.array(skimage.feature.canny(image=image,sigma=sigma,low_threshold=low_threshold,high_threshold=high_threshold)).flatten())
    #end
    return edges
#end

if __name__ == "__main__":
    img_size = (100,100)
    embedding_size = 100
    colour = cv2.COLOR_BGR2GRAY
    split = 0.7
    
    #Load training and testing set
    train,test = load("original_frames",img_size,colour,split)
    
    # show_image(train[0],train[1],train[2],train[3],num_images=20)

    #Edge detection
    train[0] = edge_detection(train[0],2,0.1,0.3)
    test[0] = edge_detection(test[0],2,0.1,0.3)
    
    # Create the parameter grid based on the results of random search 
    param_grid = {
        'bootstrap': [True],
        'max_depth': [2,20,100,200,1000],
        'max_features': [2, 3,6],
        'min_samples_leaf': [3, 4, 5],
        'min_samples_split': [8, 10, 12],
        'n_estimators': [100, 200, 300, 1000]
    }
    print(param_grid)

    #Random forest prediction
    rf = RandomForestClassifier()
    grid_search = GridSearchCV(estimator = rf, param_grid = param_grid,cv = 3, n_jobs = -1, verbose = 2)
    grid_search.fit(train[0], train[1])
    
    #Show best model
    print(grid_search.best_params_)
    print(grid_search.best_estimator_)

    #Predictions
    predicitons = grid_search.predict(test[0])

    #Imported accuracy.
    print(classification_report(test[1], predicitons)) 

    #Own Accuracy, for sanity check
    accuracy = 0
    for i in range(0,len(predicitons)):
        # print("Predicted: " + str(predicitons[i]) + " Acutal: " + str(test[1][i]))
        accuracy += predicitons[i] == test[1][i]
    #end
    print("Accuracy: " + str(accuracy/len(predicitons)))

    #Show the edge detected images with the actual and predicted values.
    # compare_images(test[0],test[1],predicitons,test[3])
#end