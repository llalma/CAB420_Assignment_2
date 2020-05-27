import numpy as np
from sklearn import svm 
import cv2
import skimage.feature
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV 
from sklearn.metrics import classification_report

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
    img_size = (100,100)
    embedding_size = 100
    colour = cv2.COLOR_BGR2GRAY
    split = 0.7
    
    #Load training and testing set
    train,test = load("grayscale_frames",img_size,colour,split)
    
    # show_image(train[0],train[1],train[2],train[3],num_images=20)

    #Edge detection
    train[0] = edge_detection(train[0],1,0.1,0.2)
    test[0] = edge_detection(test[0],1,0.1,0.2)

    #Hyper tuning.
    param_grid = {'C': [0.1, 1, 10, 100, 1000],  
                'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 
                'kernel': ['poly', 'rbf', 'sigmoid']}  
    
    grid = GridSearchCV(svm.SVC(), param_grid, refit = True, verbose = 3) 
    grid.fit(train[0],train[1])
    
    print(grid.best_params_)
    print(grid.best_estimator_)

    #Predictions
    predicitons = grid.predict(test[0])

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