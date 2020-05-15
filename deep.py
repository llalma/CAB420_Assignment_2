import cv2
import matplotlib.pyplot as plt
import numpy as np
import random

#Helper functions
from helper_funcs.load_data import load
import helper_funcs.deep_network as create_network
from helper_funcs.deep_network import deep_network as create_network


def show_image(images,action,person,file_name,num_images):
    """Show images, inputs in the form of (train[0],train[1],train[2],train[3],20)"""

    random_idx = random.sample(range(1, len(images)), num_images)
    fig = plt.figure(figsize=[25, 6])

    #Plot number of images input
    for i,val in enumerate(random_idx):
        ax = fig.add_subplot(4, num_images/2, i*2 + 1)
        ax.imshow(images[val,:,:,:])
        ax.set_title('Action ' + str(action[val]) + ' Person ' + str(person[val]) + "\nFile Name " + file_name[val])
    #end

    plt.show()
#end

if __name__ == "__main__":
    img_size = (100,100)
    embedding_size = 500
    resnet_train = False
    batch_size = 32
    epochs = 5
    colour = cv2.COLOR_BGR2RGB
    split = 0.7
    
    #Load training and testing set
    train,test = load("binary_frames",img_size,colour,split)

    #Create models
    deep_model = create_network(input_size=img_size,embedding_size=embedding_size,resnet_train=resnet_train)

    history = deep_model.train(train[0],train[1],test[0],test[1],batch_size,epochs)
    preds = deep_model.predict(test[0],test[1])
#end