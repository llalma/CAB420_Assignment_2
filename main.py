from load_data import load
import cv2
import matplotlib.pyplot as plt
import numpy as np
import random

def show_image(images,action,person,file_name,num_images):
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
    colour = cv2.COLOR_BGR2RGB
    split = 0.7

    train,test = load("binary_frames",img_size,colour,split)

    show_image(train[0],train[1],train[2],train[3],20)
#end