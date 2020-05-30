import matplotlib.pyplot as plt
import numpy as np
import random


def show_image(images,action,person,file_name,num_images=10):
    """Show images, inputs in the form of (train[0],train[1],train[2],train[3],20)"""

    random_idx = random.sample(range(1, len(images)), num_images)
    fig = plt.figure(figsize=[25, 6])

    #Plot number of images input
    for i,val in enumerate(random_idx):
        ax = fig.add_subplot(4, num_images/2, i*2 + 1)
        ax.imshow(images[val])
        ax.set_title('Action ' + str(chr(action[val]+ord('A'))) + ' Person ' + str(person[val]) + "\nFile Name " + file_name[val])
    #end

    plt.show()
#end

def compare_images(images, true, pred, file_name, num_images=10, img_size=(100,100)):
    """Shows the input images, true and false labels on 1 model."""

    random_idx = random.sample(range(1, len(images)), num_images)
    fig = plt.figure(figsize=[25, 6])

    # images = np.array(images).reshape((len(pred),img_size[0],img_size[1]))

    #Plot number of images input
    for i,val in enumerate(random_idx):
        ax = fig.add_subplot(4, num_images/2, i*2 + 1)
        ax.imshow(images[val])
        ax.set_title('True ' + str(chr(true[val]+ord('A'))) + ' Pred ' + str(chr(pred[val]+ord('A'))) + "\nFile Name " + file_name[val])
    #end

    plt.show()
#end