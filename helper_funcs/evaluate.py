import numpy as np
import random
import matplotlib.pyplot as plt

def Top_N(truth, probs,n):
    #Returns accuracy when predicted value is in the top n predicted values.
    sum_total=0

    for i in range(len(probs)): #vertical
        #Get the positions of the n largest values, larger valeus being the prediciton
        topn = np.argsort(probs[i],axis=0)[-n:] 

        #The positions correspond to the labels predicted
        # topn_labels = np.concatenate(np.array(truth)[np.array(topn)]).tolist()
        if truth[i] in topn:
            sum_total+=1
        #end
    #end

    return sum_total/len(truth)
#end

def CMC(truth,probs):
    # A CMC curve plots the top-N accuracy for all possible values of N
    accuracies = []
    for n in range(1,len(truth)):
        accuracies.append(Top_N(truth,probs,n))
    #end

    fig = plt.figure(figsize=[25, 8])
    plt.title("Top N vs accuracy")
    plt.xlabel("N")
    plt.ylabel("Accuracy")
    plt.plot(range(len(accuracies)),accuracies)
    plt.show()

#end