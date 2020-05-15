import keras
import keras.layers as layers
from helper_funcs.data_augmentation import augmentation
from helper_funcs.evaluate import CMC,Top_N
import numpy as np

class deep_network():

    def __init__(self, input_size,embedding_size,resnet_train=False):
        """resnet_train set to true will make the resnet trainable, default False. """
        self.input_shape = (input_size[0],input_size[1],3)
        self.embedding_size = embedding_size
        self.resnet_train = resnet_train

        #Models
        self.embedding = self.embedding_model()
        self.full = self.full_model()
    #end

    def embedding_model(self):
        base_network = keras.applications.resnet.ResNet50(include_top=False, weights='imagenet', input_shape=self.input_shape, pooling=None)
        for layer in base_network.layers:
            layer.trainable = self.resnet_train
        #end

        dummy_input = keras.Input(self.input_shape)
        #base network is Resnet50
        output = layers.Dense(self.embedding_size,activation='softmax')(base_network(dummy_input))
        output = layers.Flatten()(output)

        model = keras.Model(dummy_input, output, name='Embedding')
        return model
    #end

    def full_model(self):
        inputs = keras.Input(self.input_shape,name="Input")

        embedded_image = self.embedding(inputs)

        outputs = layers.Dense(64, activation='softmax')(embedded_image)
        outputs = layers.Dense(24,name = "Output")(outputs)

        model = keras.Model(inputs=inputs,outputs=[outputs],name="Full Model")
        model.compile(loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),optimizer=keras.optimizers.RMSprop(),metrics=['accuracy'])

        return model  
    #end

    def train(self,train_x,train_y,test_x,test_y,batch_size,epochs):
        """Trains the full model and returns the history of training"""

        datagen = augmentation()
        history = self.full.fit(datagen.flow(train_x,train_y,batch_size),epochs=epochs,validation_data=(test_x,test_y),shuffle=True)

        return history
    #end

    def predict(self,x,y):
        """Predics input data with the full model. Returns the predicted class for each image"""

        probs = self.full.predict(x)

        print("Accuracy for top 5 is: " + str(Top_N(y,probs,5)))
        CMC(y,probs)

        #Convert probabilites to the top class prediction. for each image
        preds = []
        for prob in probs:
            preds.append(np.argmax(prob))
        #end
        return preds
    #end
#end