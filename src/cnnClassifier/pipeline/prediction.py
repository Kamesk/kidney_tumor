import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os
from tensorflow import keras

class PredictionPipeline:
    def __init__(self,filename):
        self.filename =filename
    

    # Load the model using the absolute path
    
    def predict(self):
        # load model
        model_path = "D:\tube\kidney_tumor\artifacts\training\model.h5"
        model = keras.models.load_model(model_path)

        imagename = self.filename
        test_image = image.load_img(imagename, target_size = (224,224))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis = 0)
        result = np.argmax(model.predict(test_image), axis=1)
        print(result)

        if result[0] == 1:
            prediction = 'Tumor'
            return [{ "image" : prediction}]
        else:
            prediction = 'Normal'
            return [{ "image" : prediction}]