import tensorflow as tf
from scipy.stats import wasserstein_distance
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input

class ModelUtils:
    @staticmethod
    def save_model(model, filepath):
        model.save(filepath)

    @staticmethod
    def load_model(filepath):
        return tf.keras.models.load_model(filepath)

class ImageUtils:
    @staticmethod
    def preprocess_image(image):
        image = image.resize((224, 224))  
        image_array = img_to_array(image) 
        image_array = np.expand_dims(image_array, axis=0)  
        return preprocess_input(image_array) 