import os
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg16_preprocess_input

class DataProcessor:
    def __init__(self, img_height=224, img_width=224):
        self.img_height = img_height
        self.img_width = img_width

    def load_image(self, image_path):
        img = Image.open(image_path)
        img = img.resize((self.img_height, self.img_width))
        img = np.array(img)
        if img.shape[-1] == 3:  
            img = np.expand_dims(img, axis=0)  
        return img

    def preprocess_image(self, image):
        return image.astype('float32') / 255.0

    def load_and_preprocess_image(self, image_path):
        image = self.load_image(image_path)
        return self.preprocess_image(image)

    def save_image(self, image, save_path):
        img = Image.fromarray((image[0] * 255).astype(np.uint8))
        img.save(save_path)

    def preprocess_style_image(self, style_image_path):
        style_image = load_img(style_image_path, target_size=(self.img_height, self.img_width))
        style_array = img_to_array(style_image)  
        style_array = np.expand_dims(style_array, axis=0)  
        return vgg16_preprocess_input(style_array)

