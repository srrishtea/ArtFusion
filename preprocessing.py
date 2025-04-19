import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg16_preprocess_input
import numpy as np
import os

class Preprocessor:
    def __init__(self, img_height=224, img_width=224, batch_size=32):
        self.img_height = img_height
        self.img_width = img_width
        self.batch_size = batch_size

    def preprocess_style_image(self, style_image_path):
        style_image = load_img(style_image_path, target_size=(self.img_height, self.img_width))
        style_array = img_to_array(style_image)  
        style_array = np.expand_dims(style_array, axis=0)  
        return vgg16_preprocess_input(style_array)  

    def content_generator(self, content_dir):
        datagen = ImageDataGenerator()
        
        content_gen = datagen.flow_from_directory(
            directory=os.path.join(content_dir),
            target_size=(self.img_height, self.img_width),
            batch_size=self.batch_size,
            class_mode=None
        )
        
        return content_gen

    def generator_to_tf_dataset(self, generator, num_images):
        dataset = tf.data.Dataset.from_generator(
            lambda: generator,
            output_signature=tf.TensorSpec(shape=(None, self.img_height, self.img_width, 3), dtype=tf.float32))
        return dataset.take(num_images // self.batch_size)

    def preprocess_content_batch(self, content_batch):
        return vgg16_preprocess_input(content_batch)