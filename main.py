import os
import tensorflow as tf
from transformer import TransformerNetwork
from preprocessing import Preprocessor
from loss import LossFunctions
from train import Trainer
from model import VGG16Model
from utils import ImageUtils, ModelUtils
from visualization import Visualizer
from PIL import Image
import numpy as np

def main():
    epochs = 10
    img_height, img_width = 224, 224
    batch_size = 4
    style_weight = 100.0
    content_weight = 10.0 
    tv_weight = 2.0 
    num_images_to_load = 2000
    learning_rate = 0.001 
    
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate) 

    style_image_path = "dataset/style_images/doggo.jpg" 
    content_dir = "dataset/images" 

    preprocessor = Preprocessor(img_height=img_height, img_width=img_width, batch_size=batch_size)
    style_image = preprocessor.preprocess_style_image(style_image_path)
    content_gen = preprocessor.content_generator(content_dir)
    content_dataset = preprocessor.generator_to_tf_dataset(content_gen, num_images_to_load)

    vgg16 = VGG16Model(input_shape=(img_height, img_width, 3))
    style_model = vgg16.get_style_model()
    content_model = vgg16.get_content_model()

    transformer = TransformerNetwork(input_shape=(img_height, img_width, 3))
    transformer_model = transformer.get_model()

    style_features = style_model(style_image)

    loss_functions = LossFunctions(style_weight=style_weight, content_weight=content_weight, tv_weight=tv_weight)

    trainer = Trainer(
        transformer_network=transformer_model,
        style_model=style_model,
        content_model=content_model,
        optimizer=optimizer,
        loss_functions=loss_functions,
        style_features=style_features,
        style_weight=style_weight,
        content_weight=content_weight,
        tv_weight=tv_weight
    )

    print("Starting training...")
    trainer.train(content_dataset, epochs)
    print("Training finished.")

    save_path = "Model/model.keras"
    print(f"Saving model to {save_path}...")
    ModelUtils.save_model(transformer_model, save_path)
    print("Model saved.")

    print("Loading model for inference...")
    loaded_model = ModelUtils.load_model(save_path)
    print("Model loaded.")

    test_image_path = "dataset/images/contents_images/COCO_train2014_000000000034.jpg" # Make sure path is correct
    print(f"Loading and preprocessing test image: {test_image_path}")
    image = Image.open(test_image_path)

    image_resized = image.resize((img_height, img_width))
    image_array = tf.keras.preprocessing.image.img_to_array(image_resized)
    image_array = image_array / 255.0 
    preprocessed_image_inf = np.expand_dims(image_array, axis=0) 

    print("Generating stylized image...")
    stylized_image_tanh = loaded_model(preprocessed_image_inf, training=False) 

    stylized_image_0_1 = (stylized_image_tanh + 1.0) / 2.0
    stylized_image_0_1 = tf.clip_by_value(stylized_image_0_1, 0.0, 1.0)

    content_image_0_1 = np.array(image_resized) / 255.0

    output_path = "output_images/stylized_output.png"
    print(f"Saving comparison image to: {output_path}")
    Visualizer.save_side_by_side(content_image_0_1, stylized_image_0_1[0].numpy(), output_path) # Use .numpy() to convert tensor

    print(f"Stylized image saved at: {output_path}")

if __name__ == "__main__":
    os.makedirs("Model", exist_ok=True)
    os.makedirs("output_images", exist_ok=True)
    main()
