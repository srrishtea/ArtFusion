import tensorflow as tf
from tensorflow.keras import layers, Model, Sequential, Input
from tensorflow.keras.layers import Conv2D, MaxPooling2D

class VGG16Model:
    def __init__(self, input_shape=(224, 224, 3)):
        self.input_shape = input_shape
        self.vgg16_model = self._build_vgg16_model()
        self.style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']
        self.content_layer = ['block4_conv2']
        self._load_pretrained_weights()

    def _build_vgg16_model(self):
        vgg = tf.keras.applications.VGG16(include_top=False, weights='imagenet', input_shape=self.input_shape)
        return vgg

    def _load_pretrained_weights(self):
        for layer in self.vgg16_model.layers:
            layer.trainable = False

    def get_layer_outputs(self, layer_names):
        outputs = [self.vgg16_model.get_layer(name).output for name in layer_names]
        return Model(inputs=self.vgg16_model.input, outputs=outputs)

    def get_style_model(self):
        return self.get_layer_outputs(self.style_layers)

    def get_content_model(self):
        return self.get_layer_outputs(self.content_layer)

    def compile_model(self, model):
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model