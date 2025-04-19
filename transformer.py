from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, Add, BatchNormalization, Activation

class TransformerNetwork:
    def __init__(self, input_shape=(224, 224, 3)):
        self.input_shape = input_shape
        self.model = self._build_transformer_network()

    def _build_transformer_network(self):
        inputs = Input(shape=self.input_shape)  

        x = Conv2D(32, (9, 9), strides=1, padding='same', activation='relu')(inputs)
        x = Conv2D(64, (3, 3), strides=2, padding='same', activation='relu')(x)
        x = Conv2D(128, (3, 3), strides=2, padding='same', activation='relu')(x)

        for _ in range(5):
            x = self._residual_block(x)

        x = Conv2DTranspose(64, (3, 3), strides=2, padding='same', activation='relu')(x)
        x = Conv2DTranspose(32, (3, 3), strides=2, padding='same', activation='relu')(x)

        outputs = Conv2D(3, (9, 9), padding='same', activation='tanh')(x)

        return Model(inputs, outputs, name="TransformerNetwork")

    def _residual_block(self, x):
        y = Conv2D(128, (3, 3), padding='same')(x)
        y = BatchNormalization()(y)
        y = Activation('relu')(y)
        y = Conv2D(128, (3, 3), padding='same')(y)
        y = BatchNormalization()(y)
        return Add()([x, y])

    def get_model(self):
        return self.model