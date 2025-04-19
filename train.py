import tensorflow as tf
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg16_preprocess_input

class Trainer:
    def __init__(self, transformer_network, style_model, content_model, optimizer, loss_functions, style_features, style_weight=1e-2, content_weight=1e4, tv_weight=30):
        self.transformer_network = transformer_network
        self.style_model = style_model
        self.content_model = content_model
        self.optimizer = optimizer
        self.loss_functions = loss_functions
        self.style_features = style_features
        self.style_weight = style_weight
        self.content_weight = content_weight
        self.tv_weight = tv_weight

    @tf.function 
    def _train_step(self, content_batch):
        """Performs a single training step."""

        content_batch_vgg = vgg16_preprocess_input(tf.cast(content_batch, tf.float32))
        content_batch_transformer_input = tf.cast(content_batch, tf.float32) / 255.0

        with tf.GradientTape() as tape:
            generated_batch_tanh = self.transformer_network(content_batch_transformer_input, training=True) # Ensure training=True if using BatchNormalization etc.
            generated_batch_0_255 = ((generated_batch_tanh + 1.0) / 2.0) * 255.0
            generated_batch_vgg = vgg16_preprocess_input(generated_batch_0_255)
            content_features_vgg = self.content_model(content_batch_vgg)
            generated_style_features_vgg = self.style_model(generated_batch_vgg)
            generated_content_features_vgg = self.content_model(generated_batch_vgg)
            total_loss, style_loss_value, content_loss_value, tv_loss_value = self.loss_functions.compute_loss(
                self.style_features,              
                content_features_vgg,             
                generated_style_features_vgg,     
                generated_content_features_vgg,   
                generated_batch_tanh)

        gradients = tape.gradient(total_loss, self.transformer_network.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.transformer_network.trainable_variables))
        return total_loss, style_loss_value, content_loss_value, tv_loss_value

    def train(self, content_dataset, epochs):
        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            progbar = tf.keras.utils.Progbar(target=None) 
            step = 0
            for content_batch in content_dataset:
                total_loss, style_loss, content_loss, tv_loss = self._train_step(content_batch)
                progbar.update(step + 1, values=[
                    ("Total Loss", total_loss),
                    ("Style Loss", style_loss * self.loss_functions.style_weight), 
                    ("Content Loss", content_loss * self.loss_functions.content_weight),
                    ("TV Loss", tv_loss * self.loss_functions.tv_weight)
                ])
                step += 1
