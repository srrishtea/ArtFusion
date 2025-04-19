import tensorflow as tf

class LossFunctions:
    def __init__(self, style_weight=1e-2, content_weight=1e4, tv_weight=30):
        self.style_weight = style_weight
        self.content_weight = content_weight
        self.tv_weight = tv_weight

    @staticmethod
    def content_loss(content_feature_tensor, generated_feature_tensor):
        loss = tf.reduce_mean(tf.square(content_feature_tensor - generated_feature_tensor))
        return loss

    @staticmethod
    def style_loss(style_features, generated_features):
        loss = 0
        if not isinstance(style_features, (list, tuple)):
             style_features = [style_features]
        if not isinstance(generated_features, (list, tuple)):
             generated_features = [generated_features]

        num_style_layers = len(style_features)
        if num_style_layers == 0:
            return tf.constant(0.0, dtype=tf.float32)

        for s_feat, g_feat in zip(style_features, generated_features):
            batch_size = tf.shape(s_feat)[0]
            height = tf.shape(s_feat)[1]
            width = tf.shape(s_feat)[2]
            channels = tf.shape(s_feat)[3]
            num_elements = tf.cast(height * width * channels, tf.float32) 

            s_gram = tf.linalg.einsum('bijc,bijd->bcd', s_feat, s_feat) / num_elements
            g_gram = tf.linalg.einsum('bijc,bijd->bcd', g_feat, g_feat) / num_elements

            loss += tf.reduce_mean(tf.square(s_gram - g_gram))

        return loss / tf.cast(num_style_layers, tf.float32)


    @staticmethod
    def total_variation_loss(image):
        x_deltas = image[:, 1:, :, :] - image[:, :-1, :, :] 
        y_deltas = image[:, :, 1:, :] - image[:, :, :-1, :] 

        return tf.reduce_mean(tf.square(x_deltas)) + tf.reduce_mean(tf.square(y_deltas))


    def compute_loss(self, style_features, content_features, generated_style_features, generated_content_features, generated_image):
        if not isinstance(content_features, (list, tuple)):
             content_features = [content_features]
        if not isinstance(generated_content_features, (list, tuple)):
             generated_content_features = [generated_content_features]

        style_loss_value = self.style_loss(style_features, generated_style_features)
        content_loss_value = self.content_loss(content_features[0], generated_content_features[0])
        tv_loss_value = self.total_variation_loss(generated_image)

        total_loss = (self.style_weight * style_loss_value +
                      self.content_weight * content_loss_value +
                      self.tv_weight * tv_loss_value)

        return total_loss, style_loss_value, content_loss_value, tv_loss_value