import tensorflow as tf

class GTTransformer(tf.keras.Model):
    def __init__(self,
        input_size: int,
        out_seq_len: int=58,
        dim_val: int=512,
        num_encoder_layers: int=4,
        num_heads: int=8,
        dropout_encoder: float=0.2,
        dropout_pos_enc: float=0.1,
        dim_feedforward_encoder: int=2048,
        activation: str='relu',
        ):
        """
        """
        super().__init__() 

        # input layer        
        self.encoder_input_layer = tf.keras.layers.Dense(
            units=out_seq_length, 
            activation=activation 
            )
        # positional layer 
        # TODO: positional layer for variable-length input
        
        # attention layer (multihead)
        self.attention_layer = tf.keras.layers.MultiHeadAttention(
                num_heads,
                key_dim,)

        # encoder layer
        encoder_layer = tf.keras_nlp.layers.TransformerEncoder(
            dim_val,
            num_heads,
            dropout=dropout_encoder,
            activation=activation,
            layer_norm_epsilon=1e-05,
            kernel_initializer="glorot_uniform",
            bias_initializer="zeros",
            )
        
        # feed forward loop
        def feed_forward():
            x = self.encoder_input_layer(src)
            x = self.encoder_layer(x=x)
            return x


class DistanceLayer(tf.keras.layers.Layer):
    """
    Computes distance between two vectors with simple euclidean distance
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, sample1, sample2):
        distance = tf.reduce_sum(tf.square(sample1 - sample2), -1)
        # distance = tf.norm(sample1 - sample2, ord='euclidean')
        return distance


def build_siamese_network(base_model: tf.keras.Model, vector_size: int):
    """
    Builds siamese network
    :param vector_size: size of the sample inputs
    :return: siamese network with two inputs (s1, s2) and one output (predicted distance)
    """

    # TODO allow for channels dimension
    sample1_input = tf.keras.Input(name="sample1", shape=vector_size)
    sample2_input = tf.keras.Input(name="sample2", shape=vector_size)

    # predicted_distances = DistanceLayer()(
    predicted_distances = tf.keras.layers.Dot(axes=-1, normalize=True)(
        [base_model(sample1_input), base_model(sample2_input)]
    )
    siamese_network = tf.keras.Model(
        inputs=[sample1_input, sample2_input], outputs=predicted_distances
    )
    return siamese_network


class SiameseModel(tf.keras.Model):
    """
    Wraps siamese network for training
    """

    def __init__(self, siamese_network):
        super(SiameseModel, self).__init__()
        self.siamese_network = siamese_network
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")

    def call(self, data):
        sample1, sample2 = data[:2]
        return self.siamese_network(sample1, sample2)

    def train_step(self, data):
        """
        :param data: sample1, sample2
        :return:
        """
        sample_data = data[:2]
        target_data = data[2]

        with tf.GradientTape() as tape:
            loss = self._compute_loss(sample_data, target_data)

        gradients = tape.gradient(loss, self.siamese_network.trainable_weights)

        self.optimizer.apply_gradients(
            zip(gradients, self.siamese_network.trainable_weights)
        )

        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def test_step(self, data):
        sample_data = data[:2]
        target_data = data[2]
        loss = self._compute_loss(sample_data, target_data)

        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def _compute_loss(self, sample_data, target_distances):
        predicted_distance = self.siamese_network(sample_data)
        # loss = tf.keras.metrics.mean_squared_error(predicted_distance, target_distances)
        loss = tf.keras.losses.Huber()(predicted_distance, target_distances)
        return loss

    @property
    def metrics(self):
        return [self.loss_tracker]
