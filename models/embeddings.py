import tensorflow as tf
from models.utils import create_initializer

class BertEmbeddingsLayer(tf.keras.layers.Layer):
    def __init__(self,
                 name,
                 vocab_size,
                 use_token_type=False,
                 token_type_vocab_size=16,
                 use_position_embeddings=True,
                 hidden_size=768,
                 initializer_range=0.02,
                 word_embedding_name="word_embeddings",
                 token_type_embedding_name="token_type_embeddings",
                 dropout_rate=0.1):
        super(BertEmbeddingsLayer, self).__init__(name=name)
        self.vocab_size = vocab_size
        self.use_token_type = use_token_type
        self.token_type_vocab_size = token_type_vocab_size
        self.use_position_embeddings = use_position_embeddings
        self.dropout_rate = dropout_rate
        self.hidden_size = hidden_size
        self.initializer_range = initializer_range
        self.word_embedding_name = word_embedding_name
        self.token_type_embedding_name = token_type_embedding_name
        self.word_embeddings_layer = None
        self.token_type_embeddings_layer = None
        self.position_embeddings_layer = None
        self.layer_norm = None
        self.dropout_layer = None

    def build(self, input_shape):
        if isinstance(input_shape, list):
            assert len(input_shape) == 2
            input_ids_shape, token_type_ids_shape = input_shape
            self.input_spec = [tf.keras.layers.InputSpec(shape=input_ids_shape),
                               tf.keras.layers.InputSpec(shape=token_type_ids_shape)]
        else:
            input_ids_shape = input_shape
            self.input_spec = tf.keras.layers.InputSpec(shape=input_ids_shape)

        # init layers
        self.word_embeddings_layer = tf.keras.layers.Embedding(
            self.vocab_size,
            self.hidden_size,
            mask_zero=True,
            name=self.word_embedding_name
        )

        if self.use_token_type:
            self.token_type_embeddings_layer = tf.keras.layers.Embedding(
                self.token_type_vocab_size,
                self.hidden_size,
                mask_zero=True,
                name=self.token_type_embedding_name
            )
        if self.use_position_embeddings:
            self.position_embeddings_layer = PositionalEncoding(name="position_embeddings",
                                                                embedding_size=self.hidden_size)

        self.layer_norm = tf.keras.layers.LayerNormalization(name="LayerNorm")
        self.dropout_layer = tf.keras.layers.Dropout(rate=self.dropout_rate)
        super(BertEmbeddingsLayer, self).build(input_shape)

    def call(self, inputs, mask=None, training=None):
        if isinstance(inputs, list):
            assert 2 == len(inputs), "Expecting inputs to be a [input_ids, token_type_ids] list"
            input_ids, token_type_ids = inputs
        else:
            input_ids = inputs
            token_type_ids = None

        input_ids = tf.cast(input_ids, dtype=tf.int32)
        # [None, max_len] -> [None, max_len, hidden_size(768)]
        embedding_output = self.word_embeddings_layer(input_ids)

        if token_type_ids is not None:
            token_type_ids = tf.cast(token_type_ids, dtype=tf.int32)
            # [None, max_len, hidden_size] -> [None, max_len, hidden_size]
            embedding_output += self.token_type_embeddings_layer(token_type_ids)

        if self.position_embeddings_layer is not None:
            seq_len = input_ids.shape.as_list()[1]
            emb_size = self.hidden_size
            # [seq_len, 128]
            pos_embeddings = self.position_embeddings_layer(seq_len)

            # broadcast to all dimension except the last two
            broadcast_shape = [1] * (embedding_output.shape.ndims - 2) + [seq_len, emb_size]

            # [None, max_len, hidden_size]+[1, max_len, hidden_size] -> [None, max_len, hidden_size]
            embedding_output += tf.reshape(pos_embeddings, broadcast_shape)

        embedding_output = self.layer_norm(embedding_output)
        embedding_output = self.dropout_layer(embedding_output)

        # [None, seq_len, hidden_size]
        return embedding_output

    def compute_mask(self, inputs, mask=None):
        if isinstance(inputs, list):
            assert 2 == len(inputs), "Expecting inputs to be a [input_ids, token_type_ids] list"
            input_ids, token_type_ids = inputs
        else:
            input_ids = inputs
            token_type_ids = None

        return tf.not_equal(input_ids, 0)


class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self,
                 name,
                 position_embedding_name="position_embeddings",
                 initializer_range=0.02,
                 max_position_embeddings=512,
                 embedding_size=128,
                 ):
        super(PositionalEncoding, self).__init__(name=name)
        self.position_embedding_name = position_embedding_name
        self.initializer_range = initializer_range
        self.max_position_embeddings = max_position_embeddings
        self.embedding_size = embedding_size
        self.embedding_matrix = None

    def build(self, input_shape):
        self.embedding_matrix = self.add_variable(
            shape=[self.max_position_embeddings, self.embedding_size],
            initializer=create_initializer(self.initializer_range),
            name=self.position_embedding_name,
            dtype=tf.float32
        )
        super(PositionalEncoding, self).build(input_shape)

        # width = input_shape[2]
        # self.embedding_matrix = self.add_variable(
        #     shape=[self.max_position_embeddings, width],
        #     initializer=create_initializer(self.initializer_range),
        #     name=self.position_embedding_name,
        #     dtype=tf.float32
        # )

    def call(self, inputs, training=None):
        seq_len = inputs

        assert_op = tf.debugging.assert_less_equal(seq_len, self.max_position_embeddings)

        with tf.control_dependencies([assert_op]):
            full_position_embeddings = tf.slice(self.embedding_matrix,
                                                [0, 0],
                                                [seq_len, -1])
        output = full_position_embeddings
        return output




