import tensorflow as tf
import copy
from models.embeddings import *
from models.transformer import *

class BertModelLayer(tf.keras.layers.Layer):
    """
    BERT Model("Bidirectional Embedding Representations from a Transformer").
    From: https://arxiv.org/pdf/1810.04805.pdf
    """
    def __init__(self,
                 config,
                 name='bert',
                 ):
        super(BertModelLayer, self).__init__(name=name)
        config = copy.deepcopy(config)
        self.embedding_layer = BertEmbeddingsLayer(
            name="embeddings",
            vocab_size=config.vocab_size,
            use_token_type=config.use_token_type,
            token_type_vocab_size=config.token_type_vocab_size,
            use_position_embeddings=True,
            hidden_size=config.hidden_size,
            initializer_range=config.initializer_range,
            word_embedding_name="word_embeddings",
            token_type_embedding_name="token_type_embeddings",
            dropout_rate=config.hidden_dropout_prob
        )

        self.encoders_layer = TransformerEncoderLayer(
            name='encoder',
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            num_heads=config.num_attention_heads,
            intermediate_size=config.intermediate_size,
            intermediate_activation=self.get_activation(config.hidden_act),
            dropout_rate=config.hidden_dropout_rate,
            attention_dropout_rate=config.attention_dropout_rate,
            initializer_range=config.initializer_range
        )

        self.pooler = tf.keras.layers.Dense(
            units=config.hidden_size,
            activation=tf.nn.tanh,
            # activation='tanh',
            kernel_initializer=self.create_initializer(config.initializer_range)
        )

    def call(self, inputs, return_sequences=False, mask=None, training=None):
        if mask is None:
            mask = self.embeddings_layer.compute_mask(inputs)

        embedding_output = self.embeddings_layer(inputs, mask=mask, training=training)
        output = self.encoders_layer(embedding_output, mask=mask, training=training)

        if return_sequences:
            # [B, seq_len, hidden_size]
            return output
        else:
            first_token_tensor = tf.squeeze(output[-1][:, 0:1, :])
            pooled_output = self.pooler(first_token_tensor)
            return pooled_output

    def apply_adapter_freeze(self):
        pass