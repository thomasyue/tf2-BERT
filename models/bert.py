import tensorflow as tf
import copy
from models.embeddings import *
from models.transformer import *
from models.utils import freeze_bert_layers, create_initializer, get_activation


class BertModelLayer(tf.keras.layers.Layer):
    """
    BERT Model("Bidirectional Encoder Representations from a Transformer").
    See: https://arxiv.org/pdf/1810.04805.pdf
    """
    def __init__(self,
                 config,
                 name='bert',
                 ):
        super(BertModelLayer, self).__init__(name=name)
        config = copy.deepcopy(config)
        self.embeddings_layer = BertEmbeddingsLayer(
            name="embeddings",
            vocab_size=config.vocab_size,
            use_token_type=config.use_token_type,
            token_type_vocab_size=config.type_vocab_size,
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
            num_layers=config.num_hidden_layers,
            num_heads=config.num_attention_heads,
            intermediate_size=config.intermediate_size,
            intermediate_activation=get_activation(config.hidden_act),
            dropout_rate=config.hidden_dropout_prob,
            attention_dropout_rate=config.attention_probs_dropout_prob,
            initializer_range=config.initializer_range
        )

        self.pooler = tf.keras.layers.Dense(
            units=config.hidden_size,
            activation=tf.nn.tanh,
            # activation='tanh',
            kernel_initializer=create_initializer(config.initializer_range)
        )

    def apply_adapter_freeze(self):
        """
        from arXiv:1902.00751
        freeze layers except LayerNorm
        """
        freeze_bert_layers(self)

    def call(self, inputs, mask=None, return_sequences=True, return_pool=True, out_layer_idxs=None, training=None):
        """
        :param inputs: [B, max_len]
        :param mask:   [B, max_len]
        :param return_sequences: bool, return [B, max_len, hidden_size]
        :param return_pool: bool, return CLS [B, hidden_size]
        :param out_layer_idxs: list, return specific layer outputs from transformers. num_layers
        :param training: bool
        :return: default: tuple(output, pooled_output)
        """
        # if out_layer_idxs is not None and not return_sequences:
        #     raise ValueError(
        #         "When out_layer_idxs is not None, model has to return_sequences")

        # if out_layer_idxs is not None:
        #     return_sequences = True

        if mask is None:
            mask = self.embeddings_layer.compute_mask(inputs)

        embedding_output = self.embeddings_layer(inputs, mask=mask, training=training)

        # [B, seq_len, hidden_size]
        output = self.encoders_layer(embedding_output, mask=mask, out_layer_idxs=out_layer_idxs, training=training)

        if out_layer_idxs is None:
            first_token_tensor = tf.squeeze(output[:, 0:1, :])
        else:
            first_token_tensor = tf.squeeze(output[-1][:, 0:1, :])

        pooled_output = self.pooler(first_token_tensor)

        if not return_pool and not return_sequences:
            raise ValueError("return_sequences and return_pool can't be both False")

        elif not return_sequences:
            return pooled_output

        elif not return_pool:
            return output

        else:
            return output, pooled_output





