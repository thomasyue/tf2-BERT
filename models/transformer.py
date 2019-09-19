import tensorflow as tf
from models.attention import AttentionLayer
from models.utils import create_initializer, gelu


class TransformerEncoderLayer(tf.keras.layers.Layer):
    def __init__(self,
                 name,
                 hidden_size=768,
                 num_layers=12,
                 num_heads=12,
                 intermediate_size=3072,
                 intermediate_activation=gelu,
                 dropout_rate=0.1,
                 attention_dropout_rate=0.1,
                 initializer_range=0.2):
        super(TransformerEncoderLayer, self).__init__(name=name)
        if hidden_size % num_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, num_heads))
        attention_head_size = int(hidden_size / num_heads)

        self.attention_heads = []
        self.attention_outputs = []
        self.attention_layer_norms = []
        self.intermediate_outputs = []
        self.layer_outputs = []
        self.output_layer_norms = []

        for layer_idx in range(num_layers):
            attention_head = AttentionLayer(
                name="layer_{}/attention/self".format(layer_idx),
                num_heads=num_heads,
                size_per_head=attention_head_size,
                attention_dropout=attention_dropout_rate,
                initializer_range=initializer_range
            )
            self.attention_heads.append(attention_head)

            attention_output = tf.keras.layers.Dense(
                hidden_size,
                name="layer_{}/attention/output/dense".format(layer_idx),
                kernel_initializer=create_initializer(initializer_range)
            )
            self.attention_outputs.append(attention_output)

            attention_layer_norm = tf.keras.layers.LayerNormalization(
                name="layer_{}/attention/output/LayerNorm".format(layer_idx))
            self.attention_layer_norms.append(attention_layer_norm)

            intermediate_output = tf.keras.layers.Dense(
                intermediate_size,
                activation=intermediate_activation,
                kernel_initializer=create_initializer(initializer_range)
            )
            self.intermediate_outputs.append(intermediate_output)

            layer_output = tf.keras.layers.Dense(
                hidden_size,
                kernel_initializer=create_initializer(initializer_range),
                name="layer_{}/attention/output/dense".format(layer_idx)
            )
            self.layer_outputs.append(layer_output)

            output_layer_norm = tf.keras.layers.LayerNormalization(
                name="layer_{}/output/LayerNorm".format(layer_idx)
            )
            self.output_layer_norms.append(output_layer_norm)

        self.dropout = tf.keras.layers.Dropout(dropout_rate)

        self.hidden_size = hidden_size
        self.num_layers = num_layers

    def call(self, inputs, mask=None, out_layer_idxs=None, training=None):
        # input_shape = self.get_shape_list(inputs, expected_rank=3)
        # batch_size, seq_length, input_width = input_shape[0], input_shape[1], input_shape[2]
        #
        # if input_width != self.hidden_size:
        #     raise ValueError("The width of the input tensor (%d) != hidden size (%d)" %
        #                      (input_width, self.hidden_size))

        prev_output = inputs
        all_layer_outputs = []
        for layer_idx in range(self.num_layers):
            layer_input = prev_output

            attention_heads = []
            attention_head = self.attention_heads[layer_idx](layer_input,
                                                             mask=mask,
                                                             training=training)
            attention_heads.append(attention_head)

            if len(attention_heads) == 1:
                attention_output = attention_heads[0]
            else:
                attention_output = tf.concat(attention_heads, axis=-1)

            attention_output = self.attention_outputs[layer_idx](attention_output)
            attention_output = self.dropout(attention_output, training=training)
            attention_output = self.attention_layer_norms[layer_idx](attention_output+layer_input)

            intermediate_output = self.intermediate_outputs[layer_idx](attention_output)

            layer_output = self.layer_outputs[layer_idx](intermediate_output)
            layer_output = self.dropout(layer_output, training=training)
            layer_output = self.output_layer_norms[layer_idx](layer_output + attention_output)
            prev_output = layer_output
            all_layer_outputs.append(layer_output)

        if out_layer_idxs is None:
            final_output = all_layer_outputs[-1]

        else:
            final_output = []
            for idx in out_layer_idxs:
                final_output.append(all_layer_outputs[idx])
            final_output = tuple(final_output)

        return final_output
