import tensorflow as tf


class AttentionLayer(tf.keras.layers.Layer):
    def __init__(self,
                 name,
                 num_heads=1,
                 size_per_head=512,
                 query_activation=None,
                 key_activation=None,
                 value_activation=None,
                 attention_dropout=0.1,
                 initializer_range=0.02,):
        super(AttentionLayer, self).__init__(name=name)
        self.num_heads = num_heads
        self.size_per_head = size_per_head

        self.query_layer = tf.keras.layers.Layer(
            num_heads * size_per_head,
            activation=query_activation,
            kernel_initializer=self.create_initializer(initializer_range),
            name="query"
        )
        self.key_layer = tf.keras.layers.Layer(
            num_heads * size_per_head,
            activation=key_activation,
            kernel_initializer=self.create_initializer(initializer_range),
            name="key"
        )
        self.value_layer = tf.keras.layers.Layer(
            num_heads * size_per_head,
            activation=value_activation,
            kernel_initializer=self.create_initializer(initializer_range),
            name="value"
        )

        self.dropout_layer = tf.keras.layers.Dropout(attention_dropout)

        """
        B = batch size
        F = sequence length 'from_tensor'
        T = sequence length 'to_tensor'
        N = num_heads
        H = size_per_head
        """

    def transpose_for_scores(self, input_tensor, batch_size, num_heads, seq_len, size_per_head):
        output_shape = [batch_size, seq_len, num_heads, size_per_head]
        output_tensor = tf.reshape(input_tensor, output_shape)

        # [B, N, F, H]
        return tf.transpose(output_tensor, [0, 2, 1, 3])

    @staticmethod
    def create_attention_mask_from_input_mask(from_shape, to_mask):
        """
        Creates 3D attention.
        :param from_shape:  [B, F]
        :param to_mask:  [B, T]
        :return: [B, F, T]
        """
        # [B, T] -> [B, 1, T]
        mask = tf.cast(tf.expand_dims(to_mask, axis=1), dtype=tf.float32)
        # [B, F] -> [B, F, 1]
        ones = tf.expand_dims(tf.ones(shape=[from_shape[:2]], dtype=tf.float32), axis=-1)
        # [B, 1, T] * [B, F, 1] -> [B, F, T]
        mask = ones * mask
        return mask

    def call(self, inputs, mask=None, training=None):
        # [B, F, from_width]
        from_tensor = inputs
        # [B, T, to_width]
        to_tensor = inputs

        if mask is None:
            sh = self.get_shape_list(from_tensor)
            mask = tf.ones(sh[:2], dtype=tf.int32)

        # [B, F, T]
        attention_mask = AttentionLayer.create_attention_mask(from_shape=tf.shape(from_tensor),
                                                              to_mask=mask)
        # from_tensor.shape = [B, F, from_width]
        input_shape = tf.shape(from_tensor)
        batch_size, from_seq_len, from_width = input_shape[0], input_shape[1], input_shape[2]
        to_seq_len = from_seq_len

        # [B, F, from_width] -> [B, F, N*H]
        query = self.query_layer(from_tensor)

        # [B, T, to_width] -> [B, T, N*H]
        key = self.key_layer(to_tensor)

        # [B, T, to_width] -> [B, T, N*H]
        value = self.value_layer(to_tensor)

        # [B, F, N*H] -> [B, N, F, H]
        query = self.transpose_for_scores(input_tensor=query,
                                          batch_size=batch_size,
                                          num_heads=self.num_heads,
                                          seq_len=from_seq_len,
                                          size_per_head=self.size_per_head)
        # [B, T, N*H] -> [B, N, T, H]
        key = self.transpose_for_scores(input_tensor=key,
                                        batch_size=batch_size,
                                        num_heads=self.num_heads,
                                        seq_len=to_seq_len,
                                        size_per_head=self.size_per_head)

        # [B, N, F, H] * [B, N, H, T] -> [B, N, F, T]
        attention_score = tf.matmul(query, key, transpose_b=True)  # tf.transpose(key, perm=[0, 1, 3, 2])
        attention_score = attention_score / tf.sqrt(float(self.size_per_head))

        if attention_mask is not None:
            # [B, F, T] -> [B, 1, F, T]
            attention_mask = tf.expand_dims(attention_mask, axis=1)

            # attention_mask is 1 for position and 0 for mask, but we want 0 for mask and -10000 for mask.
            # {1: position, 0: mask} -> {0: position, -10000: mask}
            adder = (1.0 - tf.cast(attention_mask, dtype=tf.float32)) * -10000.0
            attention_score += adder

        # [B, N, F, T]
        attention_prob = tf.nn.softmax(attention_score)
        attention_prob = self.dropout(attention_prob, training=training)

        # [B, T, N*H] -> [B, T, N, H]
        value = tf.reshape(value, [batch_size, to_seq_len, self.num_heads, self.size_per_head])

        # [B, T, N, H] -> [B, N, T, H]
        value = tf.transpose(value, perm=[0, 2, 1, 3])

        # [B, N, F, T] * [B, N, T, H] -> [B, N, F, H]
        context_layer = tf.matmul(attention_prob, value)

        # [B, N, F, H] -> [B, F, N, H]
        context_layer = tf.transpose(context_layer, perm=[0, 2, 1, 3])

        output_shape = [batch_size, from_seq_len, self.num_heads*self.size_per_head]

        # [B, F, N, H] -> [B, F, N*H]
        context_layer = tf.reshape(context_layer, output_shape)
        return context_layer

    def compute_mask(self, inputs, mask=None):
        return mask   # [B, F]