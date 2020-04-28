import tensorflow as tf

class Att(object):

    def __init__(self, dropout):
        self.dropout_pl = dropout

    def mask(self, inputs, queries=None, keys=None, type=None):
        '''
                对Keys或Queries进行遮盖
                :param inputs: (N, T_q, T_k)
                :param queries: (N, T_q, d)
                :param keys: (N, T_k, d)
                :return:
        '''
        padding_num = -2 ** 32 + 1
        if type in ("k", "key", "keys"):
            # Generate masks
            masks = tf.sign(tf.reduce_sum(tf.abs(keys), axis=-1))  # (N, T_k)
            masks = tf.expand_dims(masks, 1)  # (N, 1, T_k)
            masks = tf.tile(masks, [1, tf.shape(queries)[1], 1])  # (N, T_q, T_k)
            # Apply masks to inputs
            paddings = tf.ones_like(inputs) * padding_num
            outputs = tf.where(tf.equal(masks, 0), paddings, inputs)  # (N, T_q, T_k)
        elif type in ("q", "query", "queries"):
            # Generate masks
            masks = tf.sign(tf.reduce_sum(tf.abs(queries), axis=-1))  # (N, T_q)
            masks = tf.expand_dims(masks, -1)  # (N, T_q, 1)
            masks = tf.tile(masks, [1, 1, tf.shape(keys)[1]])  # (N, T_q, T_k)

            # Apply masks to inputs
            outputs = inputs * masks
        elif type in ("f", "future", "right"):
            diag_vals = tf.ones_like(inputs[0, :, :])  # (T_q, T_k)
            tril = tf.linalg.LinearOperatorLowerTriangular(diag_vals).to_dense()  # (T_q, T_k)
            masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(inputs)[0], 1, 1])  # (N, T_q, T_k)

            paddings = tf.ones_like(masks) * padding_num
            outputs = tf.where(tf.equal(masks, 0), paddings, inputs)
        else:
            print("Check if you entered type correctly!")

        return outputs

    def ln(inputs, epsilon=1e-8, scope="ln"):
        '''
            使用层归一layer normalization
            tensorflow 在实现 Batch Normalization（各个网络层输出的归一化）时，主要用到nn.moments和batch_normalization
            其中moments作用是统计矩，mean 是一阶矩，variance 则是二阶中心矩
            tf.nn.moments 计算返回的 mean 和 variance 作为 tf.nn.batch_normalization 参数进一步调用
            :param inputs: 一个有2个或更多维度的张量，第一个维度是batch_size
            :param epsilon: 很小的数值，防止区域划分错误
            :param scope:
            :return: 返回一个与inputs相同shape和数据的dtype
            '''
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            print(inputs, type(inputs))
            inputs_shape = inputs.get_shape()
            params_shape = inputs_shape[-1:]

            mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
            beta = tf.get_variable("beta", params_shape, initializer=tf.zeros_initializer())
            gamma = tf.get_variable("gamma", params_shape, initializer=tf.ones_initializer())
            normalized = (inputs - mean) / ((variance + epsilon) ** (.5))
            outputs = gamma * normalized + beta

        return outputs

    def scaled_dot_product_attention(self, Q, K, V, dropout_rate=0.7, training=True, causality=False,
                                     scope="scaled_dot_product_attention"):
        with tf.variable_scope(scope):
            d_k = Q.get_shape().as_list()[-1]
            # dot product
            print(K.shape, Q.shape, V.shape)
            outputs = tf.matmul(Q, tf.transpose(K, [0, 2, 1]))  # (N, T_q, T_k)
            # scale
            outputs /= d_k ** 0.5
            # key masking
            outputs = self.mask(outputs, Q, K, type="key")
            # causality or future blinding masking
            if causality:
                outputs = self.mask(outputs, type="future")
            outputs = tf.nn.softmax(outputs)
            attention = tf.transpose(outputs, [0, 2, 1])
            tf.summary.image("attention", tf.expand_dims(attention[:1], -1))
            # query masking
            outputs = self.mask(outputs, Q, K, type="query")
            if training:
                outputs = tf.nn.dropout(outputs, dropout_rate)
            outputs = tf.matmul(outputs, V)  # (N, T_q, d_v)
        return outputs

    def multiAttention_layer_op(self, queries, keys, values, num_heads,
                                causality=False, scope="multihead_attention"):
        '''
          :param queries: 三维张量[N, T_q, d_model]
          :param keys: 三维张量[N, T_k, d_model]
          :param values: 三维张量[N, T_k, d_model]
          :param num_heads: heads数
          :param dropout_rate:
          :param training: 控制dropout机制
          :param causality: 控制是否遮盖
          :param scope:
          :return: 三维张量(N, T_q, C)
        '''
        d_model = queries.get_shape().as_list()[-1]
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            # Linear projections
            Q = tf.layers.dense(queries, d_model, use_bias=False)  # (N, T_q, d_model)
            K = tf.layers.dense(keys, d_model, use_bias=False)  # (N, T_k, d_model)
            V = tf.layers.dense(values, d_model, use_bias=False)  # (N, T_k, d_model)
            # Split and concat
            Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0)  # (h*N, T_q, d_model/h)
            K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0)  # (h*N, T_k, d_model/h)
            V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0)  # (h*N, T_k, d_model/h)
            # Attention
            outputs = self.scaled_dot_product_attention(Q_, K_, V_, self.dropout_pl, training=True,
                                                        causality=False)
            # Restore shape
            outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2)  # (N, T_q, d_model)
            # Residual connection
            outputs += queries
            # Normalize
            # outputs = self.ln(outputs)
            return outputs