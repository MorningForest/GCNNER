import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell
from tensorflow.contrib.crf import crf_log_likelihood, viterbi_decode
import time
import sys
import os
from utils import batch_yield, pad_sequences, conlleval
from sklearn import metrics
from attention import Att


def GCN_layer_fw(embedding_size, hidden_layer_size, features, Adjs):
    '''
    :param embedding_size:
    :param hidden_layer1_size:
    :param features: 特征矩阵
    :param Adjs: 边矩阵
    :return:
    '''
    W0_fw = tf.Variable(tf.random_uniform([embedding_size, hidden_layer_size], 0, 0.1), name='W0_fw')
    b0_fw = tf.Variable(tf.random_uniform([hidden_layer_size], -0.1, 0.1), name='b0_fw')
    left_X1_projection_fw = lambda x: tf.matmul(x, W0_fw) + b0_fw
    left_X1_fw = tf.map_fn(left_X1_projection_fw, features)  #B,N,H    B,N,N
    X1_fw = tf.nn.relu(tf.matmul(Adjs, left_X1_fw))
    return X1_fw

def load_GCN_adjs(embedd_input):
    pass


class GCNNerModel(object):

    def __init__(self, embedding_size, dropout, hidden_layer_size, output_size, vocab
                     , embedding_weight, update_embedding_weight, optimizer, clip_grad
                     , summary_path, config, epoch, tag2label, batch_size, model_path
                     , lr, logger, result_path, shuffle=False, flag=None, conv=False
                 ):
        self.embedding_size = embedding_size
        self.dropout = dropout
        self.lr = lr
        self.hidden_layer_size = hidden_layer_size
        self.output_size = output_size
        self.vocab_size = len(vocab)
        self.vocab = vocab
        self.embedding_weight = embedding_weight
        self.update_embedding_weight = update_embedding_weight
        self.optimizer = optimizer
        self.clip_grad = clip_grad
        self.summary_path = summary_path
        self.config = config
        self.epoch_num = epoch
        self.tag2label = tag2label
        self.batch_size = batch_size
        self.model_path = model_path
        self.logger = logger
        self.result_path = result_path
        self.shuffle = shuffle
        self.flags = flag
        self.conv = conv


    def _build_graph(self):
        self.__add_placeholders()
        self.__lookup_layer_op()
        self.__biLSTM_layer_op()
        self.__cnn_att_layer_op()
        self.__loss_op()
        self.__trainstep_op()
        self.__init_op()

    def __add_placeholders(self):
        self.word_ids = tf.placeholder(tf.int32, shape=[None, None], name="word_ids")
        self.labels = tf.placeholder(tf.int32, shape=[None, None], name="labels")
        self.sequence_lengths = tf.placeholder(tf.int32, shape=[None], name="sequence_lengths")

        self.dropout_pl = tf.placeholder(dtype=tf.float32, shape=[], name="dropout")
        self.lr_pl = tf.placeholder(dtype=tf.float32, shape=[], name="lr")

    def __lookup_layer_op(self):
        with tf.variable_scope("words"):
            _word_embeddings = tf.Variable(self.embedding_weight,
                                           dtype=tf.float32,
                                           trainable=self.update_embedding_weight,
                                           name="_word_embeddings")
            word_embeddings = tf.nn.embedding_lookup(params=_word_embeddings,
                                                     ids=self.word_ids,
                                                     name="word_embeddings")
        # self.word_embeddings = tf.nn.dropout(word_embeddings, self.dropout_pl)
        self.word_embeddings = word_embeddings

    def __biLSTM_layer_op(self):
        load_GCN_adjs(self.word_embeddings)
        with tf.variable_scope("bi-lstm"):
            cell_fw = LSTMCell(self.hidden_layer_size)
            cell_bw = LSTMCell(self.hidden_layer_size)
            if self.flags == 'att':
                att1 = Att(self.dropout_pl)
                att_out = att1.multiAttention_layer_op(
                    queries=self.word_embeddings, keys=self.word_embeddings,
                    values=self.word_embeddings, num_heads=6,
                    scope="bilstm_attention"
                )
                (output_fw_seq, output_bw_seq), _ = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw=cell_fw,
                    cell_bw=cell_bw,
                    inputs=att_out,
                    sequence_length=self.sequence_lengths,
                    dtype=tf.float32)
            else:
                (output_fw_seq, output_bw_seq), _ = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw=cell_fw,
                    cell_bw=cell_bw,
                    inputs=self.word_embeddings,
                    sequence_length=self.sequence_lengths,
                    dtype=tf.float32)
            output = tf.concat([output_fw_seq, output_bw_seq, self.Att_Conv], axis=-1)
            output = tf.nn.dropout(output, self.dropout_pl)

        with tf.variable_scope("proj"):
            W = tf.get_variable(name="W",
                                shape=[2 * self.hidden_layer_size + 128, self.output_size],
                                initializer=tf.contrib.layers.xavier_initializer(),
                                dtype=tf.float32)

            b = tf.get_variable(name="b",
                                shape=[self.output_size],
                                initializer=tf.zeros_initializer(),
                                dtype=tf.float32)

            s = tf.shape(output)
            output = tf.reshape(output, [-1, 2 * self.hidden_layer_size + 128])
            pred = tf.matmul(output, W) + b

            self.logits = tf.reshape(pred, [-1, s[1], self.output_size])

    def __cnn_att_layer_op(self):
        with tf.variable_scope("AttConv", initializer=tf.contrib.layers.xavier_initializer()):
            kernel = tf.get_variable(shape=[1, 3, 300, 512], initializer=tf.contrib.layers.xavier_initializer(),
                                     name='kernel')
            kernel1 = tf.get_variable(shape=[1, 5, 512, 512], initializer=tf.contrib.layers.xavier_initializer(),
                                      name='kernel1')
            # kernel2 = tf.get_variable(shape=[1, 3, 360, 420], initializer=tf.contrib.layers.xavier_initializer(),
            #                           name='kernel2')
            # kernel3 = tf.get_variable(shape=[1, 5, 512, 512], initializer=tf.contrib.layers.xavier_initializer(),
            #                           name='kernel3')
            # output = []
            att = Att(self.dropout_pl)
            Attoutput = att.multiAttention_layer_op(
                queries=self.word_embeddings,
                keys=self.word_embeddings,
                values=self.word_embeddings,
                num_heads=6,
                scope='att1'
            )
            Attoutput = tf.expand_dims(Attoutput, 1)
            conv1 = tf.nn.atrous_conv2d(
                Attoutput,
                kernel,
                rate=1,
                padding='SAME',
                name='conv1'
            )
            input = tf.squeeze(conv1, 1)
            att1 = Att(self.dropout_pl)
            Attoutput1 = att1.multiAttention_layer_op(
                queries=input,
                keys=input,
                values=input,
                num_heads=8,
                scope='att2'
            )
            Attoutput1 = tf.expand_dims(Attoutput1, 1)
            conv2 = tf.nn.atrous_conv2d(
                Attoutput1,
                kernel1,
                rate=2,
                padding='SAME',
                name='conv2'
            )
            output = tf.squeeze(conv2, 1)
            output = tf.layers.dense(output, 256, use_bias=True, kernel_initializer=tf.contrib.layers.xavier_initializer())
            output = tf.layers.dense(output, 128, use_bias=True, kernel_initializer=tf.contrib.layers.xavier_initializer())
            self.Att_Conv = output

    def __loss_op(self):
        log_likelihood, self.transition_params = crf_log_likelihood(inputs=self.logits,
                                                               tag_indices=self.labels,
                                                               sequence_lengths=self.sequence_lengths)
        self.preds, _ = tf.contrib.crf.crf_decode(self.logits, self.transition_params, self.sequence_lengths)
        self.loss = -tf.reduce_mean(log_likelihood)
        tf.summary.scalar("loss", self.loss)

    def __trainstep_op(self):
        with tf.variable_scope("train_step"):
            self.global_step = tf.Variable(0, name="global_step", trainable=False)
            if self.optimizer == 'Adam':
                optim = tf.train.AdamOptimizer(learning_rate=self.lr_pl, beta2=5e-4)
            elif self.optimizer == 'Adadelta':
                optim = tf.train.AdadeltaOptimizer(learning_rate=self.lr_pl)
            elif self.optimizer == 'Adagrad':
                optim = tf.train.AdagradOptimizer(learning_rate=self.lr_pl)
            elif self.optimizer == 'RMSProp':
                optim = tf.train.RMSPropOptimizer(learning_rate=self.lr_pl)
            elif self.optimizer == 'Momentum':
                optim = tf.train.MomentumOptimizer(learning_rate=self.lr_pl, momentum=0.9)
            elif self.optimizer == 'SGD':
                optim = tf.train.GradientDescentOptimizer(learning_rate=self.lr_pl)
            else:
                optim = tf.train.GradientDescentOptimizer(learning_rate=self.lr_pl)

            grads_and_vars = optim.compute_gradients(self.loss)
            grads_and_vars_clip = [[tf.clip_by_value(g, -self.clip_grad, self.clip_grad), v] for g, v in grads_and_vars]
            self.train_op = optim.apply_gradients(grads_and_vars_clip, global_step=self.global_step)

    def __init_op(self):
        self.init_op = tf.global_variables_initializer()

    def add_summary(self, sess):
        """

        :param sess:
        :return:
        """
        self.merged = tf.summary.merge_all()
        self.file_writer = tf.summary.FileWriter(self.summary_path, sess.graph)

    def train(self, train, dev):
        """

        :param train:
        :param dev:
        :return:
        """
        saver = tf.train.Saver(tf.global_variables())

        with tf.Session(config=self.config) as sess:
            sess.run(self.init_op)
            self.add_summary(sess)

            for epoch in range(self.epoch_num):
                self.run_one_epoch(sess, train, dev, self.tag2label, epoch, saver)

    def run_one_epoch(self, sess, train, dev, tag2label, epoch, saver):
        """
        :param sess:
        :param train:
        :param dev:
        :param tag2label:
        :param epoch:
        :param saver:
        :return:
        """
        num_batches = (len(train) + self.batch_size - 1) // self.batch_size

        start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        batches = batch_yield(train, self.batch_size, self.vocab, tag2label, shuffle=self.shuffle)
        for step, (seqs, labels) in enumerate(batches):

            sys.stdout.write(' processing: {} batch / {} batches.'.format(step + 1, num_batches) + '\r')
            step_num = epoch * num_batches + step + 1
            feed_dict, _ = self.get_feed_dict(seqs, labels, self.lr, self.dropout)
            _, loss_train, summary, step_num_ = sess.run([self.train_op, self.loss, self.merged, self.global_step],
                                                         feed_dict=feed_dict)

            if step + 1 == 1 or (step + 1) % 10 == 0 or step + 1 == num_batches:
                print(
                    '{} epoch {}, step {}, loss: {:.4}, global_step: {}'.format(start_time, epoch + 1, step + 1,
                                                                                loss_train, step_num))

            self.file_writer.add_summary(summary, step_num)

            if step + 1 == num_batches:
                saver.save(sess, self.model_path, global_step=step_num)

        self.logger.info('===========validation / test===========')
        label_list_dev, seq_len_list_dev = self.dev_one_epoch(sess, dev)
        self.evaluate(label_list_dev, seq_len_list_dev, dev, epoch)

    def get_feed_dict(self, seqs, labels=None, lr=None, dropout=None):
        """

        :param seqs:
        :param labels:
        :param lr:
        :param dropout:
        :return: feed_dict
        """
        word_ids, seq_len_list = pad_sequences(seqs, pad_mark=0)#填充0

        feed_dict = {self.word_ids: word_ids,
                     self.sequence_lengths: seq_len_list}
        if labels is not None:
            labels_, _ = pad_sequences(labels, pad_mark=0)
            feed_dict[self.labels] = labels_
        if lr is not None:
            feed_dict[self.lr_pl] = lr
        if dropout is not None:
            feed_dict[self.dropout_pl] = dropout

        return feed_dict, seq_len_list

    def dev_one_epoch(self, sess, dev):
        """

        :param sess:
        :param dev:
        :return:
        """
        label_list, seq_len_list = [], []
        for seqs, labels in batch_yield(dev, self.batch_size, self.vocab, self.tag2label, shuffle=self.shuffle):
            label_list_, seq_len_list_ = self.predict_one_batch(sess, seqs)
            label_list.extend(label_list_)
            seq_len_list.extend(seq_len_list_)
        return label_list, seq_len_list

    def predict_one_batch(self, sess, seqs):
        """
        预测数据集，解码最优序列
        :param sess:
        :param seqs:
        :return: label_list
                 seq_len_list
        """
        feed_dict, seq_len_list = self.get_feed_dict(seqs, dropout=1.0)

        logits, transition_params = sess.run([self.logits, self.transition_params],
                                             feed_dict=feed_dict)
        label_list = []
        for logit, seq_len in zip(logits, seq_len_list):
            viterbi_seq, _ = viterbi_decode(logit[:seq_len], transition_params)
            label_list.append(viterbi_seq)
        return label_list, seq_len_list

    def evaluate(self, label_list, data, epoch=None):
        """
        :param label_list:
        :param data:
        :param epoch:
        :return:
        """
        label2tag = {}
        for tag, label in self.tag2label.items():
            label2tag[label] = tag if label != 0 else label
        model_predict = []
        for label_, (sent, tag) in zip(label_list, data):
            tag_ = [label2tag[label__] for label__ in label_]
            sent_res = []
            if len(label_) != len(sent):
                print(sent)
                print(len(label_))
                print(tag)
            for i in range(len(sent)):
                sent_res.append([sent[i], tag[i], tag_[i]])
            model_predict.append(sent_res)
        epoch_num = str(epoch+1) if epoch != None else 'test'
        label_path = os.path.join(self.result_path, 'label_' + epoch_num)
        metric_path = os.path.join(self.result_path, 'result_metric_' + epoch_num)
        for _ in conlleval(model_predict, label_path, metric_path):
            self.logger.info(_)
