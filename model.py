import tensorflow as tf
import numpy as np
import os
from sklearn.metrics import f1_score


class AttentionRNN(object):
    def __init__(self, config, pre_weight=None):
        self.name = config.model_name
        self.max_len = config.max_len
        self.hidden_size = config.hidden_size
        self.vocab_size = config.vocab_size
        self.embedding_size = config.embedding_size
        self.num_classes = config.num_classes
        self.pre_weight = pre_weight

        with tf.compat.v1.variable_scope(self.name):
            self.X = tf.compat.v1.placeholder(tf.int32, [None, self.max_len], name='X')
            self.y = tf.compat.v1.placeholder(tf.int32, [None, self.num_classes], name='y')
            self.learning_rate = tf.compat.v1.placeholder(tf.float32, name='learning_rate')
            self.l2_scale = tf.compat.v1.placeholder(tf.float32, name='l2_scale')
            self.emb_keep_prob = tf.compat.v1.placeholder(tf.float32, name='emb_drop_rate')
            self.rnn_keep_prob = tf.compat.v1.placeholder(tf.float32, name='rnn_drop_rate')
            self.att_keep_prob = tf.compat.v1.placeholder(tf.float32, name='att_drop_rate')

            self.build_graph(config)

    def build_graph(self, config):
        with tf.compat.v1.variable_scope('embedding_layer'):
            if self.pre_weight is not None:
                self.init_embedding = tf.constant(self.pre_weight, dtype=tf.float32)
                self.embedding = tf.compat.v1.get_variable('embedding_matrix',
                                                           initializer=self.init_embedding,
                                                           trainable=config.train_pre_weight)
            else:
                self.init_embedding = tf.random.uniform([self.vocab_size, self.embedding_size])
                self.embedding = tf.compat.v1.get_variable('embedding_matrix',
                                                           initializer=self.init_embedding,
                                                           trainable=True)
            self.X_embedding = tf.nn.embedding_lookup(self.embedding, self.X, name='X_embedding')
            self.X_dropout = tf.nn.dropout(self.X_embedding, self.emb_keep_prob, name='X_dropout')

        with tf.compat.v1.variable_scope('bilstm_layer'):
            self.forward_cell = tf.nn.rnn_cell.LSTMCell(config.hidden_size,
                                                        config.use_peepholes,
                                                        initializer=tf.initializers.glorot_normal,
                                                        name='forward')
            self.forward_dropout = tf.compat.v1.nn.rnn_cell.DropoutWrapper(self.forward_cell,
                                                                           self.rnn_keep_prob)
            self.backward_cell = tf.nn.rnn_cell.LSTMCell(config.hidden_size,
                                                         config.use_peepholes,
                                                         initializer=tf.initializers.glorot_normal,
                                                         name='backward')
            self.backward_dropout = tf.compat.v1.nn.rnn_cell.DropoutWrapper(self.backward_cell,
                                                                            self.rnn_keep_prob)
            self.rnn_outputs, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw=self.forward_dropout,
                                                                  cell_bw=self.backward_dropout,
                                                                  inputs=self.X_dropout,
                                                                  sequence_length=self.get_seq_len(self.X),
                                                                  dtype=tf.float32)
            if config.bilstm_merge_mode == 'add':
                self.rnn_outputs = tf.add(self.rnn_outputs[0], self.rnn_outputs[1], name='bilstm_merge')
            elif config.bilstm_merge_mode == 'concat':
                self.rnn_outputs = tf.concat(self.rnn_outputs, 2)
            else:
                raise ValueError("merge mode dose not support other than 'add' or 'concat'.")

        with tf.compat.v1.variable_scope('attention_layer'):
            self.M = tf.nn.tanh(self.rnn_outputs, name='M')
            self.w = tf.compat.v1.get_variable('w', [self.rnn_outputs.shape[2].value],
                                               initializer=tf.initializers.glorot_normal)
            self.att_weight = tf.nn.softmax(tf.tensordot(self.M, self.w, axes=1), name='att_weight')
            self.attended = tf.reduce_sum(self.rnn_outputs * tf.expand_dims(self.att_weight, -1), 1)

            if config.att_activation == 'tanh':
                self.attended = tf.nn.tanh(self.attended, name='attended_output')
            elif config.att_activation == 'relu':
                self.attended = tf.nn.relu(self.attended, name='attended_output')
            else:
                raise ValueError("attention activation dose not support other than 'tanh' or 'relu'.")
            self.att_dropout = tf.nn.dropout(self.attended, self.att_keep_prob)

        with tf.compat.v1.variable_scope('output_layer'):
            self.logits = tf.layers.dense(self.att_dropout, self.num_classes, name='logits',
                                          kernel_initializer=tf.initializers.glorot_normal)
            self.prediction = tf.argmax(self.logits, 1, name='prediction')
            self.correct_prediction = tf.equal(self.prediction, tf.argmax(self.y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32), name='acc')

        self.prediction_loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=self.y)
        self.l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in tf.compat.v1.trainable_variables()])
        self.cost = tf.reduce_mean(self.prediction_loss) + (self.l2_scale * self.l2_loss)

        if config.optimizer == 'adadelta':
            self.optimizer = tf.compat.v1.train.AdadeltaOptimizer(self.learning_rate,
                                                                  config.lr_decay,
                                                                  config.epsilon).minimize(self.cost)
        elif config.optimizer == 'adam':
            self.optimizer = tf.compat.v1.train.AdamOptimizer(self.learning_rate).minimize(self.cost)
        else:
            raise ValueError("optimizer does not support other than 'adadelta' or 'adam'.")

        self.saver = tf.compat.v1.train.Saver(max_to_keep=50)

    def optimize(self, sess, X, y, learning_rate, l2_scale, keep_probs):
        feed_dict = {
            self.X: X,
            self.y: y,
            self.learning_rate: learning_rate,
            self.l2_scale: l2_scale,
            self.emb_keep_prob: keep_probs[0],
            self.rnn_keep_prob: keep_probs[1],
            self.att_keep_prob: keep_probs[2]
        }
        return sess.run([self.cost, self.optimizer], feed_dict=feed_dict)

    def predict(self, sess, X):
        feed_dict = {
            self.X: X,
            self.emb_keep_prob: 1.,
            self.rnn_keep_prob: 1.,
            self.att_keep_prob: 1.
        }
        return sess.run(self.prediction, feed_dict=feed_dict)

    def get_accuracy(self, sess, X, y):
        feed_dict = {
            self.X: X,
            self.y: y,
            self.emb_keep_prob: 1.,
            self.rnn_keep_prob: 1.,
            self.att_keep_prob: 1.
        }
        return sess.run(self.accuracy, feed_dict=feed_dict)

    def train(self, sess, epoch, config, generator):
        iterations = int(len(generator) / config.batch_size) 

        if config.lr_schedule == 'standard':
            learning_rate = self.lr_schedule(config, epoch)
        elif config.lr_schedule == 'constant':
            learning_rate = config.learning_rate
        else:
            raise ValueError("lr schedule does not support other than 'standard' or 'constant'.")

        l2_scale = config.l2_scale
        keep_probs = [config.emb_keep_prob,
                      config.rnn_keep_prob,
                      config.att_keep_prob]

        epoch_results = {'cost': [], 'acc': []}
        for step in range(1, iterations+1):
            X, y = generator.get_batch()
            c, _ = self.optimize(sess, X, y, learning_rate, l2_scale, keep_probs)
            acc = self.get_accuracy(sess, X, y)
            epoch_results['cost'].append(c)
            epoch_results['acc'].append(acc)

            if (step % config.summary_step == 0) and (step != 0):
                avg_cost = np.mean(epoch_results['cost'])
                avg_acc = np.mean(epoch_results['acc'])
                print("epoch:{}   step:{}   cost:{:.4f}   acc:{:.4f}".format(
                    epoch, step, avg_cost, avg_acc
                ))

        last_avg_cost = np.mean(epoch_results['cost'])
        last_avg_acc = np.mean(epoch_results['acc'])
        print()
        return [last_avg_cost, last_avg_acc]

    def validate(self, sess, epoch, config, generator):
        X, y = generator.X, generator.y_onehot
        feed_dict = {
            self.X: X,
            self.y: y,
            self.l2_scale: config.l2_scale,
            self.emb_keep_prob: 1.,
            self.rnn_keep_prob: 1.,
            self.att_keep_prob: 1.
        }
        y_pred = self.predict(sess, X)
        y_true = np.argmax(y, axis=1)
        val_f1 = f1_score(y_true, y_pred, average='macro')
        val_cost, val_acc = sess.run([self.cost, self.accuracy], feed_dict=feed_dict)
        print("epoch:{}   val_cost:{:.4f}   val_acc:{:.4f}   val_f1:{:.4f}".format(
            epoch, val_cost, val_acc, val_f1
        ))
        print()
        return [val_cost, val_acc, val_f1]

    def save(self, sess, path, name):
        savepath = os.path.join(path, name + '.ckpt')
        _ = self.saver.save(sess, savepath)

    @staticmethod
    def lr_schedule(config, cur_epoch):
        if cur_epoch <= (config.epochs * .5):
            return config.learning_rate
        elif ((config.epochs * .5) < cur_epoch) and (cur_epoch <= (config.epochs * .75)):
            return config.learning_rate * .1
        else:
            return config.learning_rate * .01

    @staticmethod
    def get_seq_len(seq):
        relevant = tf.sign(tf.abs(seq))
        length = tf.reduce_sum(relevant, reduction_indices=1)
        length = tf.cast(length, tf.int32)
        return length
