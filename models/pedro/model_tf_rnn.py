import tensorflow as tf
from tensorflow.python.ops import tensor_array_ops, control_flow_ops

from tensorflow.contrib import legacy_seq2seq
from tensorflow.contrib import rnn

class LSTM(object):
    def __init__(self, num_emb, batch_size, emb_dim, hidden_dim,
                 sequence_length, start_token,
                 learning_rate=0.01, reward_gamma=0.95):
        self.num_emb = num_emb
        self.batch_size = batch_size
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.sequence_length = sequence_length
        self.start_token = tf.constant([start_token] * self.batch_size, dtype=tf.int32)
        self.learning_rate = tf.Variable(float(learning_rate), trainable=False)
        self.reward_gamma = reward_gamma
        self.temperature = 1.0
        self.grad_clip = 5.0

        self.expected_reward = tf.Variable(tf.zeros([self.sequence_length]))


        with tf.variable_scope('generator') as scope:
            self.g_embeddings = tf.Variable(self.init_matrix([self.num_emb, self.emb_dim]))
            self.g_recurrent_unit = self.create_recurrent_unit()  # maps h_tm1 to h_t for generator
            self.g_output_unit = self.create_output_unit()  # maps h_t to o_t (output token logits)

            # placeholder definition
            self.x = tf.placeholder(tf.int32, shape=[self.batch_size, self.sequence_length])
            # sequence of indices of true data, not including start token

            self.rewards = tf.placeholder(tf.float32, shape=[self.batch_size, self.sequence_length])
            # get from rollout policy and discriminator

            # processed for batch
            with tf.device("/cpu:0"):
                inputs = tf.split(axis=1, num_or_size_splits=self.sequence_length, value=tf.nn.embedding_lookup(self.g_embeddings, self.x))
                self.processed_x = tf.stack(
                    [tf.squeeze(input_, [1]) for input_ in inputs])  # seq_length x batch_size x emb_dim

            cell = rnn.BasicLSTMCell(self.hidden_dim, state_is_tuple=True)
            self.cell = rnn.MultiRNNCell([cell] * 2, state_is_tuple=True)

            self.h0 = tf.zeros([self.batch_size, self.hidden_dim])
            self.h0 = tf.stack([self.h0, self.h0])

            self.h0 = self.cell.zero_state(self.batch_size, tf.float32)

            gen_o = tensor_array_ops.TensorArray(dtype=tf.float32, size=self.sequence_length,
                                                dynamic_size=False, infer_shape=True)
            gen_x = tensor_array_ops.TensorArray(dtype=tf.int32, size=self.sequence_length,
                                                dynamic_size=False, infer_shape=True)

            def _g_recurrence(x_t, h_tm1, gen_o, gen_x):
                h_t = self.g_recurrent_unit(x_t, h_tm1)  # hidden_memory_tuple
                o_t = self.g_output_unit(h_t)  # batch x vocab , logits not prob
                log_prob = tf.log(tf.nn.softmax(o_t))
                next_token = tf.cast(tf.reshape(tf.multinomial(log_prob, 1), [self.batch_size]), tf.int32)
                x_tp1 = tf.nn.embedding_lookup(self.g_embeddings, next_token)  # batch x emb_dim
                gen_o = gen_o.write(i, tf.reduce_sum(tf.multiply(tf.one_hot(next_token, self.num_emb, 1.0, 0.0),
                                                            tf.nn.softmax(o_t)), 1))  # [batch_size] , prob
                gen_x = gen_x.write(i, next_token)  # indices, batch_size
                return x_tp1, h_t, gen_o, gen_x

            # My loop
            initial_state = (tf.zeros([self.batch_size, self.hidden_dim]), self.h0)

            x_t, h_t, gen_o, gen_x = tf.nn.embedding_lookup(self.g_embeddings, self.start_token), initial_state, gen_o, gen_x
            for i in range(self.sequence_length):
                if i > 0: scope.reuse_variables()
            	x_t, h_t, gen_o, gen_x = _g_recurrence(x_t, h_t, gen_o, gen_x)
            self.gen_o, self.gen_x = gen_o, gen_x


            self.gen_x = self.gen_x.stack()  # seq_length x batch_size
            self.gen_x = tf.transpose(self.gen_x, perm=[1, 0])  # batch_size x seq_length

            # supervised pretraining for generator
            g_predictions = tensor_array_ops.TensorArray(
                dtype=tf.float32, size=self.sequence_length,
                dynamic_size=False, infer_shape=True)

            g_logits = tensor_array_ops.TensorArray(
                dtype=tf.float32, size=self.sequence_length,
                dynamic_size=False, infer_shape=True)

            ta_emb_x = tensor_array_ops.TensorArray(
                dtype=tf.float32, size=self.sequence_length)
            ta_emb_x = ta_emb_x.unstack(self.processed_x)

            def _pretrain_recurrence(x_t, h_tm1, g_predictions, g_logits):
                h_t = self.g_recurrent_unit(x_t, h_tm1)
                o_t = self.g_output_unit(h_t)
                g_predictions = g_predictions.write(i, tf.nn.softmax(o_t))  # batch x vocab_size
                g_logits = g_logits.write(i, o_t)  # batch x vocab_size
                x_tp1 = ta_emb_x.read(i)
                return x_tp1, h_t, g_predictions, g_logits

            initial_state = (tf.zeros([self.batch_size, self.hidden_dim]), self.h0)
            x_t, ht = tf.nn.embedding_lookup(self.g_embeddings, self.start_token), initial_state
            for i in range(self.sequence_length):
                if i > 0: scope.reuse_variables()
            	x_t, h_t, g_predictions, g_logits = _pretrain_recurrence(x_t, h_t, g_predictions, g_logits)
            self.g_predictions, self.g_logits = g_predictions, g_logits

            self.g_predictions = tf.transpose(
                self.g_predictions.stack(), perm=[1, 0, 2])  # batch_size x seq_length x vocab_size

            self.g_logits = tf.transpose(
                self.g_logits.stack(), perm=[1, 0, 2])  # batch_size x seq_length x vocab_size

        # pretraining loss
        self.pretrain_loss = -tf.reduce_sum(
            tf.one_hot(tf.to_int32(tf.reshape(self.x, [-1])), self.num_emb, 1.0, 0.0) * tf.log(
                tf.clip_by_value(tf.reshape(self.g_predictions, [-1, self.num_emb]), 1e-20, 1.0)
            )
        ) / (self.sequence_length * self.batch_size)

        # training updates
        pretrain_opt = self.g_optimizer(self.learning_rate)

        tvars = tf.trainable_variables()
        g_params = [var for var in tvars if 'generator' in var.name]

        self.pretrain_grad, _ = tf.clip_by_global_norm(
            tf.gradients(self.pretrain_loss, g_params), self.grad_clip)
        self.pretrain_updates = pretrain_opt.apply_gradients(zip(self.pretrain_grad, g_params))

        #######################################################################################################
        #  Unsupervised Training
        #######################################################################################################
        self.g_loss = -tf.reduce_sum(
            tf.reduce_sum(
                tf.one_hot(tf.to_int32(tf.reshape(self.x, [-1])), self.num_emb, 1.0, 0.0) * tf.log(
                    tf.clip_by_value(tf.reshape(self.g_predictions, [-1, self.num_emb]), 1e-20, 1.0)
                ), 1) * tf.reshape(self.rewards, [-1])
        )

        g_opt = self.g_optimizer(self.learning_rate)

        self.g_grad, _ = tf.clip_by_global_norm(
            tf.gradients(self.g_loss, g_params), self.grad_clip)
        self.g_updates = g_opt.apply_gradients(zip(self.g_grad, g_params))

    def generate(self, session):
        outputs = session.run([self.gen_x])
        return outputs[0]

    def pretrain_step(self, session, x):
        outputs = session.run([self.pretrain_updates, self.pretrain_loss, self.g_predictions],
                              feed_dict={self.x: x})
        return outputs

    def init_matrix(self, shape):
        return tf.random_normal(shape, stddev=0.1)

    def init_vector(self, shape):
        return tf.zeros(shape)

    def create_recurrent_unit(self):
        def unit(x, (prev_output, prev_state)):
            #previous_hidden_state, c_prev = tf.unstack(hidden_memory_tm1)
            cell_output, state = self.cell(x, prev_state)
            return (cell_output, state)
        return unit

    def create_output_unit(self):
        self.Wo = tf.Variable(self.init_matrix([self.hidden_dim, self.num_emb]))
        self.bo = tf.Variable(self.init_matrix([self.num_emb]))

        def unit((output, state)):
            #hidden_state, c_prev = tf.unstack(hidden_memory_tuple)
            # hidden_state : batch x hidden_dim
            logits = tf.matmul(output, self.Wo) + self.bo
            # output = tf.nn.softmax(logits)
            return logits

        return unit

    def _create_recurrent_unit(self):
        # Weights and Bias for input and hidden tensor
        self.Wi = tf.Variable(self.init_matrix([self.emb_dim, self.hidden_dim]))
        self.Ui = tf.Variable(self.init_matrix([self.emb_dim, self.hidden_dim]))
        self.bi = tf.Variable(self.init_matrix([self.hidden_dim]))

        self.Wf = tf.Variable(self.init_matrix([self.emb_dim, self.hidden_dim]))
        self.Uf = tf.Variable(self.init_matrix([self.hidden_dim, self.hidden_dim]))
        self.bf = tf.Variable(self.init_matrix([self.hidden_dim]))

        self.Wog = tf.Variable(self.init_matrix([self.emb_dim, self.hidden_dim]))
        self.Uog = tf.Variable(self.init_matrix([self.hidden_dim, self.hidden_dim]))
        self.bog = tf.Variable(self.init_matrix([self.hidden_dim]))

        self.Wc = tf.Variable(self.init_matrix([self.emb_dim, self.hidden_dim]))
        self.Uc = tf.Variable(self.init_matrix([self.hidden_dim, self.hidden_dim]))
        self.bc = tf.Variable(self.init_matrix([self.hidden_dim]))

        def unit(x, hidden_memory_tm1):
            previous_hidden_state, c_prev = tf.unstack(hidden_memory_tm1)

            # Input Gate
            i = tf.sigmoid(
                tf.matmul(x, self.Wi) +
                tf.matmul(previous_hidden_state, self.Ui) + self.bi
            )

            # Forget Gate
            f = tf.sigmoid(
                tf.matmul(x, self.Wf) +
                tf.matmul(previous_hidden_state, self.Uf) + self.bf
            )

            # Output Gate
            o = tf.sigmoid(
                tf.matmul(x, self.Wog) +
                tf.matmul(previous_hidden_state, self.Uog) + self.bog
            )

            # New Memory Cell
            c_ = tf.nn.tanh(
                tf.matmul(x, self.Wc) +
                tf.matmul(previous_hidden_state, self.Uc) + self.bc
            )

            # Final Memory cell
            c = f * c_prev + i * c_

            # Current Hidden state
            current_hidden_state = o * tf.nn.tanh(c)

            return tf.stack([current_hidden_state, c])

        return unit

    def _create_output_unit(self):
        self.Wo = tf.Variable(self.init_matrix([self.hidden_dim, self.num_emb]))
        self.bo = tf.Variable(self.init_matrix([self.num_emb]))

        def unit(hidden_memory_tuple):
            hidden_state, c_prev = tf.unstack(hidden_memory_tuple)
            # hidden_state : batch x hidden_dim
            logits = tf.matmul(hidden_state, self.Wo) + self.bo
            # output = tf.nn.softmax(logits)
            return logits

        return unit

    def g_optimizer(self, *args, **kwargs):
        return tf.train.GradientDescentOptimizer(*args, **kwargs)
