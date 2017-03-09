import os
import model
import numpy as np
import tensorflow as tf
import random
import time
from gen_dataloader import Gen_Data_loader
from dis_dataloader import Dis_dataloader
from text_classifier import TextCNN
from rollout import ROLLOUT
from target_lstm import TARGET_LSTM
import io_utils
import cPickle
from rdkit import Chem, rdBase

# Disables logs for Smiles conversion
rdBase.DisableLog('rdApp.error')

#########################################################################################
#  Generator  Hyper-parameters
#########################################################################################
EMB_DIM = 32
HIDDEN_DIM = 32
START_TOKEN = 0

PRE_EPOCH_NUM =  1#240
TRAIN_ITER = 1  # generator
SEED = 88
BATCH_SIZE = 64
##########################################################################################

TOTAL_BATCH = 800

#########################################################################################
#  Discriminator  Hyper-parameters
#########################################################################################
dis_embedding_dim = 64
dis_filter_sizes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]
dis_num_filters = [100, 200, 200, 200, 200, 100, 100, 100, 100, 100, 160, 160]
dis_dropout_keep_prob = 0.75
dis_l2_reg_lambda = 0.2

# Training parameters
dis_batch_size = 64
dis_num_epochs = 3
dis_alter_epoch = 3


##############################################################################################

DATA_DIR = "../../data"

smiles = io_utils.read_smiles_csv(os.path.join(DATA_DIR, 'subset_11.csv'))

def build_vocab(smiles, pad_char = '_', start_char = '^'):
    i = 1
    char_dict, ord_dict = {start_char: 0}, {0: start_char}
    for smile in smiles:
        for c in smile:
            if c not in char_dict:
                char_dict[c] = i
                ord_dict[i] = c
                i += 1
    char_dict[pad_char], ord_dict[i] = i, pad_char
    return char_dict, ord_dict


char_dict, ord_dict = build_vocab(smiles)

def pad(smile, n, pad_char = '_'):
    if n < len(smile):
        return smile
    return smile + pad_char * (n - len(smile))

def unpad(smile, pad_char = '_'): return smile.rstrip(pad_char)

def encode_smile(smile, max_len): return [char_dict[c] for c in pad(smile, max_len)]
def decode_smile(ords): return unpad(''.join([ord_dict[o] for o in ords]))

NUM_EMB = len(char_dict)

def verify_sequence(seq):
    decoded = decode_smile(seq)
    return decoded != '' and Chem.MolFromSmiles(decoded) is not None

SEQ_LENGTH = max(map(len, smiles))

smiles = [encode_smile(smile, SEQ_LENGTH) for smile in smiles]
smiles = filter(verify_sequence, smiles)

positive_samples = smiles

generated_num = len(positive_samples)

print('Starting SeqGAN with {} positive samples'.format(generated_num))
print('Size of alphabet is {}'.format(NUM_EMB))
print('Sequence length is {}'.format(SEQ_LENGTH))


def print_smile_samples(samples, n):
    for i in range(min(n, len(samples))):
        print decode_smile(samples[i])

##############################################################################################

class Generator(model.LSTM):
    def g_optimizer(self, *args, **kwargs):
        return tf.train.AdamOptimizer(0.002)  # ignore learning rate


def generate_samples(sess, trainable_model, batch_size, generated_num):
    #  Generated Samples
    generated_samples = []
    start = time.time()
    for _ in range(int(generated_num / batch_size)):
        generated_samples.extend(trainable_model.generate(sess))
    end = time.time()
    #print 'Sample generation time:', (end - start)
    return generated_samples


def target_loss(sess, target_lstm, data_loader):
    supervised_g_losses = []
    data_loader.reset_pointer()

    for it in xrange(data_loader.num_batch):
        batch = data_loader.next_batch()
        g_loss = sess.run(target_lstm.pretrain_loss, {target_lstm.x: batch})
        supervised_g_losses.append(g_loss)

    return np.mean(supervised_g_losses)


def significance_test(sess, target_lstm, data_loader, output_file):
    loss = []
    data_loader.reset_pointer()

    for it in xrange(data_loader.num_batch):
        batch = data_loader.next_batch()
        g_loss = sess.run(target_lstm.out_loss, {target_lstm.x: batch})
        loss.extend(list(g_loss))
    with open(output_file, 'w')as fout:
        for item in loss:
            buffer = str(item) + '\n'
            fout.write(buffer)


def pre_train_epoch(sess, trainable_model, data_loader):
    supervised_g_losses = []
    data_loader.reset_pointer()

    for it in xrange(data_loader.num_batch):
        batch = data_loader.next_batch()
        _, g_loss, g_pred = trainable_model.pretrain_step(sess, batch)
        supervised_g_losses.append(g_loss)

    return np.mean(supervised_g_losses)


def main():
    random.seed(SEED)
    np.random.seed(SEED)

    #assert START_TOKEN == 0

    gen_data_loader = Gen_Data_loader(BATCH_SIZE)
    likelihood_data_loader = Gen_Data_loader(BATCH_SIZE)
    vocab_size = NUM_EMB
    dis_data_loader = Dis_dataloader()

    best_score = 1000
    generator = Generator(vocab_size, BATCH_SIZE, EMB_DIM, HIDDEN_DIM, SEQ_LENGTH, START_TOKEN)
    target_lstm = TARGET_LSTM(vocab_size, BATCH_SIZE, EMB_DIM, HIDDEN_DIM, SEQ_LENGTH, 0)

    with tf.variable_scope('discriminator'):
        cnn = TextCNN(
            sequence_length=SEQ_LENGTH,
            num_classes=2,
            vocab_size=vocab_size,
            embedding_size=dis_embedding_dim,
            filter_sizes=dis_filter_sizes,
            num_filters=dis_num_filters,
            l2_reg_lambda=dis_l2_reg_lambda)

    cnn_params = [param for param in tf.trainable_variables() if 'discriminator' in param.name]
    # Define Discriminator Training procedure
    dis_global_step = tf.Variable(0, name="global_step", trainable=False)
    dis_optimizer = tf.train.AdamOptimizer(1e-4)
    dis_grads_and_vars = dis_optimizer.compute_gradients(cnn.loss, cnn_params, aggregation_method=2)
    dis_train_op = dis_optimizer.apply_gradients(dis_grads_and_vars, global_step=dis_global_step)

    config = tf.ConfigProto()
    # config.gpu_options.per_process_gpu_memory_fraction = 0.5
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())

    #samples = generate_samples(sess, target_lstm, BATCH_SIZE, generated_num)
    gen_data_loader.create_batches(positive_samples)

    log = open('log/experiment-log.txt', 'w')
    #  pre-train generator
    print 'Start pre-training...'
    log.write('pre-training...\n')
    for epoch in xrange(PRE_EPOCH_NUM):
        print 'pre-train epoch:', epoch
        loss = pre_train_epoch(sess, generator, gen_data_loader)
        if epoch % 5 == 0:
            samples = generate_samples(sess, generator, BATCH_SIZE, generated_num)
            likelihood_data_loader.create_batches(samples)
            test_loss = target_loss(sess, target_lstm, likelihood_data_loader)
            print 'pre-train epoch ', epoch, 'test_loss ', test_loss, 'train_loss ', loss
            buffer = str(epoch) + ' ' + str(test_loss) + '\n'
            log.write(buffer)

            print_smile_samples(samples, 10)

            verified_samples = [s for s in samples if verify_sequence(s)]
            print 'Verified samples. Pct: {}'.format(float(len(verified_samples)) / len(samples))
            print_smile_samples(verified_samples, 10)

    samples = generate_samples(sess, generator, BATCH_SIZE, generated_num)
    likelihood_data_loader.create_batches(samples)
    test_loss = target_loss(sess, target_lstm, likelihood_data_loader)
    buffer = 'After pre-training:' + ' ' + str(test_loss) + '\n'
    log.write(buffer)

    samples = generate_samples(sess, generator, BATCH_SIZE, generated_num)
    likelihood_data_loader.create_batches(samples)
    significance_test(sess, target_lstm, likelihood_data_loader, 'significance/supervise.txt')


    def train_discriminator():
        negative_samples = generate_samples(sess, generator, BATCH_SIZE, generated_num)

        #  train discriminator
        dis_x_train, dis_y_train = dis_data_loader.load_train_data(positive_samples, negative_samples)
        dis_batches = dis_data_loader.batch_iter(
            zip(dis_x_train, dis_y_train), dis_batch_size, dis_num_epochs
        )

        for batch in dis_batches:
            x_batch, y_batch = zip(*batch)
            feed = {
                cnn.input_x: x_batch,
                cnn.input_y: y_batch,
                cnn.dropout_keep_prob: dis_dropout_keep_prob
            }
            _, step, loss = sess.run([dis_train_op, dis_global_step, cnn.loss], feed)
        print 'Discriminator loss: {}'.format(loss)


    print 'Start training discriminator...'
    for i in range(dis_alter_epoch):
        print 'epoch {}'.format(i)
        train_discriminator()

    rollout = ROLLOUT(generator, 0.8)

    print '#########################################################################'
    print 'Start Reinforcement Training Generator...'
    log.write('Reinforcement Training...\n')

    for total_batch in range(TOTAL_BATCH):
        print '#########################################################################'
    	print 'Training generator with Reinforcement Learning. Epoch {}'.format(total_batch)
        for it in range(TRAIN_ITER):
            samples = generator.generate(sess)
            rewards = rollout.get_reward(sess, samples, 16, cnn)
            g_loss = generator.generator_step(sess, samples, rewards)

        if total_batch % 1 == 0 or total_batch == TOTAL_BATCH - 1:
            samples = generate_samples(sess, generator, BATCH_SIZE, generated_num)
            likelihood_data_loader.create_batches(samples)
            test_loss = target_loss(sess, target_lstm, likelihood_data_loader)
            buffer = str(total_batch) + ' ' + str(test_loss) + '\n'
            print 'total_batch: ', total_batch, 'test_loss: ', test_loss
            print 'Samples'
            print_smile_samples(samples, 10)

            verified_samples = [s for s in samples if verify_sequence(s)]
            print 'Verified samples. Pct: {}'.format(float(len(verified_samples)) / len(samples))
            print_smile_samples(verified_samples, 10)

            log.write(buffer)

            if test_loss < best_score:
                best_score = test_loss
                print 'best score: ', test_loss
                significance_test(sess, target_lstm, likelihood_data_loader, 'significance/seqgan.txt')

        rollout.update_params()

        # generate for discriminator
        print 'Start training discriminator'
        for i in range(5):
            print 'epoch {}'.format(i)
            train_discriminator()

    log.close()


if __name__ == '__main__':
    main()
