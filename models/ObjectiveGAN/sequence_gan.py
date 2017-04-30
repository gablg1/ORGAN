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
from rdkit.Chem import Crippen, MolFromSmiles
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect
import pandas as pd
import pickle

# Disables logs for Smiles conversion
rdBase.DisableLog('rdApp.error')

#########################################################################################
#  Generator  Hyper-parameters
#########################################################################################
EMB_DIM = 32
HIDDEN_DIM = 32
START_TOKEN = 0

PRE_EPOCH_NUM =  240
TRAIN_ITER = 1  # generator
SEED = 88
BATCH_SIZE = 64

D_WEIGHT = 0

D = max(int(5 * D_WEIGHT), 1)
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
dis_alter_epoch = 50


##############################################################################################

DATA_DIR = "../../data"

smiles = io_utils.read_smiles_csv(os.path.join(DATA_DIR, 'subset_11.csv'))
#smiles = io_utils.read_smiles_smi(os.path.join(DATA_DIR, '250k.smi'))

def pct(a, b):
    if len(b) == 0:
        return 0
    return float(len(a)) / len(b)


def print_molecules(model_samples, train_smiles):
    samples = [decode_smile(s) for s in model_samples]
    unique_samples = list(set(samples))
    print('Unique samples. Pct: {}'.format(pct(unique_samples, samples)))
    verified_samples = filter(verify_sequence, samples)

    for s in samples[0:10]:
        print(s)
    print('Verified samples. Pct: {}'.format(pct(verified_samples, samples)))
    for s in verified_samples[0:10]:
        print(s)
    print('Objective: {}'.format(objective(samples)))

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
print(ord_dict.keys())

def pad(smile, n, pad_char = '_'):
    if n < len(smile):
        return smile
    return smile + pad_char * (n - len(smile))

def unpad(smile, pad_char = '_'): return smile.rstrip(pad_char)

def encode_smile(smile, max_len): return [char_dict[c] for c in pad(smile, max_len)]
def decode_smile(ords): return unpad(''.join([ord_dict[o] for o in ords]))

NUM_EMB = len(char_dict)

def verify_sequence(decoded):
    return decoded != '' and Chem.MolFromSmiles(decoded) is not None

#MODEL_PICKLE = 'solu.pkl'
#with open(MODEL_PICKLE, "r") as afile:
#    solu_model = cPickle.load(afile)

def solubility(smile):
    df = pd.DataFrame({'smiles': [smiles]})
    df['morgan'] = addMorgan(df, col='smiles', nBits=512, radius=4)
    X = np.array([np.array(i) for i in df['morgan'].values])
    return solu_model.predict(X)[0]

def logP(smile):
    if verify_sequence(smile):
        return Crippen.MolLogP(Chem.MolFromSmiles(smile))
    return 0.


def make_reward(train_smiles):
    def reward(decoded):
        if verify_sequence(decoded):
            if decoded not in train_smiles:
                return 1 * logP(decoded)
            else:
                return 0.3 * logP(decoded)
        else:
            return 0.
    def batch_reward(samples):
        decoded = [decode_smile(sample) for sample in samples]
        pct_unique = float(len(list(set(decoded)))) / len(decoded)

        def count(x, xs):
            ret = 0
            for y in xs:
                if y == x:
                    ret += 1
            return ret


        return np.array([reward(sample) * (pct_unique / count(sample, decoded)) ** 3 for sample in decoded])
    return batch_reward

def objective(samples):
    return np.mean([logP(smile) for smile in samples if verify_sequence(smile)])


SEQ_LENGTH = max(map(len, smiles))

positive_samples = [encode_smile(smile, SEQ_LENGTH) for smile in smiles if verify_sequence(smile)]
generated_num = len(positive_samples)

print('Starting SeqGAN with {} positive samples'.format(generated_num))
print('Size of alphabet is {}'.format(NUM_EMB))
print('Sequence length is {}'.format(SEQ_LENGTH))


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



def pre_train_epoch(sess, trainable_model, data_loader):
    supervised_g_losses = []
    data_loader.reset_pointer()

    for it in xrange(data_loader.num_batch):
        batch = data_loader.next_batch()
        _, g_loss, g_pred = trainable_model.pretrain_step(sess, batch)
        supervised_g_losses.append(g_loss)

    return np.mean(supervised_g_losses)



# This is a hack. I don't even use LIkelihood data loader tbh
likelihood_data_loader = Gen_Data_loader(BATCH_SIZE)

def pretrain(sess, generator, target_lstm, train_discriminator):
    #samples = generate_samples(sess, target_lstm, BATCH_SIZE, generated_num)
    gen_data_loader = Gen_Data_loader(BATCH_SIZE)
    gen_data_loader.create_batches(positive_samples)

    #  pre-train generator
    print('Start pre-training...')
    for epoch in xrange(PRE_EPOCH_NUM):
        print('pre-train epoch:', epoch)
        loss = pre_train_epoch(sess, generator, gen_data_loader)
        if epoch % 5 == 0:
            samples = generate_samples(sess, generator, BATCH_SIZE, generated_num)
            likelihood_data_loader.create_batches(samples)
            test_loss = target_loss(sess, target_lstm, likelihood_data_loader)
            print('pre-train epoch ', epoch, 'test_loss ', test_loss, 'train_loss ', loss)

            print_molecules(samples, smiles)


    samples = generate_samples(sess, generator, BATCH_SIZE, generated_num)
    likelihood_data_loader.create_batches(samples)
    test_loss = target_loss(sess, target_lstm, likelihood_data_loader)

    samples = generate_samples(sess, generator, BATCH_SIZE, generated_num)
    likelihood_data_loader.create_batches(samples)

    print('Start training discriminator...')
    for i in range(dis_alter_epoch):
        print('epoch {}'.format(i))
        train_discriminator()

def main():
    random.seed(SEED)
    np.random.seed(SEED)

    #assert START_TOKEN == 0

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

    def train_discriminator():
        if D_WEIGHT == 0:
            return

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
            _, step, loss, accuracy = sess.run([dis_train_op, dis_global_step, cnn.loss, cnn.accuracy], feed)
        print('Discriminator loss: {} Accuracy: {}'.format(loss, accuracy))



    # Pretrain is checkpointed and only execcutes if we don't find a checkpoint
    saver = tf.train.Saver()
    pretrain_ckpt_file = 'checkpoints/pretrain_ckpt'
    if os.path.isfile(pretrain_ckpt_file + '.meta'):
        saver.restore(sess, pretrain_ckpt_file)
        print('Pretrain loaded from previous checkpoint {}'.format(pretrain_ckpt_file))
    else:
        sess.run(tf.global_variables_initializer())
        pretrain(sess, generator, target_lstm, train_discriminator)
        path = saver.save(sess, pretrain_ckpt_file)
        print('Pretrain finished and saved at {}'.format(path))

    rollout = ROLLOUT(generator, 0.8)

    print('#########################################################################')
    print('Start Reinforcement Training Generator...')

    for total_batch in range(TOTAL_BATCH):
        if total_batch % 1 == 0 or total_batch == TOTAL_BATCH - 1:
            samples = generate_samples(sess, generator, BATCH_SIZE, generated_num)
            likelihood_data_loader.create_batches(samples)
            test_loss = target_loss(sess, target_lstm, likelihood_data_loader)
            print('total_batch: ', total_batch, 'test_loss: ', test_loss)

            print_molecules(samples, smiles)

            if test_loss < best_score:
                best_score = test_loss
                print('best score: ', test_loss)

        print('#########################################################################')
        print('Training generator with Reinforcement Learning. Epoch {}'.format(total_batch))
        for it in range(TRAIN_ITER):
            samples = generator.generate(sess)
            rewards = rollout.get_reward(sess, samples, 16, cnn, make_reward(smiles), D_WEIGHT)
            print(rewards)
            g_loss = generator.generator_step(sess, samples, rewards)

            print('total_batch: ', total_batch, 'g_loss: ', g_loss)


        rollout.update_params()

        # generate for discriminator
        print('Start training discriminator')
        for i in range(D):
            print('epoch {}'.format(i))
            train_discriminator()



if __name__ == '__main__':
    main()
