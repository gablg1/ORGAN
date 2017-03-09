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

PRE_EPOCH_NUM = 240
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
dis_alter_epoch = 50


##############################################################################################

DATA_DIR = "../../data"

smiles = io_utils.read_smiles_csv(os.path.join(DATA_DIR, 'subset_11.csv'))
#smiles = io_utils.read_smiles_smi(os.path.join(DATA_DIR, '250k.smi'))

def pct(a, b):
    if len(b) == 0:
        return 0
    return float(len(a)) / len(b)

def objective(samples, verify_fn, in_train_fn, max_len):
    unique_samples = list(set(samples))
    verified = filter(verify_fn, unique_samples)
    in_train = filter(in_train_fn, verified)
    return pct(verified, samples) * (1 - pct(in_train, verified))
    #count_unique = [len(set(sample)) for sample in samples]
    #return pct(verified, samples) * (1 - pct(in_train, verified)) * (np.mean(count_unique) / float(max_len))

def print_molecules(model_samples, train_smiles):
    samples = [decode_smile(s) for s in model_samples]
    unique_samples = list(set(samples))
    print 'Unique samples. Pct: {}'.format(pct(unique_samples, samples))
    verified_samples = filter(verify_sequence, samples)

    for s in samples[0:10]:
        print s
    print 'Verified samples. Pct: {}'.format(pct(verified_samples, samples))
    for s in verified_samples[0:10]:
        print s
    print 'Objective: {}'.format(objective(samples, verify_sequence,
        lambda x: x in train_smiles, SEQ_LENGTH))

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

def verify_sequence(decoded):
    return decoded != '' and Chem.MolFromSmiles(decoded) is not None

def make_reward(train_smiles):
    def reward(decoded):
        if verify_sequence(decoded):
            if decoded not in train_smiles:
                return 1.
            else:
                return 0.3
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


        return np.array([reward(sample) / count(sample, decoded) for sample in decoded])
    return batch_reward

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
    vocab_size = NUM_EMB
    dis_data_loader = Dis_dataloader()

    best_score = 1000
    generator = Generator(vocab_size, BATCH_SIZE, EMB_DIM, HIDDEN_DIM, SEQ_LENGTH, START_TOKEN)



    config = tf.ConfigProto()
    # config.gpu_options.per_process_gpu_memory_fraction = 0.5
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())

    gen_data_loader.create_batches(positive_samples)

    #  pre-train generator
    print 'Start pre-training...'
    for epoch in xrange(PRE_EPOCH_NUM):
        print 'pre-train epoch:', epoch
        loss = pre_train_epoch(sess, generator, gen_data_loader)
        if epoch % 5 == 0:
            samples = generate_samples(sess, generator, BATCH_SIZE, generated_num)
            print 'pre-train epoch ', epoch, 'train_loss ', loss

            print_molecules(samples, smiles)


    rollout = ROLLOUT(generator, 0.8)

    print '#########################################################################'
    print 'Start Reinforcement Training Generator...'

    for total_batch in range(TOTAL_BATCH):
        print '#########################################################################'
        print 'Training generator with Reinforcement Learning. Epoch {}'.format(total_batch)
        for it in range(TRAIN_ITER):
            samples = generator.generate(sess)
            rewards = rollout.get_reward(sess, samples, 16, make_reward(smiles))
            print rewards
            g_loss = generator.generator_step(sess, samples, rewards)

        if total_batch % 1 == 0 or total_batch == TOTAL_BATCH - 1:
            samples = generate_samples(sess, generator, BATCH_SIZE, generated_num)
            print 'total_batch: ', total_batch

            print_molecules(samples, smiles)


        rollout.update_params()




if __name__ == '__main__':
    main()
