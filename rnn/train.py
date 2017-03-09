from __future__ import print_function
import numpy as np
import tensorflow as tf
from rdkit import Chem, rdBase

# Disables logs for Smiles conversion
rdBase.DisableLog('rdApp.error')

import argparse
import time
import os
import io_utils
from six.moves import cPickle

from utils import TextLoader, MolLoader
from model import Model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data/tinyshakespeare',
                       help='data directory containing input.txt')
    parser.add_argument('--save_dir', type=str, default='save',
                       help='directory to store checkpointed models')
    parser.add_argument('--rnn_size', type=int, default=128,
                       help='size of RNN hidden state')
    parser.add_argument('--num_layers', type=int, default=2,
                       help='number of layers in the RNN')
    parser.add_argument('--model', type=str, default='lstm',
                       help='rnn, gru, or lstm')
    parser.add_argument('--batch_size', type=int, default=50,
                       help='minibatch size')
    parser.add_argument('--num_epochs', type=int, default=50,
                       help='number of epochs')
    parser.add_argument('--save_every', type=int, default=1000,
                       help='save frequency')
    parser.add_argument('--grad_clip', type=float, default=5.,
                       help='clip gradients at this value')
    parser.add_argument('--learning_rate', type=float, default=0.002,
                       help='learning rate')
    parser.add_argument('--decay_rate', type=float, default=0.97,
                       help='decay rate for rmsprop')
    parser.add_argument('--init_from', type=str, default=None,
                       help="""continue training from saved model at this path. Path must contain files saved by previous training process:
                            'config.pkl'        : configuration;
                            'chars_vocab.pkl'   : vocabulary definitions;
                            'checkpoint'        : paths to model file(s) (created by tf).
                                                  Note: this file contains absolute paths, be careful when moving files around;
                            'model.ckpt-*'      : file(s) with model definition (created by tf)
                        """)
    args = parser.parse_args()
    train(args)

DATA_DIR = '../data/'

smiles = io_utils.read_smiles_csv(os.path.join(DATA_DIR, 'subset_11.csv'))

def build_vocab(smiles, pad_char = '_'):
    i = 0
    char_dict, ord_dict = {}, {}
    for smile in smiles:
        for c in smile:
            if c not in char_dict:
                char_dict[c] = i
                ord_dict[i] = c
                i += 1
    char_dict[pad_char], ord_dict[i] = i, pad_char
    return char_dict, ord_dict

def verify_sequence(smile):
    return smile != '' and Chem.MolFromSmiles(smile) is not None

def pad(smile, n, pad_char = '_'):
    if n < len(smile):
        return smile
    return smile + pad_char * (n - len(smile))

def unpad(smile, pad_char = '_'): return smile.rstrip(pad_char)


def pct(a, b):
    if len(b) == 0:
        return 0
    return float(len(a)) / len(b)

def objective(samples, verify_fn, in_train_fn, max_len):
    verified = filter(verify_fn, samples)
    in_train = filter(in_train_fn, verified)
    count_unique = [len(set(sample)) for sample in samples]
    return pct(verified, samples) * (1 - pct(in_train, verified)) * (np.mean(count_unique) / float(max_len))




def train(args):
    args.seq_length = max(map(len, smiles))
    SEQ_LENGTH = args.seq_length
    padded = [pad(smile, args.seq_length) for smile in smiles]

    data_loader = MolLoader(padded, args.batch_size, args.seq_length)
    char_dict, ord_dict = data_loader.vocab, data_loader.ord_dict
    args.vocab_size = data_loader.vocab_size
    print('{} {}'.format(args.seq_length, args.vocab_size))


    def encode_smile(smile, max_len): return [char_dict[c] for c in pad(smile, max_len)]
    def decode_smile(ords): return unpad(''.join([ord_dict[o] for o in ords]))

    # check compatibility if training is continued from previously saved model
    if args.init_from is not None:
        # check if all necessary files exist
        assert os.path.isdir(args.init_from)," %s must be a a path" % args.init_from
        assert os.path.isfile(os.path.join(args.init_from,"config.pkl")),"config.pkl file does not exist in path %s"%args.init_from
        assert os.path.isfile(os.path.join(args.init_from,"chars_vocab.pkl")),"chars_vocab.pkl.pkl file does not exist in path %s" % args.init_from
        ckpt = tf.train.get_checkpoint_state(args.init_from)
        assert ckpt,"No checkpoint found"
        assert ckpt.model_checkpoint_path,"No model path found in checkpoint"

        # open old config and check if models are compatible
        with open(os.path.join(args.init_from, 'config.pkl'), 'rb') as f:
            saved_model_args = cPickle.load(f)
        need_be_same=["model","rnn_size","num_layers","seq_length"]
        for checkme in need_be_same:
            assert vars(saved_model_args)[checkme]==vars(args)[checkme],"Command line argument and saved model disagree on '%s' "%checkme

        # open saved vocab/dict and check if vocabs/dicts are compatible
        with open(os.path.join(args.init_from, 'chars_vocab.pkl'), 'rb') as f:
            saved_chars, saved_vocab = cPickle.load(f)
        assert saved_chars==data_loader.chars, "Data and loaded model disagree on character set!"
        assert saved_vocab==data_loader.vocab, "Data and loaded model disagree on dictionary mappings!"

    with open(os.path.join(args.save_dir, 'config.pkl'), 'wb') as f:
        cPickle.dump(args, f)
    with open(os.path.join(args.save_dir, 'chars_vocab.pkl'), 'wb') as f:
        cPickle.dump((data_loader.chars, data_loader.vocab), f)

    with tf.variable_scope('model') as scope:
        model = Model(args)
    with tf.variable_scope('model', reuse=True) as scope:
        sample_model = Model(args, True)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(tf.global_variables())
        # restore model
        if args.init_from is not None:
            saver.restore(sess, ckpt.model_checkpoint_path)
        for e in range(args.num_epochs):
            sess.run(tf.assign(model.lr, args.learning_rate * (args.decay_rate ** e)))
            data_loader.reset_batch_pointer()
            state = sess.run(model.initial_state)

            def make_smiles_samples(n):
                sample = sample_model.sample(sess, data_loader.chars, data_loader.vocab, n, '_', 1)
                return [s for s in sample.split('_') if s != '']
            samples = make_smiles_samples(1000)
            verified_samples = filter(verify_sequence, samples)
            print('Verified Samples')
            for s in verified_samples:
            	print(s)


            print('Pct verified {}'.format(pct(verified_samples, samples)))
            in_train_verified = filter(lambda x: x in smiles, verified_samples)
            if verified_samples == 0:
                verified_samples = 1
            print('Pct of verified in train {}'.format(pct(in_train_verified, verified_samples)))
            print('Objective: {}'.format(objective(samples, verify_sequence, lambda x: x in smiles, SEQ_LENGTH)))

            for b in range(data_loader.num_batches):
                start = time.time()
                x, y = data_loader.next_batch()
                feed = {model.input_data: x, model.targets: y}
                for i, (c, h) in enumerate(model.initial_state):
                    feed[c] = state[i].c
                    feed[h] = state[i].h
                train_loss, state, _ = sess.run([model.cost, model.final_state, model.train_op], feed)
                end = time.time()
                if b % 10 == 0:
                    print("{}/{} (epoch {}), train_loss = {:.3f}, time/batch = {:.3f}" \
                        .format(e * data_loader.num_batches + b,
                                args.num_epochs * data_loader.num_batches,
                                e, train_loss, end - start))

                if (e * data_loader.num_batches + b) % args.save_every == 0\
                    or (e==args.num_epochs-1 and b == data_loader.num_batches-1): # save for the last result
                    checkpoint_path = os.path.join(args.save_dir, 'model.ckpt')
                    saver.save(sess, checkpoint_path, global_step = e * data_loader.num_batches + b)
                    print("model saved to {}".format(checkpoint_path))

if __name__ == '__main__':
    main()
