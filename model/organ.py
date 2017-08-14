from __future__ import absolute_import, division, print_function
import os
from gpu_utils import pick_gpu_lowest_memory
try:
    gpu_free_number = str(pick_gpu_lowest_memory())
    os.environ['CUDA_VISIBLE_DEVICES'] = '{}'.format(gpu_free_number)
    import tensorflow as tf
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    from keras import backend as K
except Exception:
    import tensorflow as tf
    from keras import backend as K
from nn_metrics import KerasNN
#from gp_metrics import GaussianProcess
from builtins import range
from collections import OrderedDict
from generator import Generator, Rollout
import numpy as np
import tensorflow as tf
import random
import dill as pickle
import mol_methods as mm
from data_loaders import Gen_Dataloader, Dis_Dataloader
from discriminator import Discriminator
from custom_metrics import get_metrics, metrics_loading
from tensorflow import logging
from rdkit import rdBase
import pandas as pd
from tqdm import tqdm
__version__ = '0.4.0'


class ORGANIC(object):
    """Main class, where every interaction between the user
    and the backend is performed.
    """

    def __init__(self, name, params={}, use_gpu=True, verbose=True):
        """Parameter initialization.

        Arguments
        -----------

            - name. String which will be used to identify the
            model in any folders or files created.

            - params. Optional. Dictionary containing the parameters
            that the user whishes to specify.

            - use_gpu. Boolean specifying whether a GPU should be
            used. True by default.

            - verbose. Boolean specifying whether output must be
            produced in-line.

        """

        self.verbose = verbose

        # Set minimum verbosity for RDKit, Keras and TF backends
        os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '3'
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        logging.set_verbosity(logging.INFO)
        rdBase.DisableLog('rdApp.error')

        # Set configuration for GPU
        self.config = tf.ConfigProto()
        self.config.gpu_options.allow_growth = True

        # Set parameters
        self.PREFIX = name

        if 'PRETRAIN_GEN_EPOCHS' in params:
            self.PRETRAIN_GEN_EPOCHS = params['PRETRAIN_GEN_EPOCHS']
        else:
            self.PRETRAIN_GEN_EPOCHS = 240

        if 'PRETRAIN_DIS_EPOCHS' in params:
            self.PRETRAIN_DIS_EPOCHS = params['PRETRAIN_DIS_EPOCHS']
        else:
            self.PRETRAIN_DIS_EPOCHS = 50

        if 'GEN_ITERATIONS' in params:
            self.GEN_ITERATIONS = params['GEN_ITERATIONS']
        else:
            self.GEN_ITERATIONS = 2

        if 'GEN_BATCH_SIZE' in params:
            self.GEN_BATCH_SIZE = params['GEN_BATCH_SIZE']
        else:
            self.GEN_BATCH_SIZE = 64

        if 'SEED' in params:
            self.SEED = params['SEED']
        else:
            self.SEED = None
        random.seed(self.SEED)
        np.random.seed(self.SEED)

        if 'DIS_BATCH_SIZE' in params:
            self.DIS_BATCH_SIZE = params['DIS_BATCH_SIZE']
        else:
            self.DIS_BATCH_SIZE = 64

        if 'DIS_EPOCHS' in params:
            self.DIS_EPOCHS = params['DIS_EPOCHS']
        else:
            self.DIS_EPOCHS = 3

        if 'EPOCH_SAVES' in params:
            self.EPOCH_SAVES = params['EPOCH_SAVES']
        else:
            self.EPOCH_SAVES = 20

        if 'CHK_PATH' in params:
            self.CHK_PATH = params['CHK_PATH']
        else:
            self.CHK_PATH = os.path.join(
                os.getcwd(), 'checkpoints/{}'.format(self.PREFIX))

        if 'GEN_EMB_DIM' in params:
            self.GEN_EMB_DIM = params['GEN_EMB_DIM']
        else:
            self.GEN_EMB_DIM = 32

        if 'GEN_HIDDEN_DIM' in params:
            self.GEN_HIDDEN_DIM = params['GEN_HIDDEN_DIM']
        else:
            self.GEN_HIDDEN_DIM = 32

        if 'START_TOKEN' in params:
            self.START_TOKEN = params['START_TOKEN']
        else:
            self.START_TOKEN = 0

        if 'SAMPLE_NUM' in params:
            self.SAMPLE_NUM = params['SAMPLE_NUM']
        else:
            self.SAMPLE_NUM = 6400

        if 'BIG_SAMPLE_NUM' in params:
            self.BIG_SAMPLE_NUM = params['BIG_SAMPLE_NUM']
        else:
            self.BIG_SAMPLE_NUM = self.SAMPLE_NUM * 5

        if 'LAMBDA' in params:
            self.LAMBDA = params['LAMBDA']
        else:
            self.LAMBDA = 0.5

        # In case this parameter is not specified by the user,
        # it will be determined later, in the training set
        # loading.
        if 'MAX_LENGTH' in params:
            self.MAX_LENGTH = params['MAX_LENGTH']

        if 'DIS_EMB_DIM' in params:
            self.DIS_EMB_DIM = params['DIS_EMB_DIM']
        else:
            self.DIS_EMB_DIM = 64

        if 'DIS_FILTER_SIZES' in params:
            self.DIS_FILTER_SIZES = params['DIS_FILTER_SIZES']
        else:
            self.DIS_FILTER_SIZES = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]

        if 'DIS_NUM_FILTERS' in params:
            self.DIS_NUM_FILTERS = params['DIS_FILTER_SIZES']
        else:
            self.DIS_NUM_FILTERS = [100, 200, 200, 200, 200, 100,
                                    100, 100, 100, 100, 160, 160]

        if 'DIS_DROPOUT' in params:
            self.DIS_DROPOUT = params['DIS_DROPOUT']
        else:
            self.DIS_DROPOUT = 0.75
        if 'DIS_L2REG' in params:
            self.DIS_L2REG = params['DIS_L2REG']
        else:
            self.DIS_L2REG = 0.2

        self.AV_METRICS = get_metrics()
        self.LOADINGS = metrics_loading()

        self.PRETRAINED = False
        self.SESS_LOADED = False
        self.USERDEF_METRIC = False

    def load_training_set(self, file):
        """Specifies a training set for the model. It also finishes
        the model set up, as some of the internal parameters require
        knowledge of the vocabulary.

        Arguments
        -----------

            - file. String pointing to a .smi or .csv file.

        """

        # Load training set
        self.train_samples = mm.load_train_data(file)

        # Process and create vocabulary
        self.char_dict, self.ord_dict = mm.build_vocab(self.train_samples)
        self.NUM_EMB = len(self.char_dict)
        self.PAD_CHAR = self.ord_dict[self.NUM_EMB - 1]
        self.PAD_NUM = self.char_dict[self.PAD_CHAR]
        self.DATA_LENGTH = max(map(len, self.train_samples))

        # If MAX_LENGTH has not been specified by the user, it
        # will be set as 1.5 times the maximum length in the
        # trining set.
        if not hasattr(self, 'MAX_LENGTH'):
            self.MAX_LENGTH = int(len(max(self.train_samples, key=len)) * 1.5)

        # Encode samples
        to_use = [sample for sample in self.train_samples
                  if mm.verified_and_below(sample, self.MAX_LENGTH)]
        self.positive_samples = [mm.encode(sam,
                                           self.MAX_LENGTH,
                                           self.char_dict) for sam in to_use]
        self.POSITIVE_NUM = len(self.positive_samples)

        # Print information
        if self.verbose:

            print('\nPARAMETERS INFORMATION')
            print('============================\n')
            print('Model name               :   {}'.format(self.PREFIX))
            print('Training set size        :   {} points'.format(
                len(self.train_samples)))
            print('Max data length          :   {}'.format(self.MAX_LENGTH))
            print('Avg length to use is     :   {}'.format(
                np.mean([len(s) for s in to_use])))
            print('Num valid data points is :   {}'.format(
                self.POSITIVE_NUM))
            print('Size of alphabet is      :   {}'.format(self.NUM_EMB))
            print('')

            params = ['PRETRAIN_GEN_EPOCHS', 'PRETRAIN_DIS_EPOCHS',
                      'GEN_ITERATIONS', 'GEN_BATCH_SIZE', 'SEED',
                      'DIS_BATCH_SIZE', 'DIS_EPOCHS', 'EPOCH_SAVES',
                      'CHK_PATH', 'GEN_EMB_DIM', 'GEN_HIDDEN_DIM',
                      'START_TOKEN', 'SAMPLE_NUM', 'BIG_SAMPLE_NUM',
                      'LAMBDA', 'MAX_LENGTH', 'DIS_EMB_DIM',
                      'DIS_FILTER_SIZES', 'DIS_NUM_FILTERS',
                      'DIS_DROPOUT', 'DIS_L2REG']

            for param in params:
                string = param + ' ' * (25 - len(param))
                value = getattr(self, param)
                print('{}:   {}'.format(string, value))

        # Set model
        self.gen_loader = Gen_Dataloader(self.GEN_BATCH_SIZE)
        self.dis_loader = Dis_Dataloader()
        self.mle_loader = Gen_Dataloader(self.GEN_BATCH_SIZE)
        self.generator = Generator(self.NUM_EMB, self.GEN_BATCH_SIZE,
                                   self.GEN_EMB_DIM, self.GEN_HIDDEN_DIM,
                                   self.MAX_LENGTH, self.START_TOKEN)

        with tf.variable_scope('discriminator'):
            self.discriminator = Discriminator(
                sequence_length=self.MAX_LENGTH,
                num_classes=2,
                vocab_size=self.NUM_EMB,
                embedding_size=self.DIS_EMB_DIM,
                filter_sizes=self.DIS_FILTER_SIZES,
                num_filters=self.DIS_NUM_FILTERS,
                l2_reg_lambda=self.DIS_L2REG)
        self.dis_params = [param for param in tf.trainable_variables()
                           if 'discriminator' in param.name]
        self.dis_global_step = tf.Variable(
            0, name="global_step", trainable=False)
        self.dis_optimizer = tf.train.AdamOptimizer(1e-4)
        self.dis_grads_and_vars = self.dis_optimizer.compute_gradients(
            self.discriminator.loss, self.dis_params, aggregation_method=2)
        self.dis_train_op = self.dis_optimizer.apply_gradients(
            self.dis_grads_and_vars, global_step=self.dis_global_step)

        self.sess = tf.Session(config=self.config)
        self.folder = 'checkpoints/{}'.format(self.PREFIX)

    def define_metric(self, name, metric, load_metric=lambda *args: None,
                      pre_batch=False, pre_metric=lambda *args: None):
        """Sets up a new metric and generates a .pkl file in
        the data/ directory.

        Arguments
        -----------

            - name. String used to identify the metric.

            - metric. Function taking as argument a SMILES
            string and returning a float value.

            - load_metric. Optional. Preprocessing needed
            at the beginning of the code.

            - pre_batch. Optional. Boolean specifying whether
            there is any preprocessing when the metric is applied
            to a batch of smiles. False by default.

            - pre_metric. Optional. Preprocessing operations
            for the metric. Will be ignored if pre_batch is False.

        Notes
        -----------

            - For combinations of already existing metrics, check
            the define_metric_as_combination method.

            - For metrics based in neural networks or gaussian
            processes, please check our more specific functions
            define_nn_metric and define_gp_metric.

            - Check the mol_methods module for useful processing
            options, and the custom_metrics module for examples
            of how metrics are defined in ORGANIC.

        """

        if pre_batch:
            def batch_metric(smiles, train_smiles=None):
                psmiles = pre_metric()
                vals = [mm.apply_to_valid(s, metric) for s in psmiles]
                return vals
        else:
            def batch_metric(smiles, train_smiles=None):
                vals = [mm.apply_to_valid(s, metric) for s in smiles]
                return vals

        self.AV_METRICS[name] = batch_metric
        self.LOADINGS[name] = load_metric

        if self.verbose:
            print('Defined metric {}'.format(name))

        metric = [batch_metric, load_metric]
        with open('../data/{}.pkl'.format(name), 'wb') as f:
            pickle.dump(metric, f)

    def define_metric_as_combination(self, name, metrics, ponderations):
        """Sets up a metric made from a combination of
        previously existing metrics. Also generates a
        metric .pkl file in the data/ directory.

        Arguments
        -----------

            - name. String used to identify the metric.

            - metrics. List containing the name identifiers
            of every metric in the list

            - ponderations. List of ponderation coefficients
            for every metric in the previous list.

        """

        funs = [self.AV_METRICS[metric] for metric in metrics]
        funs_load = [self.LOADINGS[metric] for metric in metrics]

        def metric(smiles, train_smiles=None, **kwargs):
            vals = np.zeros(len(smiles))
            for fun, c in zip(funs, ponderations):
                vals += c * np.asarray(fun(smiles))
            return vals

        def load_metric():
            return [fun() for fun in funs_load if fun() is not None]

        self.AV_METRICS[name] = metric
        self.LOADINGS[name] = load_metric

        if self.verbose:
            print('Defined metric {}'.format(name))

        metric = [nmetric, load_metric]
        with open('../data/{}.pkl'.format(name), 'wb') as f:
            pickle.dump(metric, f)


    def define_metric_as_remap(self, name, metric, remapping):
        """Sets up a metric made from a remapping of a
        previously existing metric. Also generates a .pkl
        metric file in the data/ directory.

        Arguments
        -----------

            - name. String used to identify the metric.

            - metric. String identifying the previous metric.

            - remapping. Remap function.

        Note 1
        -----------

            Use of the mathematical remappings provided in the
            mol_methods module is highly recommended.

        """

        pmetric = self.AV_METRICS[metric]

        def nmetric(smiles, train_smiles=None, **kwargs):
            vals = pmetric(smiles, train_smiles, **kwargs)
            return remapping(vals)

        self.AV_METRICS[name] = nmetric
        self.LOADINGS[name] = self.LOADINGS[metric]

        if self.verbose:
            print('Defined metric {}'.format(name))

        metric = [nmetric, self.LOADINGS[metric]]
        with open('../data/{}.pkl'.format(name), 'wb') as f:
            pickle.dump(metric, f)

    def train_nn_as_metric(self, name, train_x, train_y, nepochs=1000):
        """Sets up a metric with a neural network trained on
        a dataset.

        Arguments.
        -----------

            - name. String used to identify the metric.

            - train_x. List of SMILES identificators.

            - train_y. List of property values.

            - nepochs. Number of epochs for training.

        Note.
        -----------

            A name.h5 file is generated in the data/nns directory,
            and this metric can be loaded in the future using the
            load_prev_user_metric() method through the name.pkl
            file generated in the data/ dir.

                load_prev_user_metric('name.pkl')

        """

        cnn = KerasNN(name)
        cnn.train(train_x, train_y, 500, nepochs,
                  earlystopping=True, min_delta=0.001)
        K.clear_session()

        def batch_NN(smiles, train_smiles=None, nn=None):
            """
            User-trained neural network.
            """
            if nn == None:
                raise ValueError('The user-trained NN metric was not properly loaded.')
            fsmiles = []
            zeroindex = []
            for k, sm in enumerate(smiles):
                if mm.verify_sequence(sm):
                    fsmiles.append(sm)
                else:
                    fsmiles.append('c1ccccc1')
                    zeroindex.append(k)
            vals = np.asarray(nn.predict(fsmiles))
            for k in zeroindex:
                vals[k] = 0.0
            vals = np.squeeze(np.stack(vals, axis=1))
            return vals

        def load_NN():
            """
            Loads the Keras NN model for a user-trained metric.
            """
            nn = KerasNN(name)
            nn.load('../data/nns/{}.h5'.format(name))
            return ('nn', nn)

        self.AV_METRICS[name] = batch_NN
        self.LOADINGS[name] = load_NN

        if self.verbose:
            print('Defined metric {}'.format(name))

        metric = [batch_NN, load_NN]
        with open('../data/{}.pkl'.format(name), 'wb') as f:
            pickle.dump(metric, f)

#    def train_gp_as_metric(self, name, train_x, train_y):
#        """Sets up a metric with a gaussian process trained on
#        a dataset.
#
#        Arguments.
#        -----------
#
#            - name. String used to identify the metric.
#
#            - train_x. List of SMILES identificators.
#
#            - train_y. List of property values.
#
#            - nepochs. Number of epochs for training.
#
#        Note.
#        -----------
#
#            A name.json file is generated in the data/gps directory,
#            and this metric can be loaded in the future using the
#            load_prev_user_metric() method through the name.pkl
#            file generated in the data/ dir.
#
#                load_prev_user_metric('name.pkl')
#
#        """
#
#        gp = GaussianProcess(name)
#        gp.train(train_x, train_y)
#
#        def batch_GP(smiles, train_smiles=None, cnn=None):
#            """
#            User-trained gaussian process.
#            """
#            if gp == None:
#                raise ValueError('The user-trained GP metric was not properly loaded.')
#            fsmiles = []
#            zeroindex = []
#            for k, sm in enumerate(smiles):
#                if mm.verify_sequence(sm):
#                    fsmiles.append(sm)
#                else:
#                    fsmiles.append('c1ccccc1')
#                    zeroindex.append(k)
#            vals = np.asarray(gp.predict(fsmiles))
#            for k in zeroindex:
#                vals[k] = 0.0
#            vals = np.squeeze(np.stack(vals, axis=1))
#            return vals
#
#        def load_GP():
#            """
#            Loads the GPmol GP model for a user-trained metric.
#            """
#            gp = GaussianProcess(name)
#            gp.load('../data/gps/{}.json'.format(name))
#            return ('gp', gp)
#
#        self.AV_METRICS[name] = batch_GP
#        self.LOADINGS[name] = load_GP
#
#        if self.verbose:
#            print('Defined metric {}'.format(name))
#
#        metric = [batch_GP, load_GP]
#        with open('../data/{}.pkl'.format(name), 'wb') as f:
#            pickle.dump(metric, f)

    def load_prev_user_metric(self, name, file=None):
        """Loads a metric that the user has previously designed.

        Arguments.
        -----------

            - name. String used to identify the metric.

            - file. String pointing to the .pkl file. Will use
            data/name.pkl by default.

        """
        if file is None:
            file = '../data/{}.pkl'.format(name)
        pkl = open(file, 'rb')
        data = pickle.load(pkl)
        self.AV_METRICS[name] = data[0]
        self.LOADINGS[name] = data[1]
        if self.verbose:
            print('Loaded metric {}'.format(name))

    def set_training_program(self, metrics=None, steps=None):
        """Sets a program of metrics and epochs
        for training the model and generating molecules.

        Arguments
        -----------

            - metrics. List of metrics. Each element represents
            the metric used with a particular set of epochs. Its
            length must coincide with the steps list.

            - steps. List of epoch sets. Each element represents
            the number of epochs for which a given metric will
            be used. Its length must coincide with the steps list.

        Note
        -----------

            The program will crash if both lists have different
            lengths.

        Example
        -----------

            The following examples trains the model for, sequentially,
            20 epochs of PCE, 100 epochs of bandgap and another 20
            epochs of PCE.

                model = ORGANIC('model')
                model.load_training_set('sample.smi')
                model.set_training_program(['pce', 'bandgap', 'pce'],
                                           [20, 100, 20])

        """

        # Raise error if the lengths do not match
        if len(metrics) != len(steps):
            return ValueError('Unmatching lengths in training program.')

        # Set important parameters
        self.TOTAL_BATCH = np.sum(np.asarray(steps))
        self.METRICS = metrics

        # Build the 'educative program'
        self.EDUCATION = {}
        i = 0
        for j, stage in enumerate(steps):
            for _ in range(stage):
                self.EDUCATION[i] = metrics[j]
                i += 1

    def load_metrics(self):
        """Loads the metrics."""

        # Get the list of used metrics
        met = list(set(self.METRICS))

        # Execute the metrics loading
        self.kwargs = {}
        for m in met:
            load_fun = self.LOADINGS[m]
            args = load_fun()
            if args is not None:
                if isinstance(args, tuple):
                    self.kwargs[m] = {args[0]: args[1]}
                elif isinstance(args, list):
                    fun_args = {}
                    for arg in args:
                        fun_args[arg[0]] = arg[1]
                    self.kwargs[m] = fun_args
            else:
                self.kwargs[m] = None

    def load_prev_pretraining(self, ckpt=None):
        """
        Loads a previous pretraining.

        Arguments
        -----------

            - ckpt. String pointing to the ckpt file. By default,
            'checkpoints/name_pretrain/pretrain_ckpt' is assumed.

        Note
        -----------

            The models are stored by the Tensorflow API backend. This
            will generate various files, like in the following ls:

                checkpoint
                pretrain_ckpt.data-00000-of-00001
                pretrain_ckpt.index
                pretrain_ckpt.meta

            In this case, ckpt = 'pretrain_ckpt'.

        Note 2
        -----------

            Due to its structure, ORGANIC is very dependent on its
            hyperparameters (for example, MAX_LENGTH defines the
            embedding). Most of the errors with this function are
            related to parameter mismatching.

        """

        # Generate TF saver
        saver = tf.train.Saver()

        # Set default checkpoint
        if ckpt is None:
            ckpt_dir = 'checkpoints/{}_pretrain'.format(self.PREFIX)
            if not os.path.exists(ckpt_dir):
                print('No pretraining data was found')
                return
            ckpt = os.path.join(ckpt_dir, 'pretrain_ckpt')

        # Load from checkpoint
        if os.path.isfile(ckpt + '.meta'):
            saver.restore(self.sess, ckpt)
            print('Pretrain loaded from previous checkpoint {}'.format(ckpt))
            self.PRETRAINED = True
        else:
            print('\t* No pre-training data found as {:s}.'.format(ckpt))

    def load_prev_training(self, ckpt=None):
        """
        Loads a previous trained model.

        Arguments
        -----------

            - ckpt. String pointing to the ckpt file. By default,
            'checkpoints/name/pretrain_ckpt' is assumed.

        Note 1
        -----------

            The models are stored by the Tensorflow API backend. This
            will generate various files. An example ls:

                checkpoint
                validity_model_0.ckpt.data-00000-of-00001
                validity_model_0.ckpt.index
                validity_model_0.ckpt.meta
                validity_model_100.ckpt.data-00000-of-00001
                validity_model_100.ckpt.index
                validity_model_100.ckpt.meta
                validity_model_120.ckpt.data-00000-of-00001
                validity_model_120.ckpt.index
                validity_model_120.ckpt.meta
                validity_model_140.ckpt.data-00000-of-00001
                validity_model_140.ckpt.index
                validity_model_140.ckpt.meta

                    ...

                validity_model_final.ckpt.data-00000-of-00001
                validity_model_final.ckpt.index
                validity_model_final.ckpt.meta

            Possible ckpt values are 'validity_model_0', 'validity_model_140'
            or 'validity_model_final'.

        Note 2
        -----------

            Due to its structure, ORGANIC is very dependent on its
            hyperparameters (for example, MAX_LENGTH defines the
            embedding). Most of the errors with this function are
            related to parameter mismatching.

        """

        # If there is no Rollout, add it
        if not hasattr(self, 'rollout'):
            self.rollout = Rollout(self.generator, 0.8, self.PAD_NUM)

        # Generate TF Saver
        saver = tf.train.Saver()

        # Set default checkpoint
        if ckpt is None:
            ckpt_dir = 'checkpoints/{}'.format(self.PREFIX)
            if not os.path.exists(ckpt_dir):
                print('No pretraining data was found')
                return
            ckpt = os.path.join(ckpt_dir, 'pretrain_ckpt')

        if os.path.isfile(ckpt + '.meta'):
            saver.restore(self.sess, ckpt)
            print('Training loaded from previous checkpoint {}'.format(ckpt))
            self.SESS_LOADED = True
        else:
            print('\t* No training checkpoint found as {:s}.'.format(ckpt))

    def pretrain(self):
        """Pretrains generator and discriminator."""

        self.gen_loader.create_batches(self.positive_samples)
        # results = OrderedDict({'exp_name': self.PREFIX})

        if self.verbose:
            print('\nPRETRAINING')
            print('============================\n')
            print('GENERATOR PRETRAINING')

        for epoch in tqdm(range(self.PRETRAIN_GEN_EPOCHS)):

            supervised_g_losses = []
            self.gen_loader.reset_pointer()

            for it in range(self.gen_loader.num_batch):
                batch = self.gen_loader.next_batch()
                _, g_loss, g_pred = self.generator.pretrain_step(self.sess,
                                                                 batch)
                supervised_g_losses.append(g_loss)
            loss = np.mean(supervised_g_losses)

            if epoch % 10 == 0:

                print('\t train_loss {}'.format(loss))

        samples = self.generate_samples(self.SAMPLE_NUM)
        self.mle_loader.create_batches(samples)

        if self.LAMBDA != 0:

            if self.verbose:
                print('\nDISCRIMINATOR PRETRAINING')

            for i in tqdm(range(self.PRETRAIN_DIS_EPOCHS)):

                negative_samples = self.generate_samples(self.POSITIVE_NUM)
                dis_x_train, dis_y_train = self.dis_loader.load_train_data(
                    self.positive_samples, negative_samples)
                dis_batches = self.dis_loader.batch_iter(
                    zip(dis_x_train, dis_y_train), self.DIS_BATCH_SIZE,
                    self.PRETRAIN_DIS_EPOCHS)

                for batch in dis_batches:
                    x_batch, y_batch = zip(*batch)
                    feed = {
                        self.discriminator.input_x: x_batch,
                        self.discriminator.input_y: y_batch,
                        self.discriminator.dropout_keep_prob: self.DIS_DROPOUT
                    }
                    _, step, loss, accuracy = self.sess.run(
                        [self.dis_train_op, self.dis_global_step,
                         self.discriminator.loss, self.discriminator.accuracy],
                        feed)

        self.PRETRAINED = True

    def generate_samples(self, num):
        """Generates molecules.

        Arguments
        -----------

            - num. Integer representing the number of molecules

        """

        generated_samples = []

        for _ in range(int(num / self.GEN_BATCH_SIZE)):
            generated_samples.extend(self.generator.generate(self.sess))

        return generated_samples

    def train(self, ckpt_dir='checkpoints/'):
        """Trains the model. If necessary, also includes pretraining."""

        if not self.PRETRAINED and not self.SESS_LOADED:

            self.sess.run(tf.global_variables_initializer())
            self.pretrain()

            if not os.path.exists(ckpt_dir):
                os.makedirs(ckpt_dir)
            ckpt_file = os.path.join(ckpt_dir,
                                     '{}_pretrain_ckpt'.format(self.PREFIX))
            saver = tf.train.Saver()
            path = saver.save(self.sess, ckpt_file)
            if self.verbose:
                print('Pretrain saved at {}'.format(path))

        if not hasattr(self, 'rollout'):
            self.rollout = Rollout(self.generator, 0.8, self.PAD_NUM)

        if self.verbose:
            print('\nSTARTING TRAINING')
            print('============================\n')

        results_rows = []
        for nbatch in tqdm(range(self.TOTAL_BATCH)):

            results = OrderedDict({'exp_name': self.PREFIX})

            metric = self.EDUCATION[nbatch]

            if metric in self.AV_METRICS.keys():
                reward_func = self.AV_METRICS[metric]
            else:
                raise ValueError('Metric {} not found!'.format(metric))

            if self.kwargs[metric] is not None:

                def batch_reward(samples):
                    decoded = [mm.decode(sample, self.ord_dict)
                               for sample in samples]
                    pct_unique = len(list(set(decoded))) / float(len(decoded))
                    rewards = reward_func(decoded, self.train_samples,
                                          **self.kwargs[metric])
                    weights = np.array([pct_unique /
                                        float(decoded.count(sample))
                                        for sample in decoded])

                    return rewards * weights

            else:

                def batch_reward(samples):
                    decoded = [mm.decode(sample, self.ord_dict)
                               for sample in samples]
                    pct_unique = len(list(set(decoded))) / float(len(decoded))
                    rewards = reward_func(decoded, self.train_samples)
                    weights = np.array([pct_unique /
                                        float(decoded.count(sample))
                                        for sample in decoded])

                    return rewards * weights

            if nbatch % 10 == 0:
                gen_samples = self.generate_samples(self.BIG_SAMPLE_NUM)
            else:
                gen_samples = self.generate_samples(self.SAMPLE_NUM)
            self.gen_loader.create_batches(gen_samples)
            results['Batch'] = nbatch
            print('Batch n. {}'.format(nbatch))
            print('============================\n')

            # results
            mm.compute_results(
                gen_samples, self.train_samples, self.ord_dict, results)

            for it in range(self.GEN_ITERATIONS):
                samples = self.generator.generate(self.sess)
                rewards = self.rollout.get_reward(
                    self.sess, samples, 16, self.discriminator,
                    batch_reward, self.LAMBDA)
                nll = self.generator.generator_step(
                    self.sess, samples, rewards)

                print('Rewards')
                print('~~~~~~~~~~~~~~~~~~~~~~~~\n')
                np.set_printoptions(precision=3, suppress=True)
                mean_r, std_r = np.mean(rewards), np.std(rewards)
                min_r, max_r = np.min(rewards), np.max(rewards)
                print('Mean:                {:.3f}'.format(mean_r))
                print('               +/-   {:.3f}'.format(std_r))
                print('Min:                 {:.3f}'.format(min_r))
                print('Max:                 {:.3f}'.format(max_r))
                np.set_printoptions(precision=8, suppress=False)
                results['neg-loglike'] = nll
            self.rollout.update_params()

            # generate for discriminator
            if self.LAMBDA != 0:
                print('\nDISCRIMINATOR TRAINING')
                print('============================\n')
                for i in range(self.DIS_EPOCHS):
                    print('Discriminator epoch {}...'.format(i+1))

                    negative_samples = self.generate_samples(self.POSITIVE_NUM)
                    dis_x_train, dis_y_train = self.dis_loader.load_train_data(
                        self.positive_samples, negative_samples)
                    dis_batches = self.dis_loader.batch_iter(
                        zip(dis_x_train, dis_y_train),
                        self.DIS_BATCH_SIZE, self.DIS_EPOCHS
                    )

                    for batch in dis_batches:
                        x_batch, y_batch = zip(*batch)
                        feed = {
                            self.discriminator.input_x: x_batch,
                            self.discriminator.input_y: y_batch,
                            self.discriminator.dropout_keep_prob:
                                self.DIS_DROPOUT
                        }
                        _, step, d_loss, accuracy = self.sess.run(
                            [self.dis_train_op, self.dis_global_step,
                             self.discriminator.loss,
                             self.discriminator.accuracy],
                            feed)

                    results['D_loss_{}'.format(i)] = d_loss
                    results['Accuracy_{}'.format(i)] = accuracy
                print('\nDiscriminator trained.')
            results_rows.append(results)
            if nbatch % self.EPOCH_SAVES == 0 or \
               nbatch == self.TOTAL_BATCH - 1:

                if results_rows is not None:
                    df = pd.DataFrame(results_rows)
                    df.to_csv('{}_results.csv'.format(self.folder),
                              index=False)
                if nbatch is None:
                    label = 'final'
                else:
                    label = str(nbatch)

                # save models
                model_saver = tf.train.Saver()
                ckpt_dir = os.path.join(self.CHK_PATH, self.folder)
                if not os.path.exists(ckpt_dir):
                    os.makedirs(ckpt_dir)
                ckpt_file = os.path.join(
                    ckpt_dir, '{}_{}.ckpt'.format(self.PREFIX, label))
                path = model_saver.save(self.sess, ckpt_file)
                print('\nModel saved at {}'.format(path))

        print('\n######### FINISHED #########')


if __name__ == '__main__':

    # Setup model
    model = ORGANIC('metrics', params={'PRETRAIN_GEN_EPOCHS': 1, 'PRETRAIN_DIS_EPOCHS': 1})
    model.load_training_set('../data/trainingsets/toy.csv')
    model.set_training_program(['validity'], [1])
    model.load_metrics()
    model.train(ckpt_dir='ckpt')
