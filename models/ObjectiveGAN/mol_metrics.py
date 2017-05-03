from __future__ import absolute_import, division, print_function
from builtins import range
import os
import numpy as np
import pandas as pd
import csv
from rdkit import rdBase
from rdkit.Chem import PandasTools
from rdkit import DataStructs
from rdkit.Chem import AllChem as Chem
from rdkit.Chem import Crippen, MolFromSmiles
# Disables logs for Smiles conversion
rdBase.DisableLog('rdApp.error')
#====== math utility


def remap(x, x_min, x_max):
    return (x - x_min) / (x_max - x_min)


def pct(a, b):
    if len(b) == 0:
        return 0
    return float(len(a)) / len(b)

#====== encoding/decoding utility


def verified_and_below(smile, max_len):
    return len(smile) < max_len and verify_sequence(smile)


def verify_sequence(smile):
    return smile != '' and Chem.MolFromSmiles(smile) is not None


def build_vocab(smiles, pad_char='_', start_char='^'):
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


def pad(smile, n, pad_char='_'):
    if n < len(smile):
        return smile
    return smile + pad_char * (n - len(smile))


def unpad(smile, pad_char='_'): return smile.rstrip(pad_char)


def onehot_encode(smile, max_len, char_dict): return [
    char_dict[c] for c in pad(smile, max_len)]


def onehot_decode(ords, ord_dict): return unpad(
    ''.join([ord_dict[o] for o in ords]))


def load_train_data(filename):
    ext = filename.split(".")[-1]
    if ext == 'csv':
        return read_smiles_csv(filename)
    if ext == '.smi':
        return read_smi(filename)
    else:
        raise ValueError('data is not smi or csv!')
    return


def read_smiles_csv(filename):
    # Assumes smiles is in column 0
    with open(filename) as file:
        reader = csv.reader(file)
        smiles_idx = next(reader).index("smiles")
        data = [row[smiles_idx] for row in reader]
    return data


def save_smi(name, smiles):
    if not os.path.exists('smiles'):
        os.makedirs('smiles')
    smi_file = os.path.join('smiles', name + ".smi")
    with open(smi_file, 'w') as afile:
        afile.write('\n'.join(smiles))
    return


def read_smi(filename):
    with open(filename) as file:
        smiles = file.readlines()
    smiles = [i.strip() for i in smiles]
    return smiles


#====== results utility
def print_params(p):
    print('Using parameters:')
    for key, value in p.items():
        print('{:20s} - {:12}'.format(key, value))
    print('rest of parameters are set as default\n')
    return


def compute_results(model_samples, train_samples, ord_dict, results={}, verbose=True):
    samples = [onehot_decode(s, ord_dict) for s in model_samples]
    samples_lens = map(len, samples)
    results['mean_length'] = np.mean(samples_lens)
    results['n_samples'] = len(samples)
    results['uniq_samples'] = len(set(samples))
    verified_samples = filter(verify_sequence, samples)
    unverified_samples = filter(lambda v: not verify_sequence(v), samples)
    results['good_samples'] = len(verified_samples)
    results['bad_samples'] = len(unverified_samples)
    metrics = {'novelty': batch_novelty, 'diversity': batch_diversity}
    for key, func in metrics.items():
        results[key] = np.mean(func(verified_samples, train_samples))

    # save smiles
    if 'Batch' in results.keys():
        smi_name = '{}_{}'.format(results['exp_name'], results['Batch'])
        save_smi(smi_name, samples)
        results['model_samples'] = smi_name
    # print results
    if verbose:
        print_results(verified_samples, unverified_samples, results)
    return


def print_results(verified_samples, unverified_samples, results={}):
    print('~~~ Summary Results ~~~')
    print('Total samples: {}'.format(results['n_samples']))
    percent = results['uniq_samples'] / float(results['n_samples']) * 100
    print('Unique      : {:6d} ({:2.2f}%)'.format(
        results['uniq_samples'], percent))
    percent = results['good_samples'] / float(results['n_samples']) * 100
    print('Verified    : {:6d} ({:2.2f}%)'.format(
        results['good_samples'], percent))
    percent = results['bad_samples'] / float(results['n_samples']) * 100
    print('Unverified  : {:6d} ({:2.2f}%)'.format(
        results['bad_samples'], percent))
    for i in ['novelty', 'diversity']:
        print('{:11s} : {:1.4f}'.format(i, results[i]))

    print('\nExample of good samples:')
    for s in verified_samples[0:10]:
        print('' + s)
    print('\nExample of bad samples:')
    for s in unverified_samples[0:10]:
        print('' + s)

    return

#====== diversity metric


def batch_diversity(smiles, train_smiles):
    return diversity(smiles, train_smiles)


def diversity(smiles, train_smiles=None):
    df = pd.DataFrame({'smiles': smiles})
    PandasTools.AddMoleculeColumnToFrame(df, 'smiles', 'mol')
    fps = [Chem.GetMorganFingerprintAsBitVect(
        m, 4, nBits=2048) for m in df['mol']]
    dist_1d = tanimoto_1d(fps)
    mean_dist = np.mean(dist_1d)
    mean_rand = 0.91549  # mean random distance
    mean_diverse = 0.94170  # mean diverse distance
    norm_dist = remap(mean_dist, mean_rand, mean_diverse)
    norm_dist = np.clip(norm_dist, 0.0, 1.0)
    return norm_dist


def tanimoto_1d(fps):
    ds = []
    for i in range(1, len(fps)):
        ds.extend(DataStructs.BulkTanimotoSimilarity(
            fps[i], fps[:i], returnDistance=True))
    return ds

#==============


def batch_novelty(smiles, train_smiles):
    return np.mean([novelty(smile, train_smiles) for smile in smiles])


def novelty(smile, train_smiles):
    newness = 1.0 if smile not in train_smiles else 0.0
    return newness


#======= solubility

def logP(smile):
    val = 0.0
    # min_logp =
    # max_logp =
    if verify_sequence(smile):
        logp = Crippen.MolLogP(Chem.MolFromSmiles(smile))
        # val = remap(logp,min_logp,max_logp)
        val = np.clip(logp, 0.0, 1.0)

    return val
