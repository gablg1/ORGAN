from __future__ import absolute_import, division, print_function
from builtins import range
import os
import numpy as np
import csv
import time
import pickle
import gzip
import math
import random
from rdkit import rdBase
from rdkit import DataStructs
from rdkit.Chem import AllChem as Chem
from rdkit.Chem import Crippen, MolFromSmiles, MolToSmiles
# Disables logs for Smiles conversion
rdBase.DisableLog('rdApp.error')
#====== load data


def readNPModel(filename='NP_score.pkl.gz'):
    print("mol_metrics: reading NP model ...")
    start = time.time()
    NP_model = pickle.load(gzip.open(filename))
    end = time.time()
    print("loaded in {}".format(end - start))
    return NP_model

NP_model = readNPModel()


def readSAModel(filename='SA_score.pkl.gz'):
    print("mol_metrics: reading SA model ...")
    start = time.time()
    model_data = pickle.load(gzip.open(filename))
    outDict = {}
    for i in model_data:
        for j in range(1, len(i)):
            outDict[i[j]] = float(i[0])
    SA_model = outDict
    end = time.time()
    print("loaded in {}".format(end - start))
    return SA_model

SA_model = readSAModel()
#====== math utility


def remap(x, x_min, x_max):
    return (x - x_min) / (x_max - x_min)


def constant_bump(x, x_low, x_high, decay=0.025):
    if x <= x_low:
        return np.exp(-(x - x_low)**2 / decay)
    elif x >= x_high:
        return np.exp(-(x - x_high)**2 / decay)
    else:
        return 1
    return


def pct(a, b):
    if len(b) == 0:
        return 0
    return float(len(a)) / len(b)

#====== encoding/decoding utility


def canon_smile(smile):
    return MolToSmiles(MolFromSmiles(smile))


def verified_and_below(smile, max_len):
    return len(smile) < max_len and verify_sequence(smile)


def verify_sequence(smile):
    mol = Chem.MolFromSmiles(smile)
    return smile != '' and mol is not None and mol.GetNumAtoms() > 1


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


def encode(smile, max_len, char_dict): return [
    char_dict[c] for c in pad(smile, max_len)]


def decode(ords, ord_dict): return unpad(
    ''.join([ord_dict[o] for o in ords]))


def load_train_data(filename):
    ext = filename.split(".")[-1]
    if ext == 'csv':
        return read_smiles_csv(filename)
    if ext == 'smi':
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
    if not os.path.exists('epoch_data'):
        os.makedirs('epoch_data')
    smi_file = os.path.join('epoch_data', "{}.smi".format(name))
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


def compute_results(model_samples, train_data, ord_dict, results={}, verbose=True):
    samples = [decode(s, ord_dict) for s in model_samples]
    results['mean_length'] = np.mean([len(sample) for sample in samples])
    results['n_samples'] = len(samples)
    results['uniq_samples'] = len(set(samples))
    verified_samples = [
        sample for sample in samples if verify_sequence(sample)]
    unverified_samples = [
        sample for sample in samples if not verify_sequence(sample)]
    results['good_samples'] = len(verified_samples)
    results['bad_samples'] = len(unverified_samples)
    # collect metrics
    metrics = ['novelty', 'hard_novelty', 'soft_novelty',
               'diversity', 'conciseness', 'solubility',
               'naturalness', 'synthesizability']

    for objective in metrics:
        func = load_reward(objective)
        results[objective] = np.mean(func(verified_samples, train_data))

    # save smiles
    if 'Batch' in results.keys():
        smi_name = '{}_{}'.format(results['exp_name'], results['Batch'])
        save_smi(smi_name, samples)
        results['model_samples'] = smi_name
    # print results
    if verbose:
        print_results(verified_samples, unverified_samples,
                      metrics, results)
    return


def print_results(verified_samples, unverified_samples, metrics, results={}):
    print('~~~ Summary Results ~~~')
    print('{:15s} : {:6d}'.format("Total samples", results['n_samples']))
    percent = results['uniq_samples'] / float(results['n_samples']) * 100
    print('{:15s} : {:6d} ({:2.2f}%)'.format(
        'Unique', results['uniq_samples'], percent))
    percent = results['bad_samples'] / float(results['n_samples']) * 100
    print('{:15s} : {:6d} ({:2.2f}%)'.format('Unverified',
                                             results['bad_samples'], percent))
    percent = results['good_samples'] / float(results['n_samples']) * 100
    print('{:15s} : {:6d} ({:2.2f}%)'.format(
        'Verified', results['good_samples'], percent))
    print('\tmetrics...')
    for i in metrics:
        print('{:20s} : {:1.4f}'.format(i, results[i]))

    if len(verified_samples) > 10:
        print('\nExample of good samples:')

        for s in verified_samples[0:10]:
            print('' + s)
    else:
        print('\nno good samples found :(')

    if len(unverified_samples) > 10:
        print('\nExample of bad samples:')
        for s in unverified_samples[0:10]:
            print('' + s)
    else:
        print('\nno bad samples found :(')

    print('~~~~~~~~~~~~~~~~~~~~~~~')
    return

#====== diversity metric


def batch_diversity(smiles, train_smiles):
    rand_smiles = random.sample(train_smiles, 100)
    rand_mols = [Chem.MolFromSmiles(s) for s in rand_smiles]
    fps = [Chem.GetMorganFingerprintAsBitVect(
        m, 4, nBits=2048) for m in rand_mols]
    vals = [diversity(s, fps) if verify_sequence(s)
            else 0.0 for s in smiles]
    return vals


def diversity(smile, fps):
    val = 0.0
    low_rand_dst = 0.9
    mean_div_dst = 0.945
    ref_mol = Chem.MolFromSmiles(smile)
    ref_fps = Chem.GetMorganFingerprintAsBitVect(ref_mol, 4, nBits=2048)
    dist = DataStructs.BulkTanimotoSimilarity(
        ref_fps, fps, returnDistance=True)
    mean_dist = np.mean(np.array(dist))
    val = remap(mean_dist, low_rand_dst, mean_div_dst)
    val = np.clip(val, 0.0, 1.0)
    return val

#==============


def batch_novelty(smiles, train_smiles):
    vals = [novelty(smile, train_smiles) if verify_sequence(
        smile) else 0 for smile in smiles]
    return vals


def batch_hardnovelty(smiles, train_smiles):
    vals = [hard_novelty(smile, train_smiles) if verify_sequence(
        smile) else 0 for smile in smiles]
    return vals


def batch_softnovelty(smiles, train_smiles):
    vals = [soft_novelty(smile, train_smiles) if verify_sequence(
        smile) else 0 for smile in smiles]

    return vals


def novelty(smile, train_smiles):
    newness = 1.0 if smile not in train_smiles else 0.0
    return newness

# assumes you already filtered verified molecules


def soft_novelty(smile, train_smiles):
    newness = 1.0 if smile not in train_smiles else 0.3
    return newness


def hard_novelty(smile, train_smiles):
    newness = 1.0 if canon_smile(smile) not in train_smiles else 0.0
    return newness


#======= solubility

def batch_solubility(smiles, train_smiles=None):
    vals = [logP(s, train_smiles) if verify_sequence(s) else 0 for s in smiles]
    return vals


def logP(smile, train_smiles=None):
    low_logp = -2.12178879609
    high_logp = 6.0429063424
    logp = Crippen.MolLogP(Chem.MolFromSmiles(smile))
    val = remap(logp, low_logp, high_logp)
    val = np.clip(val, 0.0, 1.0)
    return val

#====== DrugCandidate


def drug_candidate(smile, train_smiles):
    good_logp = constant_bump(logP(smile), 0.210, 0.945)
    sa = SA_score(smile)
    novel = soft_novelty(smile, train_smiles)
    compact = conciseness(smile)
    val = (compact + good_logp + sa + novel) / 4.0
    return val


def batch_drugcandidate(smiles, train_smiles):
    vals = [drug_candidate(s, train_smiles)
            if verify_sequence(s) else 0 for s in smiles]
    return vals

#====== Conciseness


def batch_conciseness(smiles, train_smiles=None):
    vals = [conciseness(s) if verify_sequence(s) else 0 for s in smiles]
    return vals


def conciseness(smile, train_smiles=None):
    canon = canon_smile(smile)
    diff_len = len(smile) -len(canon)
    val = np.clip(diff_len, 0.0, 20)
    val = 1 - 1.0 / 20.0 * val
    return val

#====== Contains substructure


def substructure_match(smile, train_smiles=None, sub_mol=None):
    mol = Chem.MolFromSmiles(smile)
    val = mol.HasSubstructMatch(sub_mol)
    return int(val)

#====== NP-likeliness


def NP_score(smile):
    mol = Chem.MolFromSmiles(smile)
    fp = Chem.GetMorganFingerprint(mol, 2)
    bits = fp.GetNonzeroElements()

    # calculating the score
    score = 0.
    for bit in bits:
        score += NP_model.get(bit, 0)
    score /= float(mol.GetNumAtoms())

    # preventing score explosion for exotic molecules
    if score > 4:
        score = 4. + math.log10(score - 4. + 1.)
    if score < -4:
        score = -4. - math.log10(-4. - score + 1.)
    val = np.clip(remap(score, -3, 1), 0.0, 1.0)
    return val


def batch_NPLikeliness(smiles, train_smiles=None):
    scores = [NP_score(s) if verify_sequence(s) else 0 for s in smiles]
    return scores
#===== Synthetics Accesability score ===


def SA_score(smile):
    mol = Chem.MolFromSmiles(smile)
    # fragment score
    fp = Chem.GetMorganFingerprint(mol, 2)
    fps = fp.GetNonzeroElements()
    score1 = 0.
    nf = 0
    # for bitId, v in fps.items():
    for bitId, v in fps.items():
        nf += v
        sfp = bitId
        score1 += SA_model.get(sfp, -4) * v
    score1 /= nf

    # features score
    nAtoms = mol.GetNumAtoms()
    nChiralCenters = len(Chem.FindMolChiralCenters(
        mol, includeUnassigned=True))
    ri = mol.GetRingInfo()
    nSpiro = Chem.CalcNumSpiroAtoms(mol)
    nBridgeheads = Chem.CalcNumBridgeheadAtoms(mol)
    nMacrocycles = 0
    for x in ri.AtomRings():
        if len(x) > 8:
            nMacrocycles += 1

    sizePenalty = nAtoms**1.005 - nAtoms
    stereoPenalty = math.log10(nChiralCenters + 1)
    spiroPenalty = math.log10(nSpiro + 1)
    bridgePenalty = math.log10(nBridgeheads + 1)
    macrocyclePenalty = 0.
    # ---------------------------------------
    # This differs from the paper, which defines:
    #  macrocyclePenalty = math.log10(nMacrocycles+1)
    # This form generates better results when 2 or more macrocycles are present
    if nMacrocycles > 0:
        macrocyclePenalty = math.log10(2)

    score2 = 0. - sizePenalty - stereoPenalty - \
        spiroPenalty - bridgePenalty - macrocyclePenalty

    # correction for the fingerprint density
    # not in the original publication, added in version 1.1
    # to make highly symmetrical molecules easier to synthetise
    score3 = 0.
    if nAtoms > len(fps):
        score3 = math.log(float(nAtoms) / len(fps)) * .5

    sascore = score1 + score2 + score3

    # need to transform "raw" value into scale between 1 and 10
    min = -4.0
    max = 2.5
    sascore = 11. - (sascore - min + 1) / (max - min) * 9.
    # smooth the 10-end
    if sascore > 8.:
        sascore = 8. + math.log(sascore + 1. - 9.)
    if sascore > 10.:
        sascore = 10.0
    elif sascore < 1.:
        sascore = 1.0
    val = remap(sascore, 5, 1.5)
    val = np.clip(val, 0.0, 1.0)
    return val


def batch_SA(smiles, train_smiles=None):
    scores = [SA_score(s) for s in smiles]
    return scores


#===== Reward function


def load_reward(objective):

    metrics = {}
    metrics['novelty'] = batch_novelty
    metrics['hard_novelty'] = batch_hardnovelty
    metrics['soft_novelty'] = batch_softnovelty
    metrics['diversity'] = batch_diversity
    metrics['conciseness'] = batch_conciseness
    metrics['solubility'] = batch_solubility
    metrics['naturalness'] = batch_NPLikeliness
    metrics['synthesizability'] = batch_SA
    metrics['drug_candidate'] = batch_drugcandidate
    if objective in metrics.keys():
        return metrics[objective]
    else:
        raise ValueError('objective {} not found!'.format(objective))
    return
