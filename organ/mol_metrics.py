from __future__ import absolute_import, division, print_function
from builtins import range
import organ
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
from rdkit.Chem import Descriptors
from copy import deepcopy
from math import exp, log
# Disables logs for Smiles conversion
rdBase.DisableLog('rdApp.error')
#====== load data


def readNPModel(filename='NP_score.pkl.gz'):
    print("mol_metrics: reading NP model ...")
    start = time.time()
    if filename == 'NP_score.pkl.gz':
        filename = os.path.join(os.path.dirname(organ.__file__), filename)
    NP_model = pickle.load(gzip.open(filename))
    end = time.time()
    print("loaded in {}".format(end - start))
    return NP_model


NP_model = readNPModel()


def readSAModel(filename='SA_score.pkl.gz'):
    print("mol_metrics: reading SA model ...")
    start = time.time()
    if filename == 'SA_score.pkl.gz':
        filename = os.path.join(os.path.dirname(organ.__file__), filename)
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

#====== qed variables

AliphaticRings = Chem.MolFromSmarts('[$([A;R][!a])]')

AcceptorSmarts = [
    '[oH0;X2]',
    '[OH1;X2;v2]',
    '[OH0;X2;v2]',
    '[OH0;X1;v2]',
    '[O-;X1]',
    '[SH0;X2;v2]',
    '[SH0;X1;v2]',
    '[S-;X1]',
    '[nH0;X2]',
    '[NH0;X1;v3]',
    '[$([N;+0;X3;v3]);!$(N[C,S]=O)]'
]
Acceptors = []
for hba in AcceptorSmarts:
    Acceptors.append(Chem.MolFromSmarts(hba))

StructuralAlertSmarts = [
    '*1[O,S,N]*1',
    '[S,C](=[O,S])[F,Br,Cl,I]',
    '[CX4][Cl,Br,I]',
    '[C,c]S(=O)(=O)O[C,c]',
    '[$([CH]),$(CC)]#CC(=O)[C,c]',
    '[$([CH]),$(CC)]#CC(=O)O[C,c]',
    'n[OH]',
    '[$([CH]),$(CC)]#CS(=O)(=O)[C,c]',
    'C=C(C=O)C=O',
    'n1c([F,Cl,Br,I])cccc1',
    '[CH1](=O)',
    '[O,o][O,o]',
    '[C;!R]=[N;!R]',
    '[N!R]=[N!R]',
    '[#6](=O)[#6](=O)',
    '[S,s][S,s]',
    '[N,n][NH2]',
    'C(=O)N[NH2]',
    '[C,c]=S',
    '[$([CH2]),$([CH][CX4]),$(C([CX4])[CX4])]=[$([CH2]),$([CH][CX4]),$(C([CX4])[CX4])]',
    'C1(=[O,N])C=CC(=[O,N])C=C1',
    'C1(=[O,N])C(=[O,N])C=CC=C1',
    'a21aa3a(aa1aaaa2)aaaa3',
    'a31a(a2a(aa1)aaaa2)aaaa3',
    'a1aa2a3a(a1)A=AA=A3=AA=A2',
    'c1cc([NH2])ccc1',
    '[Hg,Fe,As,Sb,Zn,Se,se,Te,B,Si,Na,Ca,Ge,Ag,Mg,K,Ba,Sr,Be,Ti,Mo,Mn,Ru,Pd,Ni,Cu,Au,Cd,Al,Ga,Sn,Rh,Tl,Bi,Nb,Li,Pb,Hf,Ho]',
    'I',
    'OS(=O)(=O)[O-]',
    '[N+](=O)[O-]',
    'C(=O)N[OH]',
    'C1NC(=O)NC(=O)1',
    '[SH]',
    '[S-]',
    'c1ccc([Cl,Br,I,F])c([Cl,Br,I,F])c1[Cl,Br,I,F]',
    'c1cc([Cl,Br,I,F])cc([Cl,Br,I,F])c1[Cl,Br,I,F]',
    '[CR1]1[CR1][CR1][CR1][CR1][CR1][CR1]1',
    '[CR1]1[CR1][CR1]cc[CR1][CR1]1',
    '[CR2]1[CR2][CR2][CR2][CR2][CR2][CR2][CR2]1',
    '[CR2]1[CR2][CR2]cc[CR2][CR2][CR2]1',
    '[CH2R2]1N[CH2R2][CH2R2][CH2R2][CH2R2][CH2R2]1',
    '[CH2R2]1N[CH2R2][CH2R2][CH2R2][CH2R2][CH2R2][CH2R2]1',
    'C#C',
    '[OR2,NR2]@[CR2]@[CR2]@[OR2,NR2]@[CR2]@[CR2]@[OR2,NR2]',
    '[$([N+R]),$([n+R]),$([N+]=C)][O-]',
    '[C,c]=N[OH]',
    '[C,c]=NOC=O',
    '[C,c](=O)[CX4,CR0X3,O][C,c](=O)',
    'c1ccc2c(c1)ccc(=O)o2',
    '[O+,o+,S+,s+]',
    'N=C=O',
    '[NX3,NX4][F,Cl,Br,I]',
    'c1ccccc1OC(=O)[#6]',
    '[CR0]=[CR0][CR0]=[CR0]',
    '[C+,c+,C-,c-]',
    'N=[N+]=[N-]',
    'C12C(NC(N1)=O)CSC2',
    'c1c([OH])c([OH,NH2,NH])ccc1',
    'P',
    '[N,O,S]C#N',
    'C=C=O',
    '[Si][F,Cl,Br,I]',
    '[SX2]O',
    '[SiR0,CR0](c1ccccc1)(c2ccccc2)(c3ccccc3)',
    'O1CCCCC1OC2CCC3CCCCC3C2',
    'N=[CR0][N,n,O,S]',
    '[cR2]1[cR2][cR2]([Nv3X3,Nv4X4])[cR2][cR2][cR2]1[cR2]2[cR2][cR2][cR2]([Nv3X3,Nv4X4])[cR2][cR2]2',
    'C=[C!r]C#N',
    '[cR2]1[cR2]c([N+0X3R0,nX3R0])c([N+0X3R0,nX3R0])[cR2][cR2]1',
    '[cR2]1[cR2]c([N+0X3R0,nX3R0])[cR2]c([N+0X3R0,nX3R0])[cR2]1',
    '[cR2]1[cR2]c([N+0X3R0,nX3R0])[cR2][cR2]c1([N+0X3R0,nX3R0])',
    '[OH]c1ccc([OH,NH2,NH])cc1',
    'c1ccccc1OC(=O)O',
    '[SX2H0][N]',
    'c12ccccc1(SC(S)=N2)',
    'c12ccccc1(SC(=S)N2)',
    'c1nnnn1C=O',
    's1c(S)nnc1NC=O',
    'S1C=CSC1=S',
    'C(=O)Onnn',
    'OS(=O)(=O)C(F)(F)F',
    'N#CC[OH]',
    'N#CC(=O)',
    'S(=O)(=O)C#N',
    'N[CH2]C#N',
    'C1(=O)NCC1',
    'S(=O)(=O)[O-,OH]',
    'NC[F,Cl,Br,I]',
    'C=[C!r]O',
    '[NX2+0]=[O+0]',
    '[OR0,NR0][OR0,NR0]',
    'C(=O)O[C,H1].C(=O)O[C,H1].C(=O)O[C,H1]',
    '[CX2R0][NX3R0]',
    'c1ccccc1[C;!R]=[C;!R]c2ccccc2',
    '[NX3R0,NX4R0,OR0,SX2R0][CX4][NX3R0,NX4R0,OR0,SX2R0]',
    '[s,S,c,C,n,N,o,O]~[n+,N+](~[s,S,c,C,n,N,o,O])(~[s,S,c,C,n,N,o,O])~[s,S,c,C,n,N,o,O]',
    '[s,S,c,C,n,N,o,O]~[nX3+,NX3+](~[s,S,c,C,n,N])~[s,S,c,C,n,N]',
    '[*]=[N+]=[*]',
    '[SX3](=O)[O-,OH]',
    'N#N',
    'F.F.F.F',
    '[R0;D2][R0;D2][R0;D2][R0;D2]',
    '[cR,CR]~C(=O)NC(=O)~[cR,CR]',
    'C=!@CC=[O,S]',
    '[#6,#8,#16][C,c](=O)O[C,c]',
    'c[C;R0](=[O,S])[C,c]',
    'c[SX2][C;!R]',
    'C=C=C',
    'c1nc([F,Cl,Br,I,S])ncc1',
    'c1ncnc([F,Cl,Br,I,S])c1',
    'c1nc(c2c(n1)nc(n2)[F,Cl,Br,I])',
    '[C,c]S(=O)(=O)c1ccc(cc1)F',
    '[15N]',
    '[13C]',
    '[18O]',
    '[34S]'
]

StructuralAlerts = []
for smarts in StructuralAlertSmarts:
    StructuralAlerts.append(Chem.MolFromSmarts(smarts))


# ADS parameters for the 8 molecular properties: [row][column]
#     rows[8]:     MW, ALOGP, HBA, HBD, PSA, ROTB, AROM, ALERTS
#     columns[7]: A, B, C, D, E, F, DMAX
# ALOGP parameters from Gregory Gerebtzoff (2012, Roche)
pads1 = [[2.817065973, 392.5754953, 290.7489764, 2.419764353, 49.22325677, 65.37051707, 104.9805561],
         [0.486849448, 186.2293718, 2.066177165, 3.902720615,
             1.027025453, 0.913012565, 145.4314800],
         [2.948620388, 160.4605972, 3.615294657, 4.435986202,
             0.290141953, 1.300669958, 148.7763046],
         [1.618662227, 1010.051101, 0.985094388, 0.000000001,
             0.713820843, 0.920922555, 258.1632616],
         [1.876861559, 125.2232657, 62.90773554, 87.83366614,
             12.01999824, 28.51324732, 104.5686167],
         [0.010000000, 272.4121427, 2.558379970, 1.565547684,
             1.271567166, 2.758063707, 105.4420403],
         [3.217788970, 957.7374108, 2.274627939, 0.000000001,
             1.317690384, 0.375760881, 312.3372610],
         [0.010000000, 1199.094025, -0.09002883, 0.000000001, 0.185904477, 0.875193782, 417.7253140]]
# ALOGP parameters from the original publication
pads2 = [[2.817065973, 392.5754953, 290.7489764, 2.419764353, 49.22325677, 65.37051707, 104.9805561],
         [3.172690585, 137.8624751, 2.534937431, 4.581497897,
             0.822739154, 0.576295591, 131.3186604],
         [2.948620388, 160.4605972, 3.615294657, 4.435986202,
             0.290141953, 1.300669958, 148.7763046],
         [1.618662227, 1010.051101, 0.985094388, 0.000000001,
             0.713820843, 0.920922555, 258.1632616],
         [1.876861559, 125.2232657, 62.90773554, 87.83366614,
             12.01999824, 28.51324732, 104.5686167],
         [0.010000000, 272.4121427, 2.558379970, 1.565547684,
             1.271567166, 2.758063707, 105.4420403],
         [3.217788970, 957.7374108, 2.274627939, 0.000000001,
             1.317690384, 0.375760881, 312.3372610],
         [0.010000000, 1199.094025, -0.09002883, 0.000000001, 0.185904477, 0.875193782, 417.7253140]]


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


# def build_vocab(smiles, pad_char='_', start_char='^'):
#     i = 1
#     char_dict, ord_dict = {start_char: 0}, {0: start_char}
#     for smile in smiles:
#         for c in smile:
#             if c not in char_dict:
#                 char_dict[c] = i
#                 ord_dict[i] = c
#                 i += 1
#     char_dict[pad_char], ord_dict[i] = i, pad_char
#     return char_dict, ord_dict


# def pad(smile, n, pad_char='_'):
#     if n < len(smile):
#         return smile
#     return smile + pad_char * (n - len(smile))


# def unpad(smile, pad_char='_'): return smile.rstrip(pad_char)


# def encode(smile, max_len, char_dict): return [
#     char_dict[c] for c in pad(smile, max_len)]


# def decode(ords, ord_dict): return unpad(
#     ''.join([ord_dict[o] for o in ords]))

def build_vocab(smiles=None, pad_char='_', start_char='^'):
    # smile syntax
    chars = []
    # atoms (carbon), replace Cl for Q and Br for W
    chars = chars + ['H', 'B', 'c', 'C', 'n', 'N',
                     'o', 'O', 'p', 'P', 's', 'S', 'F', 'Q', 'W', 'I']
    # Atom modifiers: negative charge - has been replaced with ~
    # added explicit hidrogens as Z (H2) and X (H3)
    # negative charge ~ (-), ! (-2),,'&' (-3)
    # positive charge +, u (+2), y (+3)
    chars = chars + ['[', ']', '+', 'u', 'y', '~', '!', '&', 'Z', 'X']
    # bonding
    chars = chars + ['-', '=', '#']
    # branches
    chars = chars + ['(', ')']
    # cycles
    chars = chars + ['1', '2', '3', '4', '5', '6', '7', ]
    # anit/clockwise
    chars = chars + ['@']
    # directional bonds
    chars = chars + ['/', '\\']
    # Disconnected structures
    chars = chars + ['.']

    char_dict = {}
    char_dict[start_char] = 0
    for i, c in enumerate(chars):
        char_dict[c] = i + 1
    # end and start
    char_dict[pad_char] = i + 2

    ord_dict = {v: k for k, v in char_dict.items()}

    return char_dict, ord_dict


def pad(smi, n, pad_char='_'):
    if n < len(smi):
        return smi
    return smi + pad_char * (n - len(smi))


def unpad(smi, pad_char='_'):
    return smi.rstrip(pad_char)


def encode(smi, max_len, char_dict):
    # replace double char atoms symbols
    smi = smi.replace('Cl', 'Q')
    smi = smi.replace('Br', 'W')

    atom_spec = False
    new_chars = [''] * max_len
    i = 0
    for c in smi:
        if c == '[':
            atom_spec = True
            spec = []
        if atom_spec:
            spec.append(c)
        else:
            new_chars[i] = c
            i = i + 1
        # close atom spec
        if c == ']':
            atom_spec = False
            spec = ''.join(spec)
            # negative charges
            spec = spec.replace('-3', '&')
            spec = spec.replace('-2', '!')
            spec = spec.replace('-', '~')
            # positive charges
            spec = spec.replace('+3', 'y')
            spec = spec.replace('+2', 'u')
            # hydrogens
            spec = spec.replace('H2', 'Z')
            spec = spec.replace('H3', 'X')

            new_chars[i:i + len(spec)] = spec
            i = i + len(spec)

    new_smi = ''.join(new_chars)
    return [char_dict[c] for c in pad(new_smi, max_len)]


def decode(ords, ord_dict):

    smi = unpad(''.join([ord_dict[o] for o in ords]))
    # negative charges
    smi = smi.replace('~', '-')
    smi = smi.replace('!', '-2')
    smi = smi.replace('&', '-3')
    # positive charges
    smi = smi.replace('y', '+3')
    smi = smi.replace('u', '+2')
    # hydrogens
    smi = smi.replace('Z', 'H2')
    smi = smi.replace('X', 'H3')
    # replace proxy atoms for double char atoms symbols
    smi = smi.replace('Q', 'Cl')
    smi = smi.replace('W', 'Br')

    return smi


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


def compute_results(reward, model_samples, train_data, ord_dict, results={}, verbose=True):
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


    if verbose:
        print_results(verified_samples, unverified_samples, [], results)

    if not verified_samples:
        verified_samples = 'c1ccccc1'

    # save smiles
    if 'Batch' in results.keys():
        smi_name = '{}_{}'.format(results['exp_name'], results['Batch'])
        save_smi(smi_name, samples)
        results['model_samples'] = smi_name
    # print results
    if verbose:
        print_results(verified_samples, unverified_samples, [], results)
    return


def print_results(verified_samples, unverified_samples, metrics=[], results={}):
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
        print('\nno bad samples found :S')

    print('~~~~~~~~~~~~~~~~~~~~~~~')
    return

#====== diversity metric


def batch_diversity(smiles, set_smiles):
    rand_smiles = random.sample(set_smiles, 100)
    rand_mols = [Chem.MolFromSmiles(s) for s in rand_smiles]
    fps = [Chem.GetMorganFingerprintAsBitVect(
        m, 4, nBits=2048) for m in rand_mols]
    vals = [diversity(s, fps) if verify_sequence(s)
            else 0.0 for s in smiles]
    return vals


def batch_mixed_diversity(smiles, set_smiles):
    # set smiles
    rand_smiles = random.sample(set_smiles, 100)
    rand_mols = [Chem.MolFromSmiles(s) for s in rand_smiles]
    fps = [Chem.GetMorganFingerprintAsBitVect(
        m, 4, nBits=2048) for m in rand_mols]
    # gen smiles
    rand_gen_smiles = random.sample(smiles, 500)

    gen_mols = [Chem.MolFromSmiles(s) for s in smiles]
    fps = [Chem.GetMorganFingerprintAsBitVect(
        m, 4, nBits=2048) for m in gen_mols]

    vals = [diversity(s, fps) + diversity(s, fps) if verify_sequence(s)
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


def batch_hardnovelty(smiles, set_smiles):
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
    try:
        low_logp = -2.12178879609
        high_logp = 6.0429063424
        logp = Crippen.MolLogP(Chem.MolFromSmiles(smile))
        val = remap(logp, low_logp, high_logp)
        val = np.clip(val, 0.0, 1.0)
        return val
    except ValueError:
        return 0.0

#====== druglikeliness


def ads(x, a, b, c, d, e, f, dmax):
    return ((a + (b / (1 + exp(-1 * (x - c + d / 2) / e)) * (1 - 1 / (1 + exp(-1 * (x - c - d / 2) / f))))) / dmax)


def properties(mol):
    """
    Calculates the properties that are required to calculate the QED descriptor.
    """
    matches = []
    if mol is None:
        raise WrongArgument("properties(mol)", "mol argument is \'None\'")
    x = [0] * 9
    # MW
    x[0] = Descriptors.MolWt(mol)
    # ALOGP
    x[1] = Descriptors.MolLogP(mol)
    for hba in Acceptors:                                                        # HBA
        if mol.HasSubstructMatch(hba):
            matches = mol.GetSubstructMatches(hba)
            x[2] += len(matches)
    x[3] = Descriptors.NumHDonors(
        mol)                                            # HBD
    # PSA
    x[4] = Descriptors.TPSA(mol)
    x[5] = Descriptors.NumRotatableBonds(
        mol)                                    # ROTB
    x[6] = Chem.GetSSSR(Chem.DeleteSubstructs(
        deepcopy(mol), AliphaticRings))    # AROM
    for alert in StructuralAlerts:                                                # ALERTS
        if (mol.HasSubstructMatch(alert)):
            x[7] += 1
    ro5_failed = 0
    if x[3] > 5:
        ro5_failed += 1  # HBD
    if x[2] > 10:
        ro5_failed += 1  # HBA
    if x[0] >= 500:
        ro5_failed += 1
    if x[1] > 5:
        ro5_failed += 1
    x[8] = ro5_failed
    return x


def qed_eval(w, p, gerebtzoff):
    d = [0.00] * 8
    if gerebtzoff:
        for i in range(0, 8):
            d[i] = ads(p[i], pads1[i][0], pads1[i][1], pads1[i][2], pads1[
                       i][3], pads1[i][4], pads1[i][5], pads1[i][6])
    else:
        for i in range(0, 8):
            d[i] = ads(p[i], pads2[i][0], pads2[i][1], pads2[i][2], pads2[
                       i][3], pads2[i][4], pads2[i][5], pads2[i][6])
    t = 0.0
    for i in range(0, 8):
        t += w[i] * log(d[i])
    return (exp(t / sum(w)))


def qed(mol):
    """
    Calculates the QED descriptor using average descriptor weights.
    If props is specified we skip the calculation step and use the props-list of properties.
    """
    props = properties(mol)
    return qed_eval([0.66, 0.46, 0.05, 0.61, 0.06, 0.65, 0.48, 0.95], props, True)


def druglikeliness(smile, train_smiles):
    try:
        val = qed(Chem.MolFromSmiles(smile))
        return val
    except:
        return 0.0
    return val


def batch_druglikeliness(smiles, train_smiles):
    vals = [druglikeliness(s, train_smiles)
            if verify_sequence(s) else 0 for s in smiles]
    return vals

#====== Conciseness


def batch_conciseness(smiles, train_smiles=None):
    vals = [conciseness(s) if verify_sequence(s) else 0 for s in smiles]
    return vals


def conciseness(smile, train_smiles=None):
    canon = canon_smile(smile)
    diff_len = len(smile) - len(canon)
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
    scores = [SA_score(s) if verify_sequence(s) else 0 for s in smiles]
    return scores


#===== Reward function

def metrics_loading():
    loadings = {}
    loadings['novelty'] = lambda *args: None
    loadings['hard_novelty'] = lambda *args: None
    loadings['soft_novelty'] = lambda *args: None
    loadings['diversity'] = lambda *args: None
    loadings['conciseness'] = lambda *args: None
    loadings['solubility'] = lambda *args: None
    loadings['naturalness'] = lambda *args: None
    loadings['synthesizability'] = lambda *args: None
    loadings['druglikeliness'] = lambda *args: None
    return loadings


def get_metrics():
    metrics = {}
    metrics['novelty'] = batch_novelty
    metrics['hard_novelty'] = batch_hardnovelty
    metrics['soft_novelty'] = batch_softnovelty
    metrics['diversity'] = batch_diversity
    metrics['conciseness'] = batch_conciseness
    metrics['solubility'] = batch_solubility
    metrics['naturalness'] = batch_NPLikeliness
    metrics['synthesizability'] = batch_SA
    metrics['druglikeliness'] = batch_druglikeliness
    return metrics
