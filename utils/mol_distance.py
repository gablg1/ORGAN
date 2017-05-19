import string
import pandas as pd
import numpy as np
from rdkit.Chem import AllChem as Chem
from rdkit.Chem import Draw, PandasTools
from rdkit import DataStructs
from rdkit.Chem import rdMolDescriptors
#import cairosvg
import pickle
import sys, gzip
import os.path as op
import math
import editdistance
from rdkit.six import iteritems

def readNPModel(filename='publicnp.model.gz'):
  sys.stderr.write("reading NP model ...\n")
  fscore = pickle.load(gzip.open(filename))
  sys.stderr.write("model in\n")
  return fscore

def tanimoto_1d(fps):
    ds = []
    for i in range(1, len(fps)):
        ds.extend(DataStructs.BulkTanimotoSimilarity(
            fps[i], fps[:i], returnDistance=True))
    return ds


def save_svg(svg, svg_file, dpi=300):
    png_file = svg_file.replace('.svg', '.png')
    with open(svg_file, 'w') as afile:
        afile.write(svg)
    a_str = svg.encode('utf-8')
    #cairosvg.svg2png(bytestring=a_str, write_to=png_file, dpi=dpi)
    return


def molgrid_image(smiles, file_name, labels=None, molPerRow=5):
    df = pd.DataFrame({'smiles': smiles})
    PandasTools.AddMoleculeColumnToFrame(df, 'smiles', 'mol')
    if labels is None:
        labels = ['{:d}'.format(i) for i in df.index]
    svg = Draw.MolsToGridImage(
        df['mol'], molsPerRow=5, legends=labels, useSVG=True)
    save_svg(svg, file_name + '.svg', dpi=150)
    return


def mol_diversity(smiles):
    df = pd.DataFrame({'smiles': smiles})
    PandasTools.AddMoleculeColumnToFrame(df, 'smiles', 'mol')
    fps = [Chem.GetMorganFingerprintAsBitVect(
        m, 4, nBits=2048) for m in df['mol']]
    dist_1d = tanimoto_1d(fps)
    mean_dist = np.mean(dist_1d)
    return mean_dist
    mean_rand = 0.91549  # mean random distance
    mean_diverse = 0.94170  # mean diverse distance
    norm_dist = (mean_dist - mean_rand) / (mean_diverse - mean_rand)
    return norm_dist

def SA_scores(smiles):
    fscores = readFragmentScores(name='fpscores')
    df = pd.DataFrame({'smiles': smiles})
    PandasTools.AddMoleculeColumnToFrame(df, 'smiles', 'mol')
    scores = [ SA_score(m,fscores) for m in df['mol']]
    return scores

def readFragmentScores(name='fpscores'):
  import gzip
  global _fscores
  # generate the full path filename:
  if name == "fpscores":
    name = op.join(op.dirname(__file__), name)
  _fscores = pickle.load(gzip.open('%s.pkl.gz' % name))
  outDict = {}
  for i in _fscores:
    for j in range(1, len(i)):
      outDict[i[j]] = float(i[0])
  _fscores = outDict


def numBridgeheadsAndSpiro(mol, ri=None):
  nSpiro = rdMolDescriptors.CalcNumSpiroAtoms(mol)
  nBridgehead = rdMolDescriptors.CalcNumBridgeheadAtoms(mol)
  return nBridgehead, nSpiro

def SA_score(m,fscores=None):
    if fscores is None:
        readFragmentScores()

    # fragment score
    fp = rdMolDescriptors.GetMorganFingerprint(m,
                                               2)  # <- 2 is the *radius* of the circular fingerprint
    fps = fp.GetNonzeroElements()
    score1 = 0.
    nf = 0
    #for bitId, v in fps.items():
    for bitId, v in iteritems(fps):
        nf += v
        sfp = bitId
        score1 += _fscores.get(sfp, -4) * v
    score1 /= nf

    # features score
    nAtoms = m.GetNumAtoms()
    nChiralCenters = len(Chem.FindMolChiralCenters(m, includeUnassigned=True))
    ri = m.GetRingInfo()
    nBridgeheads, nSpiro = numBridgeheadsAndSpiro(m, ri)
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

    return sascore

def NP_scores(smiles):
    fscore =readNPModel()
    df = pd.DataFrame({'smiles': smiles})
    PandasTools.AddMoleculeColumnToFrame(df, 'smiles', 'mol')
    scores = [ NP_score(m,fscore) for m in df['mol']]
    return scores

def NP_score(mol, fscore=None):
    if fscore is None:
        fscore =readNPModel()
    if mol is None:
            raise ValueError('invalid molecule')
    fp = rdMolDescriptors.GetMorganFingerprint(mol, 2)
    bits = fp.GetNonzeroElements()

    # calculating the score
    score = 0.
    for bit in bits:
        score += fscore.get(bit, 0)
    score /= float(mol.GetNumAtoms())

    # preventing score explosion for exotic molecules
    if score > 4:
        score = 4. + math.log10(score - 4. + 1.)
    if score < -4:
        score = -4. - math.log10(-4. - score + 1.)
    return score

def hamming(s1, s2):
    assert(len(s1) == len(s2))
    return float(sum(c1 != c2 for c1, c2 in zip(s1, s2))) / len(s1)

def levenshtein(s1, s2):
    return float(editdistance.eval(s1, s2)) / max(len(s1), len(s2))

def H(data, characters):
    if not data:
        return 0
    entropy = 0
    for x in characters:
        p_x = float(data.count(x))/len(data)
        if p_x > 0:
            entropy += - p_x*math.log(p_x, 2)
    return entropy


strings = []
for line in sys.stdin:
    strings.append(line)


def pad(s, max_len, pad_char = '_'):
    if len(s) >= max_len:
    	return s
    return s + pad_char * (max_len - len(s))

strings = [s.rstrip() for s in strings if not s.isspace()]

levenshtein_distances = [levenshtein(x, y) for x in strings for y in strings]
print 'Average length', np.mean([len(s) for s in strings])
print 'Tanimoto diversity', mol_diversity(strings)

max_len = max([len(s) for s in strings])

strings = [pad(s, max_len) for s in strings]


hamming_distances = [hamming(x, y) for x in strings for y in strings]
entropies = [H(s, string.printable) for s in strings]
print 'Mean entropy', np.mean(entropies)
print 'Mean pairwise Hamming distance', np.mean(hamming_distances)
print 'Mean pairwise Levenshtein distance', np.mean(levenshtein_distances)



