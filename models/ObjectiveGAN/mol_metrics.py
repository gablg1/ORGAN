import numpy as np
import pandas as pd
from rdkit.Chem import PandasTools
from rdkit import DataStructs
from rdkit.Chem import AllChem as Chem
from rdkit.Chem import Crippen, MolFromSmiles

#====== utility


def verify_sequence(smile):
    return smile != '' and Chem.MolFromSmiles(smile) is not None


def remap(x, x_min, x_max):
    return (x - x_min) / (x_max - x_min)
#====== diversity metric


def diversity(smiles):
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


def novelty(smile, train_smiles):
    newness = 1.0 if smile not in train_smiles else 0.3
    return verify_sequence(smile) * newness


#======= solubility

def logP(smile):
    val = 0.0
    #min_logp = 
    #max_logp =
    if verify_sequence(smile):
        logp = Crippen.MolLogP(Chem.MolFromSmiles(smile))
        #val = remap(logp,min_logp,max_logp)
        val = np.clip(logp, 0.0, 1.0)

    return val
