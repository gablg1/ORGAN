import editdistance
from glob import glob
import numpy as np

files = glob('data/*/epoch_data/*199.abc')
#files = glob('data/*.abc')

def levenshtein(s1, s2):
    return float(editdistance.eval(s1, s2)) / 80

for path in files:
    with open(path, 'r') as fp:
        def clean(seq): return [c for c in seq if c != '']
        seqs = [clean(seq.strip().split(' ')) for seq in fp.readlines()]
        seqs = seqs[:2000]
        print len(seqs)
        levenshtein_distances = [levenshtein(x, y) for x in seqs for y in seqs]
        print path
        print 'Levenshtein distance: {}'.format(np.mean(levenshtein_distances))






