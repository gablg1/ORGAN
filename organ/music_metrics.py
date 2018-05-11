import os
import pickle
import numpy as np
import editdistance
from itertools import product

notes = list( range(48) )
note_frequencies = {
                     note: 440 * 2**( (46+m-69)/12 )
                     for m, note in enumerate(notes)
                   }

def build_vocab(sequences, start_char=-1):
    i = 1
    char_dict, ord_dict = {start_char: 0}, {0: start_char}
    for sequence in sequences:
        for c in sequence:
            if c not in char_dict.keys():
                char_dict[c] = i
                ord_dict[i] = c
                i += 1
    return char_dict, ord_dict

def pad(sequence, n, pad_char='__'):
    return sequence

def unpad(sequence, pad_char='__'):
    return sequence

def encode(sequence, max_len, char_dict):
    return [char_dict[c] for c in sequence]

def decode(ords, ord_dict):
    return [ord_dict[o] for o in ords]

def is_note(note):
    return note in notes

def verify_sequence(sequence):
    if sequence[0] == '1': 
        return False
    for note in sequence:
        if note not in notes: 
            return False
    if '^^' in notes: return False
    try:
        nas = notes_and_successors(sequence)
    except:
        return False
    return True

def editdistance(m1, m2):
    return float(editdistance.eval(m1, m2)) / max(len(m1), len(m2))

def notes_and_successors(sequence):

    processed = []
    current_note = None
    for note in sequence:
        if note == 1:
            if current_note is None:
                raise ValueError('Incorrect sequence')
            else:
                processed.append(current_note)
        else:
            current_note = note
            processed.append(current_note)

    return [(note, processed[i+1]) for i, note in enumerate(processed[:-1])]

# Order of dissonance (best to worst): P5, P4, M6, M3, m3, m6, M2, m7, m2, M7, TT
# To be melodic, it must be a M6 or better

def is_perf_fifth(note, succ):
    ratio = note_frequencies[succ] / note_frequencies[note]
    return 1.45 < ratio < 1.55

def is_perf_fourth(note, succ):
    ratio = note_frequencies[succ] / note_frequencies[note]
    return 1.28 < ratio < 1.38

def is_major_sixth(note, succ):
    ratio = note_frequencies[succ] / note_frequencies[note]
    return 1.62 < ratio < 1.72

def is_harmonic(note, succ):
    ratio = note_frequencies[succ] / note_frequencies[note]
    return is_perf_fifth(note, succ) or is_perf_fourth(note, succ) or is_major_sixth(note, succ)

def melodicity(sequence, train_data):
    if not verify_sequence(sequence):
        return 0.0
    notes_and_succs = notes_and_successors(sequence)
    try:
        score = np.mean([(1 if is_harmonic(note, successor) else 0.0) for note, successor in notes_and_succs]) if len(seq) > 1 else 0.0
        return score
    except:
        return 0.0

def tonality(sequence, train_data):
    seq = unpad(sequence)
    if not verify_sequence(seq):
        return 0.0
    notes_and_succs = notes_and_successors(seq)
    try:
        score = np.mean([(1 if is_perf_fifth(note, successor) else 0.0) for note, successor in notes_and_succs]) if len(seq) > 1 else 0.0
        return score
    except:
        return 0.0

def is_step(note, succ):
    return abs(notes.index(note) - notes.index(succ)) == 1

def ratio_of_steps(sequence, train_data):
    seq = unpad(sequence)
    if not verify_sequence(seq):
        return 0.0
    notes_and_succs = notes_and_successors(seq)
    try:
        score = np.mean([(1 if is_step(note, successor) else 0) for note, successor in notes_and_succs]) if len(seq) > 1 else 0
        return score
    except:
        return 0.0

def load_train_data(filename):
    with open(filename, 'rb') as afile:
        sequences = pickle.load(afile)
    return sequences

def print_params(p):
    print('Using parameters:')
    for key, value in p.items():
        print('{:20s} - {:12}'.format(key, value))
    print('rest of parameters are set as default\n')
    return


def verified_and_below(seq, max_len):
    return len(seq) <= max_len and verify_sequence(seq)


def save_pkl(name, sequences):
    folder = os.path.join('/n/home15/bsanchezlengeling/couteiral/music/ORGAN/',
			  'epoch_data')
    if not os.path.exists(folder):
        os.makedirs(folder)
    smi_file = os.path.join(folder, name + ".pkl")
    with open(smi_file, 'wb') as afile:
        pickle.dump(sequences, afile)
    return

def uniq_samples(samples):
    seqs    = [[str(x) for x in y] for y in samples]
    strings = [''.join(seq) for seq in seqs]
    return len(set(strings))

def compute_results(reward, model_samples, train_samples, ord_dict, results={}, verbose=True):
    samples = [decode(s, ord_dict) for s in model_samples]
    results['mean_length'] = np.mean([len(sample) for sample in samples])
    results['n_samples'] = len(samples)
    results['uniq_samples'] = uniq_samples(samples)
    verified_samples = [
        sample for sample in samples if verify_sequence(sample)]
    unverified_samples = [
        sample for sample in samples if not verify_sequence(sample)]
    results['good_samples'] = len(verified_samples)
    results['bad_samples'] = len(unverified_samples)
    abc_name = '{}_{}'.format(results['exp_name'], results['Batch'])
    save_pkl(abc_name, samples)
    results['model_samples'] = abc_name
    return

def print_results(samples, metrics, results={}):
    print('~~~ Summary Results ~~~')
    print('Total samples: {}'.format(results['n_samples']))
    percent = results['uniq_samples'] / float(results['n_samples']) * 100
    print('Unique      : {:6d} ({:2.2f}%)'.format(
        results['uniq_samples'], percent))
    for i in metrics:
        print('{:11s} : {:1.4f}'.format(i, results[i]))

    print('\nSamples:')
    for s in samples[0:10]:
        print('' + s)

    return

def remap(x, x_min, x_max):
    if x_max == 0.0 or x_max == x_min: return 0.0
    return (x - x_min) / (x_max - x_min)

def batch_melodicity(sequences, train_samples=None):
    vals = [melodicity(seq, None) if verify_sequence(seq) else 0.0
            for seq in sequences]
    rvals = [remap(x, np.min(vals), np.max(vals)) for x in vals]
    return rvals

def batch_tonality(sequences, train_samples=None):
    vals = [tonality(seq, None) if verify_sequence(seq) else 0.0
            for seq in sequences]
    rvals = [remap(x, np.min(vals), np.max(vals)) for x in vals]
    return rvals

def batch_ratio_of_steps(sequences, train_samples=None):
    vals = [ratio_of_steps(seq, None) if verify_sequence(seq) else 0.0
            for seq in sequences]
    rvals = [remap(x, np.min(vals), np.max(vals)) for x in vals]
    return rvals

def metrics_loading():

    loading = {}
    loading['melodicity'] = lambda *args: None
    loading['tonality'] = lambda *args: None
    loading['ratio_of_steps'] = lambda *args: None
    return loading

def get_metrics():

    objective = {}
    objective['melodicity'] = batch_melodicity
    objective['tonality'] = batch_tonality
    objective['ratio_of_steps'] = batch_ratio_of_steps
    return objective
