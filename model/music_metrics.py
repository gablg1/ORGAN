import os
import numpy as np
<<<<<<< HEAD
import tensorflow as tf
import random
import time
from gen_dataloader import Gen_Data_loader
from dis_dataloader import Dis_dataloader
from rollout import ROLLOUT
from target_lstm import TARGET_LSTM
import cPickle
import editdistance
=======

>>>>>>> 5c5fe367a18f2bf84ab3e5de233838343a550d48

def pct(a, b):
    if len(b) == 0:
        return 0
    return float(len(a)) / len(b)


def build_vocab(sequences, pad_char='_', start_char='^'):
    i = 1
    char_dict, ord_dict = {start_char: 0}, {0: start_char}
    for sequence in sequences:
        for c in sequence:
            if c not in char_dict:
                char_dict[c] = i
                ord_dict[i] = c
                i += 1
    char_dict[pad_char], ord_dict[i] = i, pad_char

    # FIXME: For some reason the tensorflow embedding size expects a number of
    # tokens one more
    char_dict['hi'], ord_dict[i + 1] = i + 1, 'hi'
    return char_dict, ord_dict


def pad(sequence, n, pad_char='_'):
    if n < len(sequence):
        return sequence
    return sequence + [pad_char] * (n - len(sequence))


def unpad(sequence, pad_char='_'):
    def reverse(s): return s[::-1]
    rev = reverse(sequence)
    for i, elem in enumerate(rev):
        if elem != pad_char:
            return reverse(rev[i:])
    return sequence


def encode(sequence, max_len, char_dict): return [
    char_dict[c] for c in pad(sequence, max_len)]


def decode(ords, ord_dict): return ' '.join(unpad([ord_dict[o] for o in ords]))

notes = ['C,', 'D,', 'E,', 'F,', 'G,', 'A,', 'B,', 'C', 'D', 'E', 'F', 'G', 'A', 'B',
         'c', 'd', 'e', 'f', 'g', 'a', 'b', 'c\'', 'd\'', 'e\'', 'f\'', 'g\'', 'a\'', 'b\'']

notes_and_frequencies = {'C,': 65.41, 'D,': 73.42, 'E,': 82.41, 'F,': 87.31, 'G,': 98, 'A,': 110, 'B,': 123.47,
                         'C': 130.81, 'D': 146.83, 'E': 164.81, 'F': 174.61, 'G': 196, 'A': 220, 'B': 246.94,
                         'c': 261.63, 'd': 293.66, 'e': 329.63, 'f': 349.23, 'g': 392, 'a': 440, 'b': 493.88,
                         'c\'': 523.25, 'd\'': 587.33, 'e\'': 659.25, 'f\'': 698.46, 'g\'': 783.99, 'a\'': 880, 'b\'': 987.77}


def is_note(note): return note in notes


def verify_sequence(sequence):
    clean_sequence = clean(sequence)
    return np.sum([(1 if is_note(note) else 0) for note in clean_sequence]) > 1 if len(sequence) != 0 else False


def clean(sequence):
    return [note.strip("_^=\\0123456789") for note in sequence if is_note(note.strip("_^=\\0123456789"))]

def sequence_to_clean_string(sequence):
    s = ""
    for c in clean(sequence):
        s += c
    return s

def editdistance(m1, m2):
    return float(editdistance.eval(m1, m2)) / max(len(m1), len(m2))

def notes_and_successors(sequence): return [(note, sequence[i+1]) for i, note in enumerate(sequence) if i < len(sequence) - 1]

def notes_and_successors(sequence): return [(note, sequence[
    i + 1]) for i, note in enumerate(sequence) if i < len(sequence) - 1]


def is_perf_fifth(note, succ):
    ratio = notes_and_frequencies[succ] / notes_and_frequencies[note]
    return ratio < 1.55 and ratio > 1.45


def tonality(sequence, train_data):
    if not verify_sequence(sequence):
        return 0
    clean_sequence = clean(sequence)

    notes_and_succs = notes_and_successors(clean_sequence)

    return np.mean([(1 if is_perf_fifth(note, successor) else 0) for note, successor in notes_and_succs]) if len(sequence) > 1 else 0


def batch(fn):
    return lambda seqs, data: np.mean([fn(seq, data) for seq in seqs])

# Order of dissonance (best to worst): P5, P4, M6, M3, m3, m6, M2, m7, m2, M7, TT
# To be melodic, it must be a M6 or better


def melodicity(sequence, train_data):
    if not verify_sequence(sequence):
        return 0
    clean_sequence = clean(sequence)

    notes_and_succs = notes_and_successors(clean_sequence)

    def is_perf_fourth(note, succ):
        ratio = notes_and_frequencies[succ] / notes_and_frequencies[note]
        return ratio < 1.38 and ratio > 1.28

    def is_major_sixth(note, succ):
        ratio = notes_and_frequencies[succ] / notes_and_frequencies[note]
        return ratio < 1.72 and ratio > 1.62

    def is_harmonic(note, succ):
        ratio = notes_and_frequencies[succ] / notes_and_frequencies[note]
        return is_perf_fifth(note, succ) or is_perf_fourth(note, succ) or is_major_sixth(note, succ)

    return np.mean([(1 if is_harmonic(note, successor) else 0) for note, successor in notes_and_succs]) if len(sequence) > 1 else 0


def ratio_of_steps(sequence, train_data):
    if not verify_sequence(sequence):
        return 0
    clean_sequence = clean(sequence)

    notes_and_succs = notes_and_successors(clean_sequence)

    def is_step(note, succ): return abs(
        notes.index(note) - notes.index(succ)) == 1

    return np.mean([(1 if is_step(note, successor) else 0) for note, successor in notes_and_succs]) if len(sequence) > 1 else 0


def load_train_data(filename):
    with open(filename, 'rU') as file:
        data = []
        line = file.readline()
        ignore_chars = {'T', '%', 'S', 'M', 'K', 'P', 'L', '\"', '\n', ' ',
                        '(', ')', 'm', '-', '\\', '!', 't', 'r', 'i', 'l', 'z', '[', '+', 'n', 'o', '#'}
        notes = {'C', 'D', 'E', 'F', 'G', 'A',
                 'B', 'c', 'd', 'e', 'f', 'g', 'a', 'b'}
        prev_string = ""
        same_note = False
        on_bar = False
        song = []
        while line != '':
            line = file.readline()
            if not line or line[0] in ignore_chars:
                continue
            # new song
            if line[0] == 'X':
                data.append(song)
                song = []
                continue
            for c in line:
                if c in ignore_chars:
                    continue
                elif c in notes:
                    same_note = True
                    length = len(prev_string)
                    # previous string was a note
                    if on_bar:
                        on_bar = False
                        song.append(prev_string)
                        prev_string = c
                    elif length != 0:
                        first = prev_string[length - 1]
                        if first in notes:
                            song.append(prev_string)
                            prev_string = c
                        elif first == '/':
                            prev_string = prev_string[1:]

                    else:
                        prev_string += c
                # flats, sharps, and naturals - indicate a new note
                elif c in {'_', '^', '='}:
                    if not same_note:
                        same_note = True
                    song.append(prev_string)
                    prev_string = c
                # likely to indicate length
                elif c.isdigit():
                    # if in the same note, the num closes it up by indicating
                    # length - indicate end of note
                    if same_note:
                        song.append(prev_string + c)
                        prev_string = ""
                        same_note = False
                # bars & repeats are separate from notes
                elif c in {'|', ':'}:
                    if same_note:
                        song.append(prev_string)
                        prev_string = ""
                    if on_bar:
                        song.append(prev_string + c)
                        prev_string = ""
                        on_bar = False
                    else:
                        on_bar = True
                        song.append(prev_string)
                        prev_string = c
                    same_note = False
                else:
                    prev_string += c
    return data


def load_reward(objective):

    if objective == 'melodicity':
        return melodicity
    elif objective == 'tonality':
        return tonality
    elif objective == 'ratio_of_steps':
        return ratio_of_steps
    else:
        raise ValueError('objective not found!')
    return


def print_params(p):
    print('Using parameters:')
    for key, value in p.items():
        print('{:20s} - {:12}'.format(key, value))
    print('rest of parameters are set as default\n')
    return


def verified_and_below(seq, max_len):
    return len(seq) < max_len and verify_sequence(seq)


def save_abc(name, smiles):
    folder = 'epoch_data'
    if not os.path.exists(folder):
        os.makedirs(folder)
    smi_file = os.path.join(folder, name + ".abc")
    with open(smi_file, 'w') as afile:
        afile.write('\n'.join(smiles))
    return


def compute_results(model_samples, train_samples, ord_dict, results={}, verbose=True):
    samples = [decode(s, ord_dict) for s in model_samples]
    results['mean_length'] = np.mean([len(sample) for sample in samples])
    results['n_samples'] = len(samples)
    results['uniq_samples'] = len(set(samples))
    metrics = {'ratio_of_steps': batch(ratio_of_steps),
               'melodicity': batch(melodicity),
               'tonality': batch(tonality)}
    for key, func in metrics.items():
        results[key] = np.mean(func(samples, train_samples))

    if 'Batch' in results.keys():
        file_name = '{}_{}'.format(results['exp_name'], results['Batch'])
        save_abc(file_name, samples)
        results['model_samples'] = file_name
    if verbose:
        print_results(samples, metrics.keys(), results)
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
