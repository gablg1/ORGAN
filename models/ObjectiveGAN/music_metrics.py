import os
import model
import numpy as np
import tensorflow as tf
import random
import time
from gen_dataloader import Gen_Data_loader
from dis_dataloader import Dis_dataloader
from rollout import ROLLOUT
from target_lstm import TARGET_LSTM
import cPickle

def pct(a, b):
    if len(b) == 0:
        return 0
    return float(len(a)) / len(b)


def build_vocab(sequences, pad_char = '_', start_char = '^'):
    i = 1
    char_dict, ord_dict = {start_char: 0}, {0: start_char}
    for sequence in sequences:
        for c in sequence:
            if c not in char_dict:
                char_dict[c] = i
                ord_dict[i] = c
                i += 1
    char_dict[pad_char], ord_dict[i] = i, pad_char
    return char_dict, ord_dict

char_dict, ord_dict = build_vocab(sequences)

def pad(sequence, n, pad_char = '_'):
    if n < len(sequence):
        return sequence
    return sequence + [pad_char] * (n - len(sequence))

def unpad(sequence, pad_char = '_'):
    def reverse(s): return s[::-1]
    rev = reverse(sequence)
    for i, elem in enumerate(rev):
        if elem != pad_char:
            return reverse(rev[i:])
    return sequence

def encode_smile(sequence, max_len): return [char_dict[c] for c in pad(sequence, max_len)]
def decode_smile(ords): return ' '.join(unpad([ord_dict[o] for o in ords]))

notes = ['C,', 'D,', 'E,', 'F,', 'G,', 'A,', 'B,', 'C', 'D', 'E', 'F', 'G', 'A', 'B',
    'c', 'd', 'e', 'f', 'g', 'a', 'b', 'c\'', 'd\'', 'e\'', 'f\'', 'g\'', 'a\'', 'b\'']

notes_and_frequencies = {'C,' : 65.41, 'D,' : 73.42, 'E,' : 82.41, 'F,' : 87.31, 'G,' : 98, 'A,' : 110, 'B,' : 123.47, 
    'C' : 130.81, 'D' : 146.83, 'E' : 164.81, 'F' : 174.61, 'G' : 196, 'A' : 220, 'B' : 246.94,
    'c' : 261.63, 'd' : 293.66, 'e' : 329.63, 'f' : 349.23, 'g' : 392, 'a' : 440, 'b' : 493.88,
    'c\'' : 523.25, 'd\'' : 587.33, 'e\'' : 659.25, 'f\'' : 698.46, 'g\'' : 783.99, 'a\'' : 880, 'b\'' : 987.77}

def is_note(note): return note in notes

def is_valid_sequence(sequence):
    clean_sequence = clean(sequence)
    return np.sum([(1 if is_note(note) else 0) for note in clean_sequence]) > 1 if len(sequence) != 0 else False

def clean(sequence): 
    return [note.strip("_^=\\0123456789") for note in sequence if is_note(note.strip("_^=\\0123456789"))]

def notes_and_successors(sequence): return [(note, sequence[i+1]) for i, note in enumerate(sequence) if i < len(sequence) - 1]

def is_perf_fifth(note, succ): 
        ratio = notes_and_frequencies[succ] / notes_and_frequencies[note]
        return ratio < 1.55 and ratio > 1.45

def batch_tonality(sequence):
    vals = [tonality(sequence) if is_valid_sequence(sequence)
    else 0 for note in sequence]
    return np.mean(vals)

def tonality(sequence):
    clean_sequence = clean(sequence)

    notes_and_succs = notes_and_successors(clean_sequence)

    return np.mean([(1 if is_perf_fifth(note, successor) else 0) for note, successor in notes_and_succs]) if len(sequence) > 1 else 0

def batch_melodicty(sequence):
    vals = [melodicty(sequence) if is_valid_sequence(sequence)
    else 0 for note in sequence]
    return np.mean(vals)

# Order of dissonance (best to worst): P5, P4, M6, M3, m3, m6, M2, m7, m2, M7, TT
# To be melodic, it must be a M6 or better
def melodicity(sequence):
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

def batch_ratio_of_steps(sequence):
    vals = [ratio_of_steps(sequence) if is_valid_sequence(sequence)
    else 0 for note in sequence]
    return np.mean(vals)

def ratio_of_steps(sequence):
    clean_sequence = clean(sequence)

    notes_and_succs = notes_and_successors(clean_sequence)

    def is_step(note, succ): return abs(notes.index(note) - notes.index(succ)) == 1

    return np.mean([(1 if is_step(note, successor) else 0) for note, successor in notes_and_succs]) if len(sequence) > 1 else 0

def read_songs_txt(filename):
    with open(filename, 'rU') as file:
        data = []
        line = file.readline()
        ignore_chars = {'T', '%', 'S', 'M', 'K', 'P', 'L', '\"', '\n', ' ', '(', ')', 'm', '-', '\\', '!', 't', 'r', 'i', 'l', 'z', '[', '+', 'n', 'o', '#'}
        notes = {'C', 'D', 'E', 'F', 'G', 'A', 'B', 'c', 'd', 'e', 'f', 'g', 'a', 'b'}
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
                    # if in the same note, the num closes it up by indicating length - indicate end of note
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

def print_music(model_samples, train_smiles):
    samples = [decode_smile(s) for s in model_samples]
    unique_samples = list(set(samples))
    print 'Unique samples. Pct: {}'.format(pct(unique_samples, samples))
    verified_samples = filter(verify_sequence, samples)

    for s in samples[0:10]:
        print s
    print 'Verified samples. Pct: {}'.format(pct(verified_samples, samples))
    for s in verified_samples[0:10]:
        print s
    print 'Objective: {}'.format(objective(samples))