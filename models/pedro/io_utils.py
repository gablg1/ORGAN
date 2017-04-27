import os
import csv
import numpy as np
import itertools as it
import tensorflow as tf

# Assumes smiles is in column 0
def read_sentences_csv(filename):
    with open(filename, 'rU') as file:
        reader = csv.reader(file)
        sentences_idx = next(reader).index("sentence")
        data = [row[sentences_idx] for row in reader]
    return data

def read_songs_txt(filename):
    with open(filename, 'rU') as file:
        data = []
        line = file.readline()
        ignore_chars = {'X', 'T', '%', 'S', 'M', 'K', 'P', '\"', '\n', ' ', '(', ')', 'm', '-', '\\', '!', 't', 'r', 'i', 'l', 'z', '[', '+', 'n', 'o', '#'}
        notes = {'C', 'D', 'E', 'F', 'G', 'A', 'B', 'c', 'd', 'e', 'f', 'g', 'a', 'b'}
        prev_string = ""
        same_note = False
        on_bar = False
        while line != '':
            line = file.readline()
            if not line or line[0] in ignore_chars:
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
                        data.append(prev_string)
                        prev_string = c
                    elif length != 0:
                        first = prev_string[length - 1]
                        if first in notes:
                            data.append(prev_string)
                            prev_string = c
                        elif first == '/':
                            prev_string = prev_string[1:]

                    else:
                        prev_string += c
                # flats, sharps, and naturals - indicate a new note
                elif c in {'_', '^', '='}:
                    if not same_note:
                        same_note = True
                    data.append(prev_string)
                    prev_string = c
                # likely to indicate length
                elif c.isdigit():
                    # if in the same note, the num closes it up by indicating length - indicate end of note
                    if same_note:
                        data.append(prev_string + c)
                        prev_string = ""
                        same_note = False
                # bars & repeats are separate from notes
                elif c in {'|', ':'}:
                    if same_note:
                        data.append(prev_string)
                        prev_string = ""
                    if on_bar:
                        data.append(prev_string + c) 
                        prev_string = ""
                        on_bar = False
                    else:
                        on_bar = True
                        data.append(prev_string)
                        prev_string = c
                    same_note = False
                else:
                    prev_string += c
    return data

def read_smiles_smi(filename):
    with open(filename) as file:
        return file.readlines()

def load_data(filename, sizes, input_name, target_name):
    slices = []
    start = 0
    for size in sizes:
        stop = start + size
        slices.append(slice(start, stop))
        start = stop
    return load_data_slices_nolist(filename, slices, input_name, target_name)

def get_output_file(rel_path):
    return os.path.join(output_dir(), rel_path)

def get_data_file(rel_path):
    return os.path.join(data_dir(), rel_path)

def output_dir():
    return os.path.expanduser(safe_get("OUTPUT_DIR"))
