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
        ignore_chars = {'T', '%', 'S', 'M', 'K', 'P'}
        accumulator = ""
        while line != '':
            line = file.readline()
            if not line:
                continue
            if line[0] == 'X':
                data.append(accumulator.strip('\n'))
                accumulator = ""
            if line[0] in ignore_chars:
                continue
            else:
                accumulator += line
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
