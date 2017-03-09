import numpy as np
from re import compile as _Re
import cPickle


def split_unicode_chrs(text):
    _unicode_chr_splitter = _Re('(?s)((?:[\ud800-\udbff][\udc00-\udfff])|.)').split
    return [chr for chr in _unicode_chr_splitter(text) if chr]


class Dis_dataloader():
    def load_train_data(self, sentences, labels):
        """
        Returns input vectors, labels, vocabulary, and inverse vocabulary.
        """
        # Load and preprocess data
        shuffle_indices = np.random.permutation(np.arange(len(labels)))
        x_shuffled = sentences[shuffle_indices]
        y_shuffled = labels[shuffle_indices]
        return [x_shuffled, y_shuffled]

    def load_test_data(self, positive_file, test_file):
        test_examples = []
        test_labels = []
        with open(test_file)as fin:
            for line in fin:
                line = line.strip()
                line = line.split()
                parse_line = [int(x) for x in line]
                test_examples.append(parse_line)
                test_labels.append([1, 0])

        with open(positive_file)as fin:
            for line in fin:
                line = line.strip()
                line = line.split()
                parse_line = [int(x) for x in line]
                test_examples.append(parse_line)
                test_labels.append([0, 1])

        test_examples = np.array(test_examples)
        test_labels = np.array(test_labels)
        shuffle_indices = np.random.permutation(np.arange(len(test_labels)))
        x_dev = test_examples[shuffle_indices]
        y_dev = test_labels[shuffle_indices]

        return [x_dev, y_dev]

    def batch_iter(self, data, batch_size, num_epochs):
        """
        Generates a batch iterator for a dataset.
        """
        data = np.array(data)
        data_size = len(data)
        num_batches_per_epoch = int(len(data) / batch_size) + 1
        for epoch in range(num_epochs):
            # Shuffle the data at each epoch
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
            for batch_num in range(num_batches_per_epoch):
                start_index = batch_num * batch_size
                end_index = min((batch_num + 1) * batch_size, data_size)
                yield shuffled_data[start_index:end_index]
