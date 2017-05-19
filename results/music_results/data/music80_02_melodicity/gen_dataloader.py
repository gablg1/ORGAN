import numpy as np


class Gen_Data_loader():
    def __init__(self, batch_size):
        self.batch_size = batch_size

    def create_batches(self, samples):
        self.num_batch = int(len(samples) / self.batch_size)
        samples = samples[:self.num_batch * self.batch_size]
        self.sequence_batch = np.split(np.array(samples), self.num_batch, 0)
        self.pointer = 0

    def next_batch(self):
        ret = self.sequence_batch[self.pointer]
        self.pointer = (self.pointer + 1) % self.num_batch
        return ret

    def reset_pointer(self):
        self.pointer = 0
