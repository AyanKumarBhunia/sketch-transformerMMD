import numpy as np
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class sketchRNN_Data(object):
    def __init__(self, hp):
        dataset = np.load(hp.data_set, encoding='latin1', allow_pickle=True)
        self.hp = hp
        data_train = dataset['train']
        data_valid = dataset['valid']
        hp.max_seq_len = self.max_size(np.concatenate((data_train, data_valid)))
        self.hp.max_seq_len = hp.max_seq_len
        data_train = self.purify(data_train)
        self.data_train = self.normalize(data_train)

        data_valid = self.purify(data_valid)
        self.data_valid = self.normalize(data_valid)


    def purify(self, strokes):
        """removes to small or too long sequences + removes large gaps"""
        data = []
        for seq in strokes:
            if seq.shape[0] <= self.hp.max_seq_len and seq.shape[0] > 10:
                seq = np.minimum(seq, 1000)
                seq = np.maximum(seq, -1000)
                seq = np.array(seq, dtype=np.float32)
                data.append(seq)
        return data

    def max_size(self, data):
        """larger sequence length in the data set"""
        sizes = [len(seq) for seq in data]
        return max(sizes)


    def calculate_normalizing_scale_factor(self, strokes):
        """Calculate the normalizing factor explained in appendix of sketch-rnn."""
        data = []
        for i in range(len(strokes)):
            for j in range(len(strokes[i])):
                data.append(strokes[i][j, 0])
                data.append(strokes[i][j, 1])
        data = np.array(data)
        return np.std(data)

    def normalize(self, strokes):
        """Normalize entire dataset (delta_x, delta_y) by the scaling factor."""
        data = []
        scale_factor = self.calculate_normalizing_scale_factor(strokes)
        for seq in strokes:
            seq[:, 0:2] /= scale_factor
            data.append(seq)
        return data

    def train_batch(self, batch_size=100):
        batch_idx = np.random.choice(len(self.data_train), batch_size)
        batch_sequences = [self.data_train[idx] for idx in batch_idx]
        strokes = []
        lengths = []
        indice = 0
        for seq in batch_sequences:
            len_seq = len(seq[:, 0])
            new_seq = np.zeros((self.hp.max_seq_len, 5))
            new_seq[:len_seq, :2] = seq[:, :2]
            new_seq[:len_seq - 1, 2] = 1 - seq[:-1, 2]
            new_seq[:len_seq, 3] = seq[:, 2]
            new_seq[(len_seq - 1):, 4] = 1
            new_seq[len_seq - 1, 2:4] = 0
            lengths.append(len(seq[:, 0]))
            strokes.append(new_seq)
            indice += 1

        batch = torch.from_numpy(np.stack(strokes, 1)).to(device).float()
        return batch, torch.tensor(lengths).type(torch.int64).to(device)


    def valid_batch(self, batch_size=100):
        batch_idx = np.random.choice(len(self.data_valid), batch_size)
        batch_sequences = [self.data_valid[idx] for idx in batch_idx]
        strokes = []
        lengths = []
        indice = 0
        for seq in batch_sequences:
            len_seq = len(seq[:, 0])
            new_seq = np.zeros((self.hp.max_seq_len, 5))
            new_seq[:len_seq, :2] = seq[:, :2]
            new_seq[:len_seq - 1, 2] = 1 - seq[:-1, 2]
            new_seq[:len_seq, 3] = seq[:, 2]
            new_seq[(len_seq - 1):, 4] = 1
            new_seq[len_seq - 1, 2:4] = 0
            lengths.append(len(seq[:, 0]))
            strokes.append(new_seq)
            indice += 1

        batch = torch.from_numpy(np.stack(strokes, 1)).to(device).float()
        return batch, torch.tensor(lengths).type(torch.int64).to(device)





