import pandas as pd
import numpy as np


class BatchGenerator(object):

    def __init__(self, name, config, tokenizer):
        self.name = name

        if name == 'train':
            self.data = pd.read_csv(config.train_data_path)
        elif name == 'dev':
            self.data = pd.read_csv(config.dev_data_path)
        else:
            self.data = pd.read_csv(config.test_data_path)

        self.tokenizer = tokenizer
        self.max_len = config.max_len
        self.batch_size = config.batch_size
        self.iteration = int(len(self.data) / self.batch_size)
        self.l2i = self.label2ids(self.data['Emotion'])

        self.X = np.asarray([self.__getitem__(idx)[0] for idx in range(len(self.data))])
        self.y = np.asarray([self.__getitem__(idx)[1] for idx in range(len(self.data))])
        self.y_onehot = self.onehot(config)

    def onehot(self, config):
        return np.eye(config.num_classes)[self.y.reshape(-1)]

    @staticmethod
    def label2ids(labels):
        unique_labels = set(list(labels))
        return {l: i for i, l in enumerate(unique_labels)}

    def pad(self, sample):
        diff = self.max_len - len(sample)
        if diff > 0:
            sample = sample + [0 for _ in range(diff)]
        else:
            sample = sample[:self.max_len]
        return sample

    def get_batch(self):
        batch_idxs = np.random.choice(range(len(self.data)), self.batch_size)
        X_batch, y_batch = self.X[batch_idxs], self.y_onehot[batch_idxs]
        return X_batch, y_batch

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sentence = self.tokenizer.tokenize(self.data.iloc[idx]['Sentence'], to_ids=True)
        sentence = self.pad(sentence)
        label = self.l2i[self.data.iloc[idx]['Emotion']]
        return sentence, label

    def get_max_len(self):
        total_len = []
        for idx in range(len(self.data)):
            sentence = self.tokenizer.tokenize(self.data.iloc[idx]['Sentence'], to_ids=True)
            total_len.append(len(sentence))
        return np.max(total_len)
