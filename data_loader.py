import torch.utils.data as data
from PIL import Image
import os
import pickle as dill
import numpy as np
import torch
from torch.utils.data import TensorDataset

class GetDataset():
    def __init__(self, data_root, unseen_index, val_split):

        with open(os.path.join(data_root, 'af_normal_data_processed.pkl'), 'rb') as file:
            data = dill.load(file)

        datasets = ['CSPC_data', 'PTB_XL_data', 'G12EC_data', 'Challenge2017_data']

        test_data = []
        train_datas = []
        val_datas = []

        for source in datasets:
            af_data, normal_data = data[source]
            all_data = np.concatenate((af_data, normal_data), axis=0)
            all_label = np.zeros((len(all_data),))
            all_label[len(af_data):] = 1

            # use all data of this source as test data
            permuted_idx = np.random.permutation(len(all_data))
            x = all_data[permuted_idx]
            y = all_label[permuted_idx]

            split_idx = int(val_split * len(all_data))
            x_val = all_data[permuted_idx[split_idx:]]
            y_val = all_label[permuted_idx[split_idx:]]
            x_train = all_data[permuted_idx[:split_idx]]
            y_train = all_label[permuted_idx[:split_idx]]

            # swap axes
            x = x.swapaxes(1, 2)
            x_train = x_train.swapaxes(1, 2)
            x_val = x_val.swapaxes(1, 2)

            test_data.append([x, y])
            train_datas.append([x_train, y_train])
            val_datas.append([x_val, y_val])

        self.train_datas = train_datas
        self.val_datas = val_datas

        a = [0, 1, 2, 3]
        a.remove(unseen_index)

        self.unseen_data = test_data[unseen_index]
        del self.train_datas[unseen_index]
        del self.val_datas[unseen_index]

        print(0)

    def get_datasets(self):

        train_datasets = []
        for train_data in self.train_datas:
            X_train, Y_train = train_data
            X_train = torch.from_numpy(X_train).float()
            Y_train = torch.from_numpy(Y_train).long()
            dataset = TensorDataset(X_train, Y_train)
            train_datasets.append(dataset)

        val_datasets = []
        for val_data in self.val_datas:
            X_val, Y_val = val_data
            X_val = torch.from_numpy(X_val).float()
            Y_val = torch.from_numpy(Y_val).long()
            dataset = TensorDataset(X_val, Y_val)
            val_datasets.append(dataset)

        X_test, Y_test = self.unseen_data
        X_test = torch.from_numpy(X_test).float()
        Y_test = torch.from_numpy(Y_test).long()
        test_dataset = TensorDataset(X_test, Y_test)

        return train_datasets, val_datasets, test_dataset








