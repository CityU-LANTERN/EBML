import json
import os
import sys
from PIL import Image
from torch.utils.data import Dataset, DataLoader, sampler
import tqdm
import concurrent.futures
import numpy as np
import pickle
import torch
from src.datasets.drug import preprocess
import copy

class DrugDataLoader(object):
    def __init__(self, configs):
        file = open(f'src/datasets/drug/{configs.dataset.name}/reg_inputs.pickle', 'rb')
        data = pickle.load(file)
        self.train_x, self.test_iid_x, self.test_ood_x = data['train'], data['test_iid'], data['test_ood']
        file.close()
        file = open(f'src/datasets/drug/{configs.dataset.name}/reg_labels.pickle', 'rb')
        data = pickle.load(file)
        self.train_y, self.test_iid_y, self.test_ood_y = data['train'], data['test_iid'], data['test_ood']
        file.close()
        self.batch_size = configs.train.meta_batch_size
        self.current_iteration = 0
        self.val_test_seed = configs.seed
        self.vary_max_shot = True

    def get_train_batches(self, n_batches=None):
        rng = np.random.RandomState(seed=self.current_iteration)
        task_idx = np.arange(len(self.train_x))
        rng.shuffle(task_idx)
        train_batches = []
        if not n_batches:
            n_batches = len(self.train_x) // self.batch_size
        else:
            task_idx = rng.choice(task_idx, n_batches * self.batch_size, replace=True)
        for i in range(n_batches):
            task_idx_in_batch = task_idx[i * self.batch_size:(i + 1) * self.batch_size]
            batch = []
            for idx in task_idx_in_batch:
                x = torch.tensor(self.train_x[idx])
                y = torch.tensor(self.train_y[idx])
                sample_idx = np.arange(len(x))
                rng.shuffle(sample_idx)
                n_support = len(sample_idx) // 2
                max_support = rng.randint(10, 50) if self.vary_max_shot else 50
                n_support = min(n_support, max_support)
                support_idx = sample_idx[:n_support]
                query_idx = sample_idx[n_support:n_support + 50]
                # print(x.shape,type(x))
                batch.append([x[support_idx], y[support_idx], x[query_idx], y[query_idx]])
            train_batches.append(batch)
        self.current_iteration += 1
        return train_batches

    def get_val_batches(self):
        rng = np.random.RandomState(seed=self.val_test_seed)
        task_idx = np.arange(len(self.train_x))
        val_tasks = []
        task_idx = rng.choice(task_idx, 50, replace=True)
        for idx in task_idx:
            x = torch.tensor(self.train_x[idx])
            y = torch.tensor(self.train_y[idx])
            sample_idx = np.arange(len(x))
            rng.shuffle(sample_idx)
            n_support = len(sample_idx) // 2
            max_support = rng.randint(10, 50) if self.vary_max_shot else 50
            n_support = min(n_support, max_support)
            support_idx = sample_idx[:n_support]
            query_idx = sample_idx[n_support:n_support + 50]
            val_tasks.append([x[support_idx], y[support_idx], x[query_idx], y[query_idx]])
        return val_tasks

    def get_test_tasks(self, split, n_tasks=None):
        assert split in ['ID', 'OOD'], f'unknown split {split}, must be either ID or OOD'
        rng = np.random.RandomState(seed=self.val_test_seed)
        split_x = self.test_iid_x if split == 'ID' else self.test_ood_x
        split_y = self.test_iid_y if split == 'ID' else self.test_ood_y
        task_idx = np.arange(len(split_x))
        rng.shuffle(task_idx)
        test_tasks = []
        if n_tasks:
            task_idx = rng.choice(task_idx, n_tasks, replace=True)
        for idx in task_idx:
            x = torch.tensor(split_x[idx])
            y = torch.tensor(split_y[idx])
            if len(x) <= 3:
                continue
            sample_idx = np.arange(len(x))
            rng.shuffle(sample_idx)
            n_support = len(sample_idx) // 2
            max_support = rng.randint(10, 50) if self.vary_max_shot else 50
            n_support = min(n_support, max_support)
            support_idx = sample_idx[:n_support]
            query_idx = sample_idx[n_support:n_support + 50]
            test_tasks.append([x[support_idx], y[support_idx], x[query_idx], y[query_idx]])
        return test_tasks