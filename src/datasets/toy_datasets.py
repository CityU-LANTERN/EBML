import os
import numpy as np
import torch
# from sklearn.datasets import make_moons
from torch.utils.data import Dataset

#######################
'''Sinusoid Dataset '''
#######################
class SineRegression(Dataset):
    def __init__(self, configs):
        self.tasks_inuse = None
        self.supp_per_sine = configs.num_support
        self.query_per_sine = configs.num_query
        self.num_tasks = configs.num_tasks
        self.include_lines = configs.include_lines
        self._mode = 'train'
        self.set_mode('train')
        self._hash= {'train':0,'test_id':79000,'test_ood':5024,'val_id':12345}
        if configs.name == 'multi-sinusoids':
            self.multi_sine =True
            print('Using Multi-Sine for Regression')
        else:
            self.multi_sine=False

    def __len__(self):
        return self.num_tasks

    def __getitem__(self, i):
        # tasks x idx are fixed once these tasks are generated
        y = []
        if self._mode == 'train':
            # sample a new task from the distribution
            x = np.random.uniform(low=-5.0, high=5.0, size=(self.supp_per_sine + self.query_per_sine, 1))
        else:
            # used 512 fixed features points for testing
            x = np.linspace(start=-5.0, stop=5.0, num=500).reshape(-1, 1)
        if self.multi_sine:
            x = [x,x]
        else:
            x= [x]
        # generate task
        if self.include_lines and i%2:
            intersection, grad = self._generate_lines(seed=(i + self._hash[self._mode]))
            for sine_idx, x_value in enumerate(x, 1):
                y_value = grad * (x_value - intersection)
                if self._mode =='train':
                    y_value += np.random.randn(*y_value.shape) * 0.2
                y.append(y_value)
                fx = f'{grad:.3f}(x - {intersection:.3f})'
            a = intersection
            b = grad
        else:
            amp,phase,freq =self._generate_sinusoids(seed=(i + self._hash[self._mode]))
            # generate observation
            for sine_idx, x_value in enumerate(x,1):
                if (sine_idx % 2) == 1:
                    y_value = amp * np.sin(freq * (x_value + phase))
                else:
                    y_value = amp * np.sin(freq * (x_value + phase + np.pi * 0.30))
                # noisy observation
                if self._mode =='train':
                    y_value += np.random.randn(*y_value.shape) * 0.2
                y.append(y_value)
                # fx = f'{amp:.3f}Sin({freq:.3f}(x + {phase:.3f}))'
            a= amp
            b= phase

        y = np.stack(y, axis=0)
        x = np.stack(x, axis=0)
        rand_idx = np.random.RandomState(seed=(i+self._hash[self._mode])).permutation(x.shape[1])
        x = x[:,rand_idx,:].T.reshape(-1,1)
        y = y[:,rand_idx,:].T.reshape(-1,1)
        num_supp = sine_idx * self.supp_per_sine
        return (x[:num_supp], y[:num_supp]), \
               (x[num_supp:], y[num_supp:]), \
               (a,b)

    def set_mode(self, mode):
        if mode in ['train','test_id','test_ood','val_id']:
            self._mode = mode
            print(f'using {self._mode} dataset')
        else:
            raise ValueError('mode is not defined')


    def _generate_sinusoids(self,seed):
        if self._mode == 'train':
            rng = np.random.RandomState(None)
        else:
            rng = np.random.RandomState(seed)
        if self._mode == 'train': # ID train distribution, only select amp/phase at discrete values
            amp = rng.choice(np.arange(1.0,4.0,0.1),1)
            phase = rng.choice(np.arange(0.0,0.5 * np.pi,0.1),1)
        else:   # ID test/val distribution, sample amp/phase from a continuous range
            amp = rng.uniform(low=1.0, high=4.0)
            phase = rng.uniform(low=0, high=0.5 * np.pi)
        freq = np.ones_like(amp)
        # OOD distribution
        if self._mode == 'test_ood':
            ood_idx = rng.randint(low=0, high=4)
            if ood_idx == 0:
                phase = rng.uniform(low=0.6 * np.pi, high=0.75 * np.pi)
            elif ood_idx == 1:
                amp = rng.uniform(low=0.1, high=0.8)
            elif ood_idx == 2:
                amp = rng.uniform(low=4.2, high=5.0)
            elif ood_idx == 3:
                freq = rng.uniform(low=1.1, high=1.25)
        return amp, phase, freq

    def _generate_lines(self,seed):
        if self._mode == 'train':
            rng = np.random.RandomState(None)
        else:
            rng = np.random.RandomState(seed)
        # ID distribution
        if self._mode == 'train':
            intersection = rng.choice(np.arange(-3.0, 3.0, 0.1), 1)
            grad = rng.choice(np.arange(-3.0, 3.0, 0.1), 1)
        else:
            intersection = rng.uniform(low=-3.0, high=3.0)
            grad = rng.uniform(low=-3.0, high=3.0)
        # OOD distribution
        if self._mode == 'test_ood':
            ood_idx = rng.randint(low=0, high=4)
            if ood_idx == 0:
                intersection = rng.uniform(low=-5.0, high=-3.5)
            elif ood_idx == 1:
                intersection = rng.uniform(low=3.5, high=5.0)
            elif ood_idx == 2:
                grad = rng.uniform(low=-5.0, high=-3.5)
            elif ood_idx == 3:
                grad = rng.uniform(low=3.5, high=5.0)
        return intersection, grad


#################
'''plot tasks '''
##################
if __name__ == '__main__':
    ''' run this to verify the features generated'''
    from dacite import from_dict
    from dataclasses import dataclass
    import matplotlib
    from matplotlib import pyplot as plt
    from matplotlib.colors import ListedColormap

    @dataclass
    class Configuration:
        include_lines= False
        num_ways= 1
        num_support= 5
        num_query= 10
        num_tasks= 500
        varying_shot= True
        name = 'sinusoids'

    configs = Configuration()

    datasets = SineRegression(configs)
    datasets.set_mode('test_id')
    fig, axs = plt.subplots(3, 3, sharex=True, sharey=True, figsize=(10, 10))
    for i in range(9):
        (x_s, y_s), (x_q, y_q), (title, id, ood) = datasets[i]
        j, k = int(i / 3), i % 3
        axs[0, 0].scatter(x_q, y_q, c='b', marker='.')
        axs[j, k].scatter(x_s, y_s, c='r', marker='o')
        axs[j, k].set_title(title)
    axs[j, k].set_xlim(-10, 10)
    plt.show()








