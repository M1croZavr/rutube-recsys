from .base import AbstractNegativeSampler

from tqdm import trange

import numpy as np


class RandomNegativeSampler(AbstractNegativeSampler):
    @classmethod
    def code(cls):
        return 'random'

    def generate_negative_samples(self):
        assert self.seed is not None, 'Specify seed for random sampling'
        np.random.seed(self.seed)
        negative_samples = {}
        print('Sampling negative items')
        for user in trange(self.user_count):
            try:
                if isinstance(self.train[user][0], tuple):
                    seen = set(x[0] for x in self.train.get(user, []))
                    seen.update(x[0] for x in self.val.get(user, []))
                    seen.update(x[0] for x in self.test.get(user, []))
                else:
                    seen = set(self.train.get(user,[]))
                    seen.update(self.val.get(user,[]))
                    seen.update(self.test.get(user, []))
                samples = []
                for _ in range(self.sample_size):
                    item = np.random.choice(self.item_count) + 1
                    while item in seen or item in samples:
                        item = np.random.choice(self.item_count) + 1
                    samples.append(item)

                negative_samples[user] = samples
            except Exception as ex:
                pass

        return negative_samples
