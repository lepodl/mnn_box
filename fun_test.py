import numpy as np
import random
import matplotlib.pyplot as plt

def data_iter(batch_size, features, labels):
    num_examples = features.shape[0]
    index = list(range(num_examples))
    random.shuffle(index)
    for i in range(0, num_examples, batch_size):
        j = index[i: min(i + batch_size, num_examples)]
        yield features[j], labels[j]

# total_samples = 100
# neurons = 10
# bs =16
# u_all = np.random.uniform(0.02, 0.1, size=(total_samples, neurons))
# for u_input, u_target in data_iter(bs, u_all, u_all):
#     print(u_input.shape)
#     break

neurons = 2
W_ = np.random.exponential(0.3, 1000)
plt.hist(W_, bins=40)
plt.show()