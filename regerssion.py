from mnnbox import *
import matplotlib.pyplot as plt
import random
import os


def data_iter(batch_size, features, labels):
    num_examples = features.shape[0]
    index = list(range(num_examples))
    random.shuffle(index)
    for i in range(0, num_examples, batch_size):
        j = index[i: min(i + batch_size, num_examples)]
        yield features[j], labels[j]


def main():
    epochs = 1200
    total_samples = 100
    neurons = 10
    bs = 16
    X = Input('input')
    layer = [X]
    weight = []
    hidd = 3
    W = [Variable('weight_{}'.format(i)) for i in range(hidd)]
    Gamma = [Variable('gamma_{}'.format(i)) for i in range(hidd)]
    Beta = [Variable('beta_{}'.format(i)) for i in range(hidd)]
    combination = []
    batchnorm = []
    for i in range(hidd):
        combine = Combine(layer[i], W[i], 'combine_{}'.format(i))
        combination.append([combine])
        bn = BatchNormalization(1, combine, Gamma[i], Beta[i], 'bn_{}'.format(i))
        batchnorm.append(bn)
        activate = Activate(bn, 'activate_{}'.format(i))
        layer.append(activate)
    target = Variable('target')
    cost = MSE(layer[-1], target)

    u_all = np.random.uniform(0.02, 0.1, size=(total_samples, neurons))
    u_init = np.random.uniform(0.02, 0.1, size=(bs, neurons))
    s_init = np.sqrt(u_init)  # cv = 1
    X_ = np.stack([u_init, s_init], axis=-1)
    W_ = np.random.uniform(0., 1., size=(neurons, neurons))
    gamma_ = np.array([1., 1.])
    beta_ = np.stack([np.ones(neurons) * 2., np.ones(neurons) * 10.], axis=-1)
    target_ = X_
    feed_dict = {X: X_, target: target_}
    for i in range(hidd):
        feed_dict[W[i]] = W_
        feed_dict[Gamma[i]] = gamma_
        feed_dict[Beta[i]] = beta_
    graph = topological_sort(feed_dict)

    path = os.path.join('./fig', 'regression_result')
    os.makedirs(path, exist_ok=True)
    plt.figure()
    plt.scatter(u_init[0, :], s_init[0, :], marker='o')
    plt.xlabel('mu')
    plt.ylabel('sigma')
    plt.savefig(os.path.join(path, 'test_sample.distribution.png'))
    plt.close()

    train_ables = W + Gamma + Beta
    count = 0
    loss_train = []
    for epoch in range(epochs):
        count += 1
        print('epoch {} of {}'.format(count, epochs))
        for u_input, u_target in data_iter(bs, u_all, u_all):
            X.value = np.stack([u_input, np.sqrt(u_input)], axis=-1)
            target.value = np.stack([u_target, np.sqrt(u_target)], axis=-1)
            forward_and_backward(graph)
            loss_train.append(cost.value)
            sgd_update(train_ables, 0.1)
        if count % 400 == 0:
            X.value = X_
            target.value = target_
            loss = forward_pass(cost, graph)
            plt.figure()
            plt.scatter(layer[-1].value[0, :, 0], layer[-1].value[0, :, 1], marker='o')
            plt.xlabel('mu')
            plt.ylabel('sigma')
            plt.savefig(os.path.join(path, '{}epoch.output_distribution.png'.format(count)))
            plt.close()

    fig = plt.figure()
    plt.plot(np.range(len(loss_train)), loss_train, label='loss_train')
    plt.legend(loc='best')
    plt.savefig(os.path.join(path, 'loss_train.png'))
    plt.close()


def single_batch_repeat_training():
    epochs = 2000
    neurons = 10
    bs = 16
    X = Input('input')
    layer = [X]
    weight = []
    hidd = 3
    W = [Variable('weight_{}'.format(i)) for i in range(hidd)]
    Gamma = [Variable('gamma_{}'.format(i)) for i in range(hidd)]
    Beta = [Variable('beta_{}'.format(i)) for i in range(hidd)]
    combination = []
    batchnorm = []
    for i in range(hidd):
        combine = Combine(layer[i], W[i], 'combine_{}'.format(i))
        combination.append([combine])
        bn = BatchNormalization(1, combine, Gamma[i], Beta[i], 'bn_{}'.format(i))
        batchnorm.append(bn)
        activate = Activate(bn, 'activate_{}'.format(i))
        layer.append(activate)
    target = Variable('target')
    cost = MSE(layer[-1], target)

    u_init = np.random.uniform(0.02, 0.1, size=(bs, neurons))
    s_init = np.sqrt(u_init)  # cv = 1
    X_ = np.stack([u_init, s_init], axis=-1)
    W_ = np.random.exponential(0.3, (neurons, neurons))
    gamma_ = np.array([1., 1.])
    beta_ = np.stack([np.ones(neurons) * 2., np.ones(neurons) * 10.], axis=-1)
    target_ = X_
    feed_dict = {X: X_, target: target_}
    for i in range(hidd):
        feed_dict[W[i]] = W_
        feed_dict[Gamma[i]] = gamma_
        feed_dict[Beta[i]] = beta_
    graph = topological_sort(feed_dict)

    path = os.path.join('./fig', 'single_batch_regression')
    os.makedirs(path, exist_ok=True)
    plt.figure()
    plt.scatter(u_init[0, :], s_init[0, :], marker='o')
    plt.xlabel('mu')
    plt.ylabel('sigma')
    plt.savefig(os.path.join(path, 'test_sample.distribution.png'))
    plt.close()

    train_ables = W + Gamma + Beta
    count = 0
    loss_train = []
    for epoch in range(epochs):
        count += 1
        print('epoch {} of {}'.format(count, epochs))
        forward_and_backward(graph)
        loss_train.append(cost.value)
        sgd_update(train_ables, 1)
        if count % 500 == 0:
            plt.figure()
            plt.scatter(layer[-1].value[0, :, 0], layer[-1].value[0, :, 1], marker='o')
            plt.xlabel('mu')
            plt.ylabel('sigma')
            plt.savefig(os.path.join(path, '{}epoch.output_distribution.png'.format(count)))
            plt.close()
    np.save(os.path.join(path, 'loss_train.npy'), loss_train)
    fig = plt.figure()
    plt.plot(np.arange(len(loss_train)), loss_train, label='loss_train')
    plt.legend(loc='best')
    plt.savefig(os.path.join(path, 'loss_train.png'))
    plt.close()


if __name__ == '__main__':
    single_batch_repeat_training()
