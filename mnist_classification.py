import numpy as np
import torch
import torchvision
from matplotlib import pyplot as plt
import unittest
import mnnbox as mnn
import os
import time


def one_hot(labels, depth=10):
    label = labels.numpy()
    return np.eye(depth)[label.reshape(-1)]


def main():
    epochs = 2
    batch_size = 256

    data_train = torchvision.datasets.MNIST("./mnist/", train=True, download=False,
                                            transform=torchvision.transforms.Compose(
                                                [torchvision.transforms.ToTensor(),
                                                 torchvision.transforms.Normalize((0.1307,), (0.3081,))]
                                            ))
    data_test = torchvision.datasets.MNIST("./mnist/", train=False, download=False,
                                           transform=torchvision.transforms.Compose(
                                               [torchvision.transforms.ToTensor(),
                                                torchvision.transforms.Normalize((0.1307,), (0.3081,))]))

    train_loader = torch.utils.data.DataLoader(
        data_train, batch_size=batch_size, shuffle=True
    )

    test_loader = torch.utils.data.DataLoader(
        data_test, batch_size=batch_size, shuffle=True
    )

    x_test, y_test = next(iter(test_loader))
    in_units = 784
    o_units = 10
    hi_units = 300
    w1 = mnn.Variable(mnn.truncated_normal([hi_units, in_units], stddev=0.1), 'weight1')
    x = mnn.Input(np.empty((batch_size, in_units)))
    hidden1 = mnn.Compose(x, w1, 'compose_op_1')
    gamma1, beta1 = mnn.Variable(np.ones((hi_units, 2)), 'gamma1'), mnn.Variable(
        np.stack([np.ones(hi_units) * 2., np.ones(hi_units) * 10.], axis=1), 'beta1')
    bn1 = mnn.BatchNormalization(hidden1, gamma1, beta1, {'mode': 'train'}, "bn_op_1")
    activate1 = mnn.Activate(bn1, 'activation_op_1')
    w2 = mnn.Variable(mnn.truncated_normal([o_units, hi_units], stddev=0.1), 'weight2')
    hidden2 = mnn.Compose(activate1, w2, 'compose_op_2')
    label = mnn.Variable(np.empty(o_units), 'label')
    cross_entropy = mnn.softmax_cross_entropy_with_logits(logits=hidden2, labels=label, name='cost')

    feed_dict = {x: None, w1: None, gamma1: None, beta1: None, w2: None, label: None}
    graph = mnn.topological_sort(feed_dict)

    train_loss_list, test_loss_list, train_acc_list, test_acc_list = [], [], [], []
    train_step = mnn.GradientDescentOptimizer(graph, [w1, w2, gamma1, beta1], 1.)
    for _ in range(epochs):
        i = 0
        for x_, y_ in train_loader:
            i += 1
            start = time.time()
            x__ = x_.numpy().reshape((-1, in_units))
            x.value = np.stack([x__, np.abs(x__)], axis=-1)
            label.value = one_hot(y_)
            train_step.run()
            train_loss = cross_entropy.value.mean()
            train_acc = np.mean(np.argmax(cross_entropy.my_eval(), axis=1) == y_.numpy())
            print('\ntrain loss:\t', train_loss)
            print('train accuracy:\t', train_acc)
            print('batch {} takes time:'.format(i), time.time()-start)
            if i % 10 ==0:
                train_loss_list.append(train_loss)
                train_acc_list.append(train_acc)
                xx_test = x_test.numpy().reshape((-1, in_units))
                x.value = np.stack([xx_test, np.abs(xx_test)], axis=-1)
                label.value = one_hot(y_test)
                bn1.mode = 'test'
                test_loss = mnn.forward_pass(cross_entropy, graph)
                test_loss_list.append(test_loss.mean())
                test_acc = np.mean(np.argmax(cross_entropy.my_eval(), axis=1) == y_test.numpy())
                test_acc_list.append(test_acc)
                print('============================\ntest accuracy:\t', test_acc)
                bn1.mode = 'train'
    path = os.path.join('./', 'mnist_classification')
    os.makedirs(path, exist_ok=True)
    plt.figure(1)
    plt.plot(np.arange(len(train_loss_list)) * 5, train_loss_list, label='train_loss')
    plt.plot(np.arange(len(test_loss_list)) * 5, test_loss_list, label='test_loss')
    plt.xlabel('batch idx')
    plt.ylabel('loss')
    plt.legend(loc='best')
    plt.savefig(os.path.join(path, 'loss.png'))
    plt.figure(2)
    plt.plot(np.arange(len(train_acc_list)), train_acc_list, label='train_acc')
    plt.plot(np.arange(len(test_acc_list)), test_acc_list, label='test_acc')
    plt.xlabel('batch idx')
    plt.ylabel('accuracy')
    plt.legend(loc='best')
    plt.savefig(os.path.join(path, 'accuracy.png'))

if __name__ == '__main__':
    main()
