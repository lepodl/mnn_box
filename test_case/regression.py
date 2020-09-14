import mnnbox as mnn
import matplotlib.pyplot as plt
import random
import numpy as np
import os

def data_iter(batch_size, features, labels):
    num_examples = features.shape[0]
    index = list(range(num_examples))
    random.shuffle(index)
    for i in range(0, num_examples, batch_size):
        j = index[i: min(i + batch_size, num_examples)]
        yield features[j], labels[j]


# test case 1:
# we want to regress a function using the batch training
# within the framework of mnn. e.g., f(x) = x.
def main():
    # step 1: to design the network
    # e.g., mlp network with 3 hidden layer, i.e, totally 4 layers
    # from now on, each node of mnn represents a operator class.
    total_samples = 1000
    epochs = 2000
    bs = 16  # batch size
    neurons = 10 # here, we suppose the network is consistent that each layer consists of 10 neurons.
    hidden = 3 # num of hidden layers

    #-----------------------------------------------------------------------------
    # if no batch normalization

    x =  mnn.Input(value=np.empty((bs, neurons, 2)), name="input") # a node but only act as a placeholder, no operator
    target = mnn.Variable(value=np.empty((bs, neurons, 2)), name='target')
    all_weight = [mnn.Variable(value=mnn.truncated_normal([neurons, neurons], stddev=0.1), name='weight{}'.format(i)) for i in range(hidden)]

    # compose1 = mnn.Compose(x, all_weight[0], name="compose_op_1")
    # layer1 = mnn.Activate(compose1, name="layer1")
    # then layer 2, layer 3, layer 4

    # so we can make a loop to construct the nn at once
    all_compose = []
    total_layer = [x, ]
    for i in range(hidden):
        compose = mnn.Compose(total_layer[i], all_weight[i], name="compoe_op{}".format(i+1))
        all_compose.append(compose)
        layer = mnn.Activate(compose, name="layer{}".format(i+1))
        total_layer.append(layer)
    cost = mnn.MSE(total_layer[-1], target,)

    #------------------------------------------------------------------------------
    # if implement batch normalization, we need to add two other place holder
    # all_gamma = [mnn.Variable(np.ones((neurons, 2)), 'gamma{}'.format(i)) for i in range(hidden)]
    # all_beta = [ mnn.Variable(np.stack([np.ones(neurons) * 2., np.ones(neurons) * 10.], axis=1), 'beta{}'.format(i)) for i in range(hidden)]
    # all_compose = []
    # total_layer = [x, ]
    # bn = []
    # for i in range(hidden):
    #     compose = mnn.Compose(total_layer[i], all_weight[i], name="compoe_op{}".format(i + 1))
    #     all_compose.append(compose)
    #     bn = mnn.BatchNormalization(compose, all_gamma[i], all_beta[i], {'mode': 'train'}, "bn_op{}".format(i+1))
    #     layer = mnn.Activate(bn, name="layer{}".format(i + 1))
    #     total_layer.append(layer)
    # cost = mnn.MSE(total_layer[-1], target, )

    #-------------------------------------------------------------------------------

    # step 2: generate the node sequence fo the given graph
    # feed_dict is a dictionary includes of all place holder node,
    # i.e, the header node, and we can use toplogical_sort function to
    # generate a node sequence and assign values ​​to these nodes.
    feed_dict = {x: None}
    for i in range(hidden):
        feed_dict[all_weight[i]] = None
        # feed_dict[all_gamma[i]] = None
        # feed_dict[all_beta[i]] = None
    graph_sequence = mnn.topological_sort(feed_dict=feed_dict)

    # step 3: pass_forward and back_forward
    u_all = np.random.uniform(0.02, 0.1, size=(total_samples, neurons))
    s_all = np.sqrt(u_all)
    u_test = np.random.uniform(0.02, 0.1, size=(bs, neurons))
    s_test = np.sqrt(u_test)

    train_ables = all_weight
    # train_ables = all_weight + all_gamma + all_beta
    count = 0
    loss_train = []
    loss_test = []
    for epoch in range(epochs):
        count += 1
        print('epoch {} of {}'.format(count, epochs))
        for u_input, u_target in data_iter(bs, u_all, u_all):
            x.value = np.stack([u_input, np.sqrt(u_input)], axis=-1)
            target.value = np.stack([u_target, np.sqrt(u_target)], axis=-1)
            mnn.forward_and_backward(graph_sequence)
            loss_train.append(cost.value)
            mnn.sgd_update(train_ables, 0.1)
        # test:
        if count % 400 == 0:
            x.value = np.stack([u_test, s_test])
            target.value = np.stack([u_test, s_test])
            loss = mnn.forward_pass(cost, graph_sequence)
            loss_test.append(loss)

    # step 4: plot and analyse
    fig = plt.figure()
    plt.plot(np.range(len(loss_train)), loss_train, label='loss_train')
    plt.plot(np.range(len(loss_test)) * 400, loss_test, label='loss_train')
    plt.legend(loc='best')
    plt.show()


if __name__ == '__main__':
    main()

