from mnnbox import *
import matplotlib.pyplot as plt
import unittest
import os


class Test_MnnBox(unittest.TestCase):
    def _test_gradient_bn(self):
        x = Input()
        gamma = Variable('gamma')
        beta = Variable('beta')
        target = Input('target')
        u_input = np.random.uniform(size=(4, 100))
        s_input = np.random.uniform(size=(4, 100))
        x_ = np.stack([u_input, s_input], axis=-1)
        gamma_ = np.array([0.5, 0.5])
        beta_ = np.ones((100, 2)) * 2.
        target_ = np.stack([u_input, s_input], axis=-1)
        feed_dict = {x: x_, gamma: gamma_, beta: beta_, target: target_}
        out = BatchNormalization(1, x, gamma, beta)
        cost = MSE(out, target)
        graph = topological_sort(feed_dict)
        forward_and_backward(graph)
        loss = cost.value
        # print('\n---------------->the gradient of x\n', out.gradients[x][1, 1, 0])
        # print('\n---------------->the gradient of gamma\n', gamma.gradients[gamma][1])
        print('\n---------------->the gradient of beta\n', beta.gradients[beta][1, 0])

        # x_[1, 1, 0] = x_[1, 1, 0] + 0.0001
        # x.value = x_
        # gamma_[1] = gamma_[1] + 0.0001
        # gamma.value = gamma_
        beta_[1, 0] = beta_[1, 0] + 0.0001
        loss_ = forward_pass(cost, graph)
        grad = (loss_ - loss) / 0.0001
        # print('\n---------------->validate the gradient of x\n', grad)
        # print('\n---------------->validate the gradient of gamma\n', grad)
        print('\n---------------->validate the gradient of beta\n', grad)

    def _test_gradient_of_act(self):
        X = Input()
        W = Variable()
        u_input = np.ones((4, 100)) * 2.3
        s_input = np.ones((4, 100)) * 1.
        X_ = np.stack([u_input, s_input], axis=-1)
        u_output = np.ones((4, 100)) * 0.1
        s_output = np.ones((4, 100)) * 0.44
        target_ = np.stack([u_output, s_output], axis=-1)
        tar = Input('target')
        res = Activate(X)
        cost = MSE(res, tar)
        feed_dict = {X: X_, tar: target_}
        graph = topological_sort(feed_dict)
        forward_and_backward(graph)
        cost_1 = cost.value
        print('the gradient of X:', X.gradients[X][0, 0, 0])

        X_[0, 0, 0] = X_[0, 0, 0] + 1e-3
        X.value = X_
        cost_2 = forward_pass(cost, graph)
        grad_check = (cost_2 - cost_1) / (1e-3)
        print('\n=====================\ngradient for check', grad_check)

    def test_whole_nn(self):
        neurons = 10
        bs = 9
        X = Input('input')
        layer = [X]
        weight = []
        hidd = 4
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

        u_input = np.random.uniform(0., 0.1, size=(bs, neurons))
        s_input = np.sqrt(u_input)  # cv = 1
        X_ = np.stack([u_input, s_input], axis=-1)
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
        forward_and_backward(graph)
        # train_ables = W + Gamma + Beta

        path = os.path.join('./fig', 'nn_property')
        os.makedirs(path, exist_ok=True)
        for l in range(hidd):
            bn_out = batchnorm[l].value
            out = layer[l].value
            grad = W[l].gradients[W[l]].flatten()
            fig, axes = plt.subplots(3,3, figsize=(12, 8))
            for i in range(3):
                for j in range(3):
                    axes[i][j].scatter(out[3 * i + j, :, 0], out[3 * i + j, :, 1], marker='o')
            plt.savefig(os.path.join(path, 'layer{}_distribution.png'.format(l)))
            plt.close()
            fig, axes = plt.subplots(3, 3, figsize=(12, 8))
            for i in range(3):
                for j in range(3):
                    axes[i][j].scatter(bn_out[3 * i + j, :, 0], bn_out[3 * i + j, :, 1], marker='o')
            plt.savefig(os.path.join(path, 'bn_layer{}_distribution.png'.format(l)))
            plt.close()
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            for i in range(2):
                for j in range(2):
                    axes[i][j].hist(grad, bins=20, facecolor="blue", edgecolor="black", alpha=0.7)
            plt.savefig(os.path.join(path, 'gradient_of_weight{}.png'.format(l)))
            plt.close()
        grad_gamma_u = [Gamma[j].gradients[Gamma[j]][0] for j in range(4)]
        grad_gamma_s = [Gamma[j].gradients[Gamma[j]][1] for j in range(4)]
        print('\n --------gradient of gamma_u:\n', grad_gamma_u)
        print('\n --------gradient of gamma_s:\n', grad_gamma_s)
        print('\n --------gradient of 0th layer beta_u:\n', Beta[0].gradients[Beta[0]][:, 0])
        print('\n --------gradient of 0th layer beta_s:\n', Beta[0].gradients[Beta[0]][:, 1])


if __name__ == '__main__':
    unittest.main()



