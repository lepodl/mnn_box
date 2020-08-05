import numpy as np
import matplotlib.pyplot as plot
from utils import Fun, dawson, dbl_dawson

DEBUG = False
T_DEBUG = False


# TODO(luckyzlb15@163.com): extend the model to deal with the data in batch
# batch information: (bathch_size, neurons, 2)

class Node(object):
    """
    Base class for nodes in the network.
    Should have following properties:
    1. Should hold its value, including the fire_rate and variance (u, s)
    2. Should know what are incoming nodes
    3. Should know to which node(s) it outputs the value
    4. Should hold the gradient calculated

    Arguments:

        `inbound_nodes`: A list of nodes with edges into this node.
    """

    def __init__(self, inbound_nodes=[], ratio=0.8, v_r=0, v_th=20, L=0.05, t_ref=5):
        """
        Node's constructor (runs when the object is instantiated). Sets
        properties that all nodes need.
        """
        self.ratio = ratio
        self.v_r = v_r
        self.v_th = v_th
        self.L = L
        self.t_ref = t_ref

        self.name = "Node"

        # The eventual value of this node. Set by running
        # the forward() method.
        # neglect the correction coefficient
        self.value = None

        # A list of nodes with edges into this node.
        # Just like input arguments to any function/method
        self.inbound_nodes = inbound_nodes

        # A list of nodes that this node outputs to.
        # Is it possible to know which node I am gonna send the result? Definelty NO!!!
        self.outbound_nodes = []

        # Keys are the inputs to this node and
        # their values are the partials of this node with
        # respect to that input.
        # Totally include four values refers to different gradients w.r.t one node.
        # [[du/du,  du/ds]
        #  [ds/du,  ds/ds]]
        self.gradients = {}

        # Sets this node as an outbound node for all of
        # this node's inputs.
        # Hey there I am your output node, do send me your results, ok!
        for node in inbound_nodes:
            node.outbound_nodes.append(self)

    def forward(self):
        """
        Every node that uses this class as a base class will
        need to define its own `forward` method.
        """
        raise NotImplementedError

    def backward(self):
        """
        Every node that uses this class as a base class will
        need to define its own `backward` method.
        """
        raise NotImplementedError


class Input(Node):
    """
    A generic input into the network.
    """

    def __init__(self, name='Input'):
        # The base class constructor has to run to set all
        # the properties here.
        #
        # The most important property on an Input is value.
        # self.value is set during `topological_sort` later.
        Node.__init__(self)
        self.name = name

    # NOTE: Input node is the only node where the value
    # may be passed as an argument to forward().
    #
    # All other node implementations should get the value
    # of the previous node from self.inbound_nodes
    #
    # Example:
    # val0 = self.inbound_nodes[0].value
    def forward(self, value=None):
        # Overwrite the value if one is passed in.
        if (DEBUG): print("\n----->Forward pass @ ", self.name)
        if value is not None:
            self.value = value
            if (DEBUG): print("w.r.t {} node of value: {} ".format(self.name, self.value))
        batch_size = self.value.shape[0]
        dim = self.value.shape[1]
        # each entry has a rou
        rou = np.eye(dim) * 0.9 + np.ones(dim) * 0.1
        self.rou = np.expand_dims(rou, 0).repeat(batch_size, axis=0)

    def backward(self):
        # An Input node has no inputs (the output is the node's value),
        # so the gradient (derivative) is zero.
        # The key, `self`, is reference to this object.
        self.gradients = {self: 0}
        # Weights and bias may be inputs, so you need to sum
        # the gradient from output gradients.
        if (DEBUG): print('\n')
        if (DEBUG): print('=============================\n\tBP @ {}\n=============================\n'.format(self.name))
        if (DEBUG): print('Initial Gradients:\n------------------')
        if (DEBUG): print('W.r.t {}: \n------------\n{}'.format(self.name, self.gradients[self]))

        for n in self.outbound_nodes:
            grad_cost = n.gradients[self]

            if (DEBUG): print('\n')
            if (DEBUG): print('Getting ', n.name, 'gradient : \n<-----------------------------\n', grad_cost)
            if (DEBUG): print('\n')

            self.gradients[self] += grad_cost * 1

        if (DEBUG): print('Calculated Final Gradient:(Note: Calculated by next node in the graph!!!)\n----------------')
        if (DEBUG): print('W.r.t ', self.name, ' : \n-------------\n', self.gradients[self])


class Variable(Node):
    def __init__(self, name='variable'):
        # The base class constructor has to run to set all.
        Node.__init__(self)
        self.name = name

    def forward(self, value=None):
        # Overwrite the value if one is passed in.
        if (DEBUG): print("\n----->Forward pass @ ", self.name)
        if value is not None:
            self.value = value
            if (DEBUG): print("w.r.t {} node of value: {} ".format(self.name, self.value))

    def backward(self):
        # An Input node has no inputs (the output is the node's value),
        # so the gradient (derivative) is zero.
        # The key, `self`, is reference to this object.
        self.gradients = {self: 0}
        # Weights and bias may be inputs, so you need to sum
        # the gradient from output gradients.
        if (DEBUG): print('\n')
        if (DEBUG): print('=============================\n\tBP @ {}\n=============================\n'.format(self.name))
        if (DEBUG): print('Initial Gradients:\n------------------')
        if (DEBUG): print('W.r.t {}: \n------------\n{}'.format(self.name, self.gradients[self]))

        for n in self.outbound_nodes:
            grad_cost = n.gradients[self]

            if (DEBUG): print('\n')
            if (DEBUG): print('Getting ', n.name, 'gradient : \n<-----------------------------\n', grad_cost)
            if (DEBUG): print('\n')

            self.gradients[self] += grad_cost * 1

        if (DEBUG): print('Calculated Final Gradient:(Note: Calculated by next node in the graph!!!)\n----------------')
        if (DEBUG): print('W.r.t ', self.name, ' : \n-------------\n', self.gradients[self])


def topological_sort(feed_dict):
    """
    Sort the nodes in topological order using Kahn's Algorithm.

    `feed_dict`: A dictionary where the key is a `Input` Node and the value is
    the respective value feed to that Node.

    Returns a list of sorted nodes.
    """
    if T_DEBUG: print('-----> topological_sort')
    input_nodes = [n for n in feed_dict.keys()]

    G = {}
    nodes = [n for n in input_nodes]

    if T_DEBUG: print('Input Nodes:'); [print(n.name) for n in input_nodes]

    while len(nodes) > 0:
        n = nodes.pop(0)

        if T_DEBUG: print('Pop: ', n.name)

        if n not in G:
            if T_DEBUG: print('Adding: ', n.name, 'to the Graph')
            G[n] = {'in': set(), 'out': set()}

        for m in n.outbound_nodes:
            if m not in G:
                if T_DEBUG: print('Adding: ', m.name, 'to the Graph')
                G[m] = {'in': set(), 'out': set()}

            G[n]['out'].add(m)
            if T_DEBUG: print('Adding', n.name, '----->', m.name)

            G[m]['in'].add(n)
            if T_DEBUG: print('Adding', m.name, '<-----', n.name)

            nodes.append(m)
            if T_DEBUG: print('Appending ', m.name)

    L = []
    S = set(input_nodes)
    if T_DEBUG: print('Input Nodes:'); [print(n.name) for n in S]
    while len(S) > 0:
        n = S.pop()
        if T_DEBUG: print('Pop: ', n.name)

        # Assign values to the input node
        if isinstance(n, Input) or isinstance(n, Variable):
            if T_DEBUG: print('Feeding value: ', feed_dict[n], ' =====>  ', n.name)
            n.value = feed_dict[n]

        L.append(n)
        if T_DEBUG: print('Adding ', n.name, 'to the sorted List')
        for m in n.outbound_nodes:
            G[n]['out'].remove(m)
            G[m]['in'].remove(n)
            if T_DEBUG: print('Removing', n.name, '----->', m.name)
            if T_DEBUG: print('Removing', m.name, '<-----', n.name)
            # if no other incoming edges add to S
            if len(G[m]['in']) == 0:
                if T_DEBUG: print('\nNo input nodes!!! Adding: ', m.name, 'to the Graph\n')
                S.add(m)

    if T_DEBUG: print('Sorted Nodes:\n'); [print(n.name) for n in L]

    if T_DEBUG: print('<------------------------------------ topological_sort')

    return L


def forward_pass(output_node, sorted_nodes):
    """
    Performs a forward pass through a list of sorted nodes.

    Arguments:

        `output_node`: A node in the graph, should be the output node (have no outgoing edges).
        `sorted_nodes`: A topologically sorted list of nodes.

    Returns the output Node's value
    """

    for n in sorted_nodes:
        n.forward()

    return output_node.value


# define the operations in the mnn framework
class Combine(Node):
    """
    Represents a node that performs a linear/nonlinear transform.
    to specify the combine_operation on the u and s of a neuron/node.
    """

    def __init__(self, X, W, name='combine_op'):
        # The base class (Node) constructor. Weights and bias
        # are treated like inbound nodes.
        Node.__init__(self, [X, W])
        self.name = name

    def forward(self):
        """
        Performs the math behind a nonlinear transform.
        """

        self.X = self.inbound_nodes[0]  # X.value.shape=(batch_size, nodes, 2)
        self.W = self.inbound_nodes[1]
        self.rou = self.inbound_nodes[0].rou  # shape = (batch_size, nodes, nodes)

        # u and s, shape = (batch_size, nodes)
        u = np.einsum('ij,kj->ki', self.W.value, self.X.value[:, :, 0]) * (1 - self.ratio)
        s = np.sqrt(
            np.einsum('im,km,in,kn,kmn->ki', self.W.value, self.X.value[:, :, 1], self.W.value, self.X.value[:, :, 1],
                      self.rou) * (1 + self.ratio ** 2))
        self.value = np.stack([u, s], axis=-1)

        # denominator = np.einsum('ki,kj->kij', s, s)
        # self.rou = np.einsum('im,km,jn,kn,kmn->kij', self.W.value, self.X.value[:, :, 1], self.W.value, self.X.value[:, :, 1], self.rou) * (1 + self.ratio ** 2) / denominator

        # if True:  print("\n================>Forward pass @ ", self.name)
        # if True: print("u_hat:{}".format(u[1, :5]))
        # if True: print("s_hat:{}".format(s[1, :5]))

    def backward(self):
        """
        Calculates the gradient based on the output values.
        """
        # Initialize a partial for each of the inbound_nodes.
        self.gradients = {n: np.zeros_like(n.value) for n in self.inbound_nodes}
        # Cycle through the outputs. The gradient will change depending
        # on each output, so the gradients are summed over all outputs.

        if (DEBUG): print('\n')
        if (DEBUG): print('=============================\n\tBP @ Combine\n=============================\n')
        if (DEBUG): print('Initial Gradients:\n------------------')
        if (DEBUG): print('W.r.t {}: \n---------------\n{}'.format(self.X.name, self.gradients[self.X]))
        if (DEBUG): print('W.r.t {}: \n---------------\n{}'.format(self.W.name, self.gradients[self.W]))

        for n in self.outbound_nodes:
            # Get the partial of the cost with respect to this node.
            # The out is mostly only one node.
            grad_cost = n.gradients[self]  # shape = self.value.shape = (this layer neurons, 2)

            if (DEBUG): print('\n')
            if (DEBUG): print('Getting ', n.name, 'gradient is : \n<-----------------------------\n', grad_cost)
            if (DEBUG): print('\n')

            # Get the gradient for this node from next node and respective operation
            # with each input of this node to set their respective gradients
            # Set the partial of the loss with respect to this node's inputs.

            # shape = (batch_size, neurons, 2)
            grad_u = np.einsum('ij,ki->kj', self.W.value, grad_cost[:, :, 0]) * (1 - self.ratio)
            # grad_u = np.dot(self.W.value.T, grad_cost[:, 0]) * (1 - self.ratio)
            inv = 1 / (2 * self.value[:, :, 1])
            temp = np.einsum('bi,ij,ik,bjk,bk->bij', inv, self.W.value, self.W.value, self.X.rou,
                             self.X.value[:, :, 1]) * 2 * (
                           1 + self.ratio ** 2)
            grad_s = np.einsum('bij,bi->bj', temp, grad_cost[:, :, 1])
            self.gradients[self.X] += np.stack([grad_u, grad_s], axis=-1)

            # Set the partial of the loss with respect to this node's weights.
            ds_dw = np.einsum('bk,ki,bi,bj,bij->bkj', inv, self.W.value, self.X.value[:, :, 1], self.X.value[:, :, 1],
                              self.X.rou) * 2 * (1 + self.ratio ** 2)
            grad_u_part_w = np.einsum('bi,bj->bij', grad_cost[:, :, 0], self.X.value[:, :, 0]) * (
                        1 - self.ratio)  # dE/du * du/dw
            grad_s_part_w = np.einsum('bi,bij->bij', grad_cost[:, :, 1], ds_dw)  # dE/ds * ds/dw
            batch_grad_w = grad_u_part_w + grad_s_part_w
            self.gradients[self.W] += np.sum(batch_grad_w, axis=0)
        if (DEBUG): print('Calculated Final Gradient:\n----------------')
        if (DEBUG): print('W.r.t ', self.X.name, ': \n-------------\n', self.gradients[self.inbound_nodes[0]])
        if (DEBUG): print('W.r.t ', self.W.name, ': \n-------------\n', self.gradients[self.inbound_nodes[1]])


class MSE(Node):
    def __init__(self, y, a):
        """
        The mean squared error cost function.
        Should be used as the last node for a network.
        """
        # Call the base class' constructor.
        Node.__init__(self, [y, a])  # y is the output of the network and a is the target.
        self.name = "MSE_Op"

    def forward(self):
        """
        Calculates the mean squared error.
        """
        # NOTE: We reshape these to avoid possible matrix/vector broadcast
        # errors.
        #
        # For example, if we subtract an array of shape (3,) from an array of shape
        # (3,1) we get an array of shape(3,3) as the result when we want
        # an array of shape (3,1) instead.
        #
        # Making both arrays (3,1) insures the result is (3,1) and does
        # an elementwise subtraction as expected.
        if (DEBUG): print("\n----->Forward pass @ ", self.name)
        if (DEBUG): print("Initial value of {} is {}".format(self.name, self.value))

        y_u = self.inbound_nodes[0].value[:, :, 0]
        y_s = self.inbound_nodes[0].value[:, :, 1]
        a_u = self.inbound_nodes[1].value[:, :, 0]
        a_s = self.inbound_nodes[1].value[:, :, 1]

        batch_size = self.inbound_nodes[0].value.shape[0]
        num = self.inbound_nodes[0].value.shape[1]
        self.m = batch_size * num
        # Save the computed output for backward.
        self.diff_u = y_u - a_u
        self.diff_s = y_s - a_s

        value_u = np.mean(np.square(self.diff_u))
        value_s = np.mean(np.square(self.diff_s))
        self.value = value_u + value_s

        if (DEBUG): print("\n {}:\n{}".format(self.name, self.value))

    def backward(self):
        """
        Calculates the gradient of the cost.

        This is the final node of the network so outbound nodes
        are not a concern.
        """
        if (DEBUG): print('\n')
        if (DEBUG): print('=============================\n\tBP @ MSE\n=============================\n')
        if (DEBUG): print('Initial Gradients:\n------------------')
        if (DEBUG): print('Nothing! Since this node will be the last node!!!\n')

        grad_out_u = (2 / self.m) * self.diff_u
        grad_out_s = (2 / self.m) * self.diff_s

        grad_target_u = (-2 / self.m) * self.diff_u
        grad_target_s = (-2 / self.m) * self.diff_s

        self.gradients[self.inbound_nodes[0]] = np.stack([grad_out_u, grad_out_s], axis=-1)
        self.gradients[self.inbound_nodes[1]] = np.stack([grad_target_u, grad_target_s], axis=-1)

        if (DEBUG): print('Calculated Final Gradient:\n----------------')
        if (DEBUG): print('W.r.t ', self.inbound_nodes[0].name, ': \n------------------\n',
                          self.gradients[self.inbound_nodes[0]])
        if (DEBUG): print('W.r.t ', self.inbound_nodes[1].name, ': \n------------------\n',
                          self.gradients[self.inbound_nodes[1]])


def forward_and_backward(graph):
    """
    Performs a forward pass and a backward pass through a list of sorted Nodes.

    Arguments:

        `graph`: The result of calling `topological_sort`.
    """
    # Forward pass
    for n in graph:
        n.forward()
    # print('\n<===========Forward over====================>\n')

    # Backward pass
    for n in graph[::-1]:
        n.backward()


class Activate(Node):
    '''
    implement the math behind the Mnn framework
    '''

    def __init__(self, X, name='activate_op'):
        Node.__init__(self, [X])
        self.name = name

    def forward(self):
        self.X = self.inbound_nodes[0]  # X.value.shape=(nodes, 2)
        self.rou = self.X.rou
        self.u_hat, self.s_hat = self.X.value[:, :, 0], self.X.value[:, :, 1]
        func = Fun()
        u = func.s_1(self.u_hat, self.s_hat)
        cv = func.s_2(self.u_hat, self.s_hat)
        s = cv * np.sqrt(u)
        self.value = np.stack([u, s], axis=-1)

        # if True:  print("\n================>Forward pass @ ", self.name)
        # if True: print("u:{}".format(u[1, :5]))
        # if True: print("s:{}".format(s[1, :5]))

    def backward(self):
        self.gradients = {n: np.zeros_like(n.value) for n in self.inbound_nodes}
        # Cycle through the outputs. The gradient will change depending
        # on each output, so the gradients are summed over all outputs.

        if (DEBUG): print('\n')
        if (DEBUG): print('=============================\n\tBP @ Activate\n=============================\n')
        if (DEBUG): print('Initial Gradients:\n------------------')
        if (DEBUG): print('W.r.t {}: \n---------------\n{}'.format(self.X.name, self.gradients[self.X]))

        for n in self.outbound_nodes:
            # Get the partial of the cost with respect to this node.
            # The out is mostly only one node.
            grad_cost = n.gradients[self]  # shape = self.value.shape = (this layer neurons, 2)

            if (DEBUG): print('\n')
            if (DEBUG): print('Getting ', n.name, 'gradient is : \n<-----------------------------\n', grad_cost)
            if (DEBUG): print('\n')

            # Get the gradient for this node from next node and respective operation
            # (mutliply/add) with each input of this node to set their respective gradients
            # Set the partial of the loss with respect to this node's inputs.

            u, s = self.value[:, :, 0], self.value[:, :, 1]
            s_3 = s / (u ** (3 / 2))
            i_1 = (self.v_r * self.L - self.u_hat) / self.s_hat
            i_2 = (self.v_th * self.L - self.u_hat) / self.s_hat
            du_du = (2 * u ** 2) * (dawson(i_2) - dawson(i_1)) / (self.s_hat * self.L)
            du_ds = (2 * u ** 2) * (i_2 * dawson(i_2) - i_1 * dawson(i_1)) / (self.s_hat * self.L)
            ds_du = (-4 / (s_3 * self.s_hat * self.L ** 2)) * (dbl_dawson(i_2) - dbl_dawson(i_1)) * u ** (
                    3 / 2) + 3 * s_3 * np.sqrt(u) * du_du / 2
            ds_ds = -4 * (i_2 * dbl_dawson(i_2) - i_1 * dbl_dawson(i_1)) * u ** (3 / 2) / (
                    s_3 * self.L ** 2 * self.s_hat) + 3 * s_3 * np.sqrt(u) * du_ds / 2

            grad_u = grad_cost[:, :, 0] * du_du + grad_cost[:, :, 1] * ds_du
            grad_s = grad_cost[:, :, 0] * du_ds + grad_cost[:, :, 1] * ds_ds
            self.gradients[self.X] += np.stack([grad_u, grad_s], axis=-1)

        if (DEBUG): print('Calculated Final Gradient:\n----------------')
        if (DEBUG): print('W.r.t ', self.X.name, ': \n-------------\n', self.gradients[self.inbound_nodes[0]])


def sgd_update(trainables, learning_rate=1e-2):
    """
    Updates the value of each trainable with SGD.

    Arguments:

        `trainables`: A list of `Input` Nodes representing weights/biases.
        `learning_rate`: The learning rate.
    """
    # Performs SGD
    #
    # Loop over the trainables
    for t in trainables:
        # Change the trainable's value by subtracting the learning rate
        # multiplied by the partial of the cost with respect to this
        # trainable.
        partial = t.gradients[t]
        t.value -= learning_rate * partial


class BatchNormalization(Node):
    '''
    During training the sample mean and (uncorrected) sample variance are
    computed from minibatch statistics and used to normalize the incoming data.
    During training we also keep an exponentially decaying running mean of the
    mean and variance of each feature, and these averages are used to normalize
    data at test-time.
    At each timestep we update the running averages for mean and variance using
    an exponential decay based on the momentum parameter:
    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var
    '''

    def __init__(self, X, gamma, beta, bn_param, name='bn_op'):
        '''

        :param X: shape=(bs, neurons, 2)
        :param gamma: shape=(neurons,)
        :param beta: shape=(neurons,)
        :param bn_param: dict consists of: {mode, eps, momentum, running_mean, running_var}
        :param name: default 'bo_op'
        '''

        Node.__init__(self, [X, gamma, beta])
        self.X = X
        self.gamma = gamma
        self.beta = beta
        self.name = name
        self.mode = bn_param['mode']
        self.eps = bn_param.get('eps', 1e-5)
        self.momentum = bn_param.get('momentum', 0.9)
        self.count = 0
        self.running_mean = bn_param.get('running_mean', np.zeros(2))
        self.running_var = bn_param.get('running_var', np.zeros(2))

    def forward(self):

        self.rou = self.inbound_nodes[0].rou
        if self.mode == 'train':
            dim = self.inbound_nodes[0].value.shape[1]
            sample_mean = self.X.value.mean(axis=0)
            sample_var = self.X.value.var(axis=0)

            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * sample_mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * sample_var

            self.std = np.sqrt(sample_var + self.eps)
            self.x_centered = self.X.value - sample_mean
            self.x_norm = self.x_centered / self.std
            out = self.gamma.value * self.x_norm + self.beta.value
            self.value = out

        elif self.mode == 'test':
            x_norm = (self.X.value - self.running_mean) / np.sqrt(self.running_var + self.eps)
            out = self.gamma.value * x_norm + self.beta
            self.value = out

        else:
            raise ValueError('Invalid forward batchnorm mode {}'.format(self.mode))

    def backward(self):
        self.gradients = {n: np.zeros_like(n.value) for n in self.inbound_nodes}
        for n in self.outbound_nodes:
            grad_cost = n.gradients[self]  # shape = self.value.shape = (batch_size, this layer neurons, 2)
            bs = grad_cost.shape[0]
            dl_gamma = np.einsum('bij,bij->ij', grad_cost, self.x_norm)
            dl_beta = np.sum(grad_cost, axis=0)
            dl_norm = np.einsum('bij,ij->bij', grad_cost, self.gamma.value)
            dl_centered = dl_norm / self.std
            dl_std = (dl_norm * self.x_centered * -self.std ** (-2)).sum(axis=0)
            dl_var = dl_std / 2 / self.std
            dl_mean = -(dl_centered.sum(axis=0)) - dl_var * (self.x_centered.sum(axis=0)) * 2 / bs
            dl_x = dl_centered + (dl_mean + dl_var * 2 * self.x_centered) / bs

            self.gradients[self.inbound_nodes[0]] += dl_x
            self.gradients[self.inbound_nodes[1]] += dl_gamma
            self.gradients[self.inbound_nodes[2]] += dl_beta


class cross_entropy_with_logits(Node):
    def __init__(self, logits, labels, name='cross_entropy'):
        Node.__init__(self, inbound_nodes=[logits, labels])
        self.name = name

    def forward(self):
        labels = self.inbound_nodes[1].value
        logits = self.inbound_nodes[0].value
        self.m = labels.shape[0]
        self.value = np.sum(-labels * np.log(logits) - (1 - labels) * np.log(1 - logits)) / self.m
        self.diff = (logits - labels) / (logits * (1 - logits))

    def backward(self):
        self.gradients[self.inbound_nodes[0]] = self.diff


class softmax_cross_entropy_with_logits(Node):
    '''
    it performs a softmax on logits internally for efficiency。
    '''

    def __init__(self, logits, labels, name='soft_cross_entropy'):
        """softmax loss function"""
        Node.__init__(self, inbound_nodes=[logits, labels])
        self.name = name

    @staticmethod
    def softmax(logits):
        logits = np.exp(logits)
        for i in range(logits.shape[0]):
            logits[i, :] = logits[i, :] / np.sum(logits[i, :])
        return logits

    def forward(self):
        labels = self.inbound_nodes[1].value
        logits = self.inbound_nodes[0].value
        logits_ = softmax_cross_entropy_with_logits.softmax(logits)
        self.value = np.sum(-labels * np.log(logits_) - (1 - labels) * np.log(1 - logits_))
        self.diff = logits_ - self.inbound_nodes[1].value

    def backward(self):
        self.gradients[self.inbound_nodes[0]] = self.diff
