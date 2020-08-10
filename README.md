# mnn_box
A python package used for reproducing the moment mapping neural network on cpus using `numpy` functionality.

Moment mapping nn is a spiking neural nn which can implement feedforward process based  concentrating on 
some statistics of spike pattern. Inspired from deep learning, we aim to construct a complete mnn framework and 
compare the effects with mlp based on `pytorch`.

> Feng, Jianfeng, Yingchun Deng, and Enrico Rossoni. "Dynamics of moment neuronal networks." Physical Review E 73.4 (2006): 041906.

>Lu, Wenlian, Enrico Rossoni, and Jianfeng Feng. "On a Gaussian neuronal field model." NeuroImage 52.3 (2010): 913-933.

## Requirement
- python3.7
- mnn_box package

# Getting started
we choose the mnist classification task to test the learning power. It's analogous 
to the `tensorflow`:
+ import mnn_box as mnn
+ build the computational graph
+ run the nn and test the accuracy

# results
The MNIST database of handwritten digits, has a training set of 60,000 examples,
and a test set of 10,000 examples. we test the 1 hidden layer mnn's performance, and the following 
picture show the loss and accuracy during training and predicting.
![train_loss](mnist_classification\loss.png)
![train_loss](mnist_classification\accuracy.png)
