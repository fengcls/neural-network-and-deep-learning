"""
Source Code for Homework 3.a of ECBM E6040, Spring 2016, Columbia University

This code is based on
[1] http://deeplearning.net/tutorial/logreg.html
[2] http://deeplearning.net/tutorial/mlp.html
[3] http://deeplearning.net/tutorial/lenet.html
"""
import numpy

import theano
import theano.tensor as T

from hw3_utils import shared_dataset, load_data
from hw3_nn import LogisticRegression, HiddenLayer, myMLP, LeNetConvPoolLayer, train_nn

# TODO
def noise_injection(signal,noise_level,noise_dist):
    mean_signal = signal.mean()
    # Standard deviation (spread or width) of the distribution
    if abs(mean_signal)<1e-7:
        scale_std = noise_level*1e-7
    else:
        scale_std = abs(noise_level*mean_signal)
    
    if scale_std>1e-15:
        if noise_dist=='normal':
            noise = numpy.random.normal(mean_signal,scale_std,signal.shape).astype(theano.config.floatX)
        else:
            noise = numpy.random.uniform(-mean_signal*scale_std,mean_signal*scale_std,signal.shape).astype(theano.config.floatX)
    else:
        noise = numpy.zeros(signal.shape).astype(theano.config.floatX)
    return noise#T.cast(noise,theano.config.floatX)

# TODO
def translate_image(signal,translate_direction,translate_len,pad_value):
    # reshape was carried out using C mode which is row first
    # with the last axis index changing fastest, back to the first axis index changing slowest
    num_pic,num_pixel = signal.shape
    edge_len = numpy.int(numpy.sqrt(num_pixel/3))
    signal = numpy.reshape(signal.T,(3,edge_len,edge_len,num_pic),order='C')
    signal = signal.transpose(1,2,0,3)
    
    if translate_direction == 'u':
        signal = numpy.concatenate(
        (signal[translate_len:,:,:,:],pad_value*numpy.ones((translate_len,edge_len,3,num_pic))),axis=0)
    elif translate_direction=='d':
        signal = numpy.concatenate(
        (pad_value*numpy.ones((translate_len,edge_len,3,num_pic)),signal[:-translate_len,:,:,:]),axis=0)
    elif translate_direction=='l':
        signal = numpy.concatenate(
        (signal[:,translate_len:,:,:],pad_value*numpy.ones((edge_len,translate_len,3,num_pic))),axis=1)
    elif translate_direction=='r':
        signal = numpy.concatenate(
        (pad_value*numpy.ones((edge_len,translate_len,3,num_pic)),signal[:,:-translate_len,:,:]),axis=1)
    
    signal = numpy.reshape(signal,(numpy.prod(signal.shape[:-1]), num_pic),order='C').T
    return signal

# TODO
def test_mlp(learning_rate=0.1, L1_reg=0.00, L2_reg=0.0001, n_epochs=100,
             batch_size=128, n_hidden=500, n_hiddenLayers=3,
             verbose=True, smaller_set=True):
    """
    Wrapper function for training and testing MLP

    :type learning_rate: float
    :param learning_rate: learning rate used (factor for the stochastic
    gradient.

    :type L1_reg: float
    :param L1_reg: L1-norm's weight when added to the cost (see
    regularization).

    :type L2_reg: float
    :param L2_reg: L2-norm's weight when added to the cost (see
    regularization).

    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer.

    :type batch_size: int
    :param batch_szie: number of examples in minibatch.

    :type n_hidden: int or list of ints
    :param n_hidden: number of hidden units. If a list, it specifies the
    number of units in each hidden layers, and its length should equal to
    n_hiddenLayers.

    :type n_hiddenLayers: int
    :param n_hiddenLayers: number of hidden layers.

    :type verbose: boolean
    :param verbose: to print out epoch summary or not to.

    :type smaller_set: boolean
    :param smaller_set: to use the smaller dataset or not to.

    """

    # load the dataset; download the dataset if it is not present
    if smaller_set:
        datasets = load_data(ds_rate=5)
    else:
        datasets = load_data()

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] // batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] // batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] // batch_size

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print('... building the model')

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch
    x = T.matrix('x')  # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of
                        # [int] labels

    rng = numpy.random.RandomState(1234)

    # TODO: construct a neural network, either MLP or CNN.
    classifier = myMLP(
        rng=rng,
        input=x,
        n_in=32*32*3,
        n_hidden=n_hidden,
        n_hiddenLayers=n_hiddenLayers,
        n_out=10
    )

    # the cost we minimize during training is the negative log likelihood of
    # the model plus the regularization terms (L1 and L2); cost is expressed
    # here symbolically
    cost = (
        classifier.negative_log_likelihood(y)
        + L1_reg * classifier.L1
        + L2_reg * classifier.L2_sqr
    )

    # compiling a Theano function that computes the mistakes that are made
    # by the model on a minibatch
    test_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: test_set_x[index * batch_size:(index + 1) * batch_size],
            y: test_set_y[index * batch_size:(index + 1) * batch_size]
        }
    )

    validate_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: valid_set_x[index * batch_size:(index + 1) * batch_size],
            y: valid_set_y[index * batch_size:(index + 1) * batch_size]
        }
    )

    # compute the gradient of cost with respect to theta (sotred in params)
    # the resulting gradients will be stored in a list gparams
    gparams = [T.grad(cost, param) for param in classifier.params]

    # specify how to update the parameters of the model as a list of
    # (variable, update expression) pairs

    # given two lists of the same length, A = [a1, a2, a3, a4] and
    # B = [b1, b2, b3, b4], zip generates a list C of same size, where each
    # element is a pair formed from the two lists :
    #    C = [(a1, b1), (a2, b2), (a3, b3), (a4, b4)]
    updates = [
        (param, param - learning_rate * gparam)
        for param, gparam in zip(classifier.params, gparams)
    ]

    # compiling a Theano function `train_model` that returns the cost, but
    # in the same time updates the parameter of the model based on the rules
    # defined in `updates`
    train_model = theano.function(
        inputs=[index],
        outputs=cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    ###############
    # TRAIN MODEL #
    ###############
    print('... training')

    train_nn(train_model, validate_model, test_model,
        n_train_batches, n_valid_batches, n_test_batches, n_epochs, verbose)

# TODO
def test_noise_injection_at_input(learning_rate=0.1, L1_reg=0.00,
             L2_reg=0.0001, n_epochs=100,
             batch_size=128, n_hidden=500, n_hiddenLayers=3,
             verbose=True,noise_level=0.01,noise_dist='normal'):
    """
    Wrapper function for experiment of noise injection at input

    :type learning_rate: float
    :param learning_rate: learning rate used (factor for the stochastic
    gradient.

    :type L1_reg: float
    :param L1_reg: L1-norm's weight when added to the cost (see
    regularization).

    :type L2_reg: float
    :param L2_reg: L2-norm's weight when added to the cost (see
    regularization).

    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer.

    :type batch_size: int
    :param batch_szie: number of examples in minibatch.

    :type n_hidden: int or list of ints
    :param n_hidden: number of hidden units. If a list, it specifies the
    number of units in each hidden layers, and its length should equal to
    n_hiddenLayers.

    :type n_hiddenLayers: int
    :param n_hiddenLayers: number of hidden layers.

    :type verbose: boolean
    :param verbose: to print out epoch summary or not to.

    :type smaller_set: boolean
    :param smaller_set: to use the smaller dataset or not to.

    :type noise_level: float
    :param noise_level: the percentage of the mean true signal, std

    """
    rng = numpy.random.RandomState(23455)

    # Load down-sampled dataset in raw format (numpy.darray, not Theano.shared)
    # train_set, valid_set, test_set format: tuple(input, target)
    # input is a numpy.ndarray of 2 dimensions (a matrix), where each row
    # corresponds to an example. target is a numpy.ndarray of 1 dimension
    # (vector) that has the same length as the number of rows in the input.

    # Load the smaller dataset in raw format, since we need to preprocess it
    train_set, valid_set, test_set = load_data(ds_rate=5, theano_shared=False)

    # Repeat the training set 5 times
    train_set[0] = numpy.tile(train_set[0], [5,1])
    train_set[1] = numpy.tile(train_set[1], 5)

    # TODO: injection noise to the training set
    # call noise_injection() ...
    train_set[0]=train_set[0] + noise_injection(train_set[0],noise_level,noise_dist)

    # Convert raw dataset to Theano shared variables.
    test_set_x, test_set_y = shared_dataset(test_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    train_set_x, train_set_y = shared_dataset(train_set)

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0]
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
    n_test_batches = test_set_x.get_value(borrow=True).shape[0]
    n_train_batches //= batch_size
    n_valid_batches //= batch_size
    n_test_batches //= batch_size

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch

    x = T.matrix('x')   # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of
                        # [int] labels

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print('... building the model')

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch
    x = T.matrix('x')  # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of
                        # [int] labels

    rng = numpy.random.RandomState(1234)

    classifier = myMLP(
        rng=rng,
        input=x,
        n_in=32*32*3,
        n_hidden=n_hidden,
        n_hiddenLayers=n_hiddenLayers,
        n_out=10
    )

    # the cost we minimize during training is the negative log likelihood of
    # the model plus the regularization terms (L1 and L2); cost is expressed
    # here symbolically
    cost = (
        classifier.negative_log_likelihood(y)
        + L1_reg * classifier.L1
        + L2_reg * classifier.L2_sqr
    )

    # compiling a Theano function that computes the mistakes that are made
    # by the model on a minibatch
    test_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: test_set_x[index * batch_size:(index + 1) * batch_size],
            y: test_set_y[index * batch_size:(index + 1) * batch_size]
        }
    )

    validate_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: valid_set_x[index * batch_size:(index + 1) * batch_size],
            y: valid_set_y[index * batch_size:(index + 1) * batch_size]
        }
    )

    # compute the gradient of cost with respect to theta (sotred in params)
    # the resulting gradients will be stored in a list gparams
    gparams = [T.grad(cost, param) for param in classifier.params]

    # specify how to update the parameters of the model as a list of
    # (variable, update expression) pairs

    # given two lists of the same length, A = [a1, a2, a3, a4] and
    # B = [b1, b2, b3, b4], zip generates a list C of same size, where each
    # element is a pair formed from the two lists :
    #    C = [(a1, b1), (a2, b2), (a3, b3), (a4, b4)]
    updates = [
        (param, param - learning_rate * gparam)
        for param, gparam in zip(classifier.params, gparams)
    ]

    # compiling a Theano function `train_model` that returns the cost, but
    # in the same time updates the parameter of the model based on the rules
    # defined in `updates`
    train_model = theano.function(
        inputs=[index],
        outputs=cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    ###############
    # TRAIN MODEL #
    ###############
    print('... training')

    train_nn(train_model, validate_model, test_model,
        n_train_batches, n_valid_batches, n_test_batches, n_epochs, verbose)

# TODO
def test_noise_injection_at_weight(learning_rate=0.1,
             L1_reg=0.00, L2_reg=0.0001, n_epochs=100,
             batch_size=128, n_hidden=500, n_hiddenLayers=3,
             verbose=True,noise_level=0.001,noise_dist='uniform'):
    """
    Wrapper function for experiment of noise injection at weights

    :type learning_rate: float
    :param learning_rate: learning rate used (factor for the stochastic
    gradient.

    :type L1_reg: float
    :param L1_reg: L1-norm's weight when added to the cost (see
    regularization).

    :type L2_reg: float
    :param L2_reg: L2-norm's weight when added to the cost (see
    regularization).

    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer.

    :type batch_size: int
    :param batch_szie: number of examples in minibatch.

    :type n_hidden: int or list of ints
    :param n_hidden: number of hidden units. If a list, it specifies the
    number of units in each hidden layers, and its length should equal to
    n_hiddenLayers.

    :type n_hiddenLayers: int
    :param n_hiddenLayers: number of hidden layers.

    :type verbose: boolean
    :param verbose: to print out epoch summary or not to.

    :type smaller_set: boolean
    :param smaller_set: to use the smaller dataset or not to.

    """
    rng = numpy.random.RandomState(23455)

    # Load down-sampled dataset in raw format (numpy.darray, not Theano.shared)
    # train_set, valid_set, test_set format: tuple(input, target)
    # input is a numpy.ndarray of 2 dimensions (a matrix), where each row
    # corresponds to an example. target is a numpy.ndarray of 1 dimension
    # (vector) that has the same length as the number of rows in the input.

    # Load the smaller dataset
    datasets = load_data(ds_rate=5)

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0]
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
    n_test_batches = test_set_x.get_value(borrow=True).shape[0]
    n_train_batches //= batch_size
    n_valid_batches //= batch_size
    n_test_batches //= batch_size

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch

    x = T.matrix('x')   # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of
                        # [int] labels

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print('... building the model')

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch
    x = T.matrix('x')  # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of
                        # [int] labels

    rng = numpy.random.RandomState(1234)

    classifier = myMLP(
        rng=rng,
        input=x,
        n_in=32*32*3,
        n_hidden=n_hidden,
        n_hiddenLayers=n_hiddenLayers,
        n_out=10
    )

    # the cost we minimize during training is the negative log likelihood of
    # the model plus the regularization terms (L1 and L2); cost is expressed
    # here symbolically
    cost = (
        classifier.negative_log_likelihood(y)
        + L1_reg * classifier.L1
        + L2_reg * classifier.L2_sqr
    )

    # compiling a Theano function that computes the mistakes that are made
    # by the model on a minibatch
    test_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: test_set_x[index * batch_size:(index + 1) * batch_size],
            y: test_set_y[index * batch_size:(index + 1) * batch_size]
        }
    )

    validate_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: valid_set_x[index * batch_size:(index + 1) * batch_size],
            y: valid_set_y[index * batch_size:(index + 1) * batch_size]
        }
    )

    # compute the gradient of cost with respect to theta (stored in params)
    # the resulting gradients will be stored in a list gparams
    gparams = [T.grad(cost, param) for param in classifier.params]

    # specify how to update the parameters of the model as a list of
    # (variable, update expression) pairs

    # given two lists of the same length, A = [a1, a2, a3, a4] and
    # B = [b1, b2, b3, b4], zip generates a list C of same size, where each
    # element is a pair formed from the two lists :
    #    C = [(a1, b1), (a2, b2), (a3, b3), (a4, b4)]

    # TODO: modify updates to inject noise to the weight
    #    # the parameters of the model are the parameters of the two layer it is made out of
    #    self.params = sum([x.params for x in self.hiddenLayers], []) + self.logRegressionLayer.params
    #        # parameters of hiddenlayer and logRegressionLayer
    #        self.params = [self.W, self.b]

    updates = [
    # W b W b W b layernumx2
#        (classifier.params[0::2], classifier.params[0::2] - learning_rate * gparams[0::2]),
#        (classifier.params[1::2], classifier.params[1::2] - learning_rate * gparams[1::2])
#        (param, param - learning_rate * gparam)
#        for param, gparam in zip(classifier.params, gparams)
        (param, param - learning_rate * gparam + noise_injection(param.get_value(),noise_level,noise_dist))
        for param, gparam in zip(classifier.params[0::2], gparams[0::2]) 
        +
        [(param, param - learning_rate * gparam)
        for param, gparam in zip(classifier.params[1::2], gparams[1::2])]
        
    ]

    # compiling a Theano function `train_model` that returns the cost, but
    # in the same time updates the parameter of the model based on the rules
    # defined in `updates`
    train_model = theano.function(
        inputs=[index],
        outputs=cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    ###############
    # TRAIN MODEL #
    ###############
    print('... training')

    train_nn(train_model, validate_model, test_model,
        n_train_batches, n_valid_batches, n_test_batches, n_epochs, verbose)

def test_data_augmentation(learning_rate=0.1,
             L1_reg=0.00, L2_reg=0.0001, n_epochs=100,
             batch_size=128, n_hidden=500, n_hiddenLayers=3,
             verbose=True):
    """
    Wrapper function for experiment of data augmentation

    :type learning_rate: float
    :param learning_rate: learning rate used (factor for the stochastic
    gradient.

    :type L1_reg: float
    :param L1_reg: L1-norm's weight when added to the cost (see
    regularization).

    :type L2_reg: float
    :param L2_reg: L2-norm's weight when added to the cost (see
    regularization).

    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer.

    :type batch_size: int
    :param batch_szie: number of examples in minibatch.

    :type n_hidden: int or list of ints
    :param n_hidden: number of hidden units. If a list, it specifies the
    number of units in each hidden layers, and its length should equal to
    n_hiddenLayers.

    :type n_hiddenLayers: int
    :param n_hiddenLayers: number of hidden layers.

    :type verbose: boolean
    :param verbose: to print out epoch summary or not to.

    :type smaller_set: boolean
    :param smaller_set: to use the smaller dataset or not to.

    """
    rng = numpy.random.RandomState(23455)

    # Load down-sampled dataset in raw format (numpy.darray, not Theano.shared)
    # train_set, valid_set, test_set format: tuple(input, target)
    # input is a numpy.ndarray of 2 dimensions (a matrix), where each row
    # corresponds to an example. target is a numpy.ndarray of 1 dimension
    # (vector) that has the same length as the number of rows in the input.

    # Load the smaller dataset in raw Format, since we need to preprocess it
    train_set, valid_set, test_set = load_data(ds_rate=5, theano_shared=False)

    # Repeat the training set 5 times
    train_set[1] = numpy.tile(train_set[1], 5)

    # TODO: translate the dataset
    train_set_x_u = translate_image(train_set[0],'u',1,0.0)
    train_set_x_d = translate_image(train_set[0],'d',1,0.0)
    train_set_x_r = translate_image(train_set[0],'r',1,0.0)
    train_set_x_l = translate_image(train_set[0],'l',1,0.0)
    
    from matplotlib import pyplot as plt
    plt.figure(),plt.imshow(train_set[0][0,:].reshape([3,32,32]).transpose((1,2,0))),plt.title('original')
    plt.figure(),plt.imshow(train_set_x_l[0,:].reshape([32,32,3])),plt.title('translated left')
    plt.figure(),plt.imshow(train_set_x_r[0,:].reshape([32,32,3])),plt.title('translated right')
    plt.figure(),plt.imshow(train_set_x_u[0,:].reshape([32,32,3])),plt.title('translated up')
    plt.figure(),plt.imshow(train_set_x_d[0,:].reshape([32,32,3])),plt.title('translated down')
    
    # Stack the original dataset and the synthesized datasets
    train_set[0] = numpy.vstack((train_set[0],
                       train_set_x_u,
                       train_set_x_d,
                       train_set_x_r,
                       train_set_x_l))
    random_ind = numpy.random.permutation(train_set[0].shape[0])
    train_set[0] = train_set[0][random_ind,:]
    train_set[1] = train_set[1][random_ind]

    # Convert raw dataset to Theano shared variables.
    test_set_x, test_set_y = shared_dataset(test_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    train_set_x, train_set_y = shared_dataset(train_set)

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0]
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
    n_test_batches = test_set_x.get_value(borrow=True).shape[0]
    n_train_batches //= batch_size
    n_valid_batches //= batch_size
    n_test_batches //= batch_size

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch

    x = T.matrix('x')   # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of
                        # [int] labels

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print('... building the model')

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch
    x = T.matrix('x')  # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of
                        # [int] labels

    rng = numpy.random.RandomState(1234)

    classifier = myMLP(
        rng=rng,
        input=x,
        n_in=32*32*3,
        n_hidden=n_hidden,
        n_hiddenLayers=n_hiddenLayers,
        n_out=10
    )

    # the cost we minimize during training is the negative log likelihood of
    # the model plus the regularization terms (L1 and L2); cost is expressed
    # here symbolically
    cost = (
        classifier.negative_log_likelihood(y)
        + L1_reg * classifier.L1
        + L2_reg * classifier.L2_sqr
    )

    # compiling a Theano function that computes the mistakes that are made
    # by the model on a minibatch
    test_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: test_set_x[index * batch_size:(index + 1) * batch_size],
            y: test_set_y[index * batch_size:(index + 1) * batch_size]
        }
    )

    validate_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: valid_set_x[index * batch_size:(index + 1) * batch_size],
            y: valid_set_y[index * batch_size:(index + 1) * batch_size]
        }
    )

    # compute the gradient of cost with respect to theta (sotred in params)
    # the resulting gradients will be stored in a list gparams
    gparams = [T.grad(cost, param) for param in classifier.params]

    # specify how to update the parameters of the model as a list of
    # (variable, update expression) pairs

    # given two lists of the same length, A = [a1, a2, a3, a4] and
    # B = [b1, b2, b3, b4], zip generates a list C of same size, where each
    # element is a pair formed from the two lists :
    #    C = [(a1, b1), (a2, b2), (a3, b3), (a4, b4)]
    updates = [
        (param, param - learning_rate * gparam)
        for param, gparam in zip(classifier.params, gparams)
    ]

    # compiling a Theano function `train_model` that returns the cost, but
    # in the same time updates the parameter of the model based on the rules
    # defined in `updates`
    train_model = theano.function(
        inputs=[index],
        outputs=cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    ###############
    # TRAIN MODEL #
    ###############
    print('... training')

    train_nn(train_model, validate_model, test_model,
        n_train_batches, n_valid_batches, n_test_batches, n_epochs, verbose)

# TODO
def test_adversarial_example(learning_rate=0.1, L1_reg=0.00, L2_reg=0.0001, n_epochs=100,
             batch_size=128, n_hidden=500, n_hiddenLayers=3,
             verbose=True, smaller_set=True):
    """
    Wrapper function for testing adversarial examples
    """
    # load the dataset; download the dataset if it is not present
    if smaller_set:
        datasets = load_data(ds_rate=5)
    else:
        datasets = load_data()

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

#    test_set_x = test_set_x[0:1]
#    test_set_y = test_set_y[0:1]
    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] // batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] // batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] // batch_size

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print('... building the model')

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch
    x = T.matrix('x')  # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of
                        # [int] labels

    rng = numpy.random.RandomState(1234)

    # TODO: construct a neural network, either MLP or CNN.
    classifier = myMLP(
        rng=rng,
        input=x,
        n_in=32*32*3,
        n_hidden=n_hidden,
        n_hiddenLayers=n_hiddenLayers,
        n_out=10
    )


    # the cost we minimize during training is the negative log likelihood of
    # the model plus the regularization terms (L1 and L2); cost is expressed
    # here symbolically
    cost = (
        classifier.negative_log_likelihood(y)
        + L1_reg * classifier.L1
        + L2_reg * classifier.L2_sqr
    )

    # compiling a Theano function that computes the mistakes that are made
    # by the model on a minibatch

    validate_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: valid_set_x[index * batch_size:(index + 1) * batch_size],
            y: valid_set_y[index * batch_size:(index + 1) * batch_size]
        }
    )
    
    test_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: test_set_x[index * batch_size:(index + 1) * batch_size],
            y: test_set_y[index * batch_size:(index + 1) * batch_size]
        }
    )

    test_model_single = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: test_set_x[index:index+1],
            y: test_set_y[index:index+1]
        }
    )


    # compute the gradient of cost with respect to theta (sotred in params)
    # the resulting gradients will be stored in a list gparams
    gparams = [T.grad(cost, param) for param in classifier.params]

    # specify how to update the parameters of the model as a list of
    # (variable, update expression) pairs

    # given two lists of the same length, A = [a1, a2, a3, a4] and
    # B = [b1, b2, b3, b4], zip generates a list C of same size, where each
    # element is a pair formed from the two lists :
    #    C = [(a1, b1), (a2, b2), (a3, b3), (a4, b4)]
    updates = [
        (param, param - learning_rate * gparam)
        for param, gparam in zip(classifier.params, gparams)
    ]

    # compiling a Theano function `train_model` that returns the cost, but
    # in the same time updates the parameter of the model based on the rules
    # defined in `updates`
    train_model = theano.function(
        inputs=[index],
        outputs=cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    ###############
    # TRAIN MODEL #
    ###############
    print('... training')
    
    gx = T.grad(cost, x)

    train_nn(train_model, validate_model, test_model,
       n_train_batches, n_valid_batches, n_test_batches, n_epochs, verbose)
    
    f = theano.function(
        inputs=[index],
        outputs=gx,
        givens={
            x: test_set_x[index : (index + 1)],
            y: test_set_y[index : (index + 1)]
        }
    )
    ind_oi = 3

    from matplotlib import pyplot as plt
    plt.figure()
    plt.imshow(test_set_x.get_value()[ind_oi,:].reshape(3,32,32).transpose((1,2,0)))

    h = theano.function(
        inputs=[index],
        outputs=classifier.logRegressionLayer.y_pred,
        givens={
            x: test_set_x[index : (index + 1)]
        }
    )

    print('predicted number original: %i' % h(ind_oi))	
    
    Y = T.matrix()
    X_update = (test_set_x, T.inc_subtensor(test_set_x[ind_oi:(ind_oi+1)], Y))
    g = theano.function([Y], updates=[X_update])
    g(0.01*numpy.sign(f(ind_oi)))

    print('predicted number adverserial: %i' % h(ind_oi))
    
    plt.figure()
    plt.imshow(test_set_x.get_value()[ind_oi,:].reshape(3,32,32).transpose((1,2,0)))
    
    
if __name__ == "__main__":
    # test_mlp()
    # test_noise_injection_at_weight()
    # test_noise_injection_at_input()
    # test_data_augmentation()
    test_adversarial_example()
