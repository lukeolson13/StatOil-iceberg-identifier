import json
import numpy as np
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import SGD, RMSprop, Adagrad, Adadelta, Adam
import theano

def load_and_condition_MNIST_data():
    ''' loads and shapes MNIST image data '''
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    # print("\nLoaded MNIST images")
    theano.config.floatX = 'float32'
    X_train = X_train.astype(theano.config.floatX) #before conversion were uint8
    X_test = X_test.astype(theano.config.floatX)
    X_train.resize(len(y_train), 784) # 28 pix x 28 pix = 784 pixels
    X_test.resize(len(y_test), 784)
    # print('\nFirst 5 labels of MNIST y_train: {}'.format(y_train[:5]))
    y_train_ohe = np_utils.to_categorical(y_train)
    # print('\nFirst 5 labels of MNIST y_train (one-hot):\n{}'.format(y_train_ohe[:5]))
    # print()
    return X_train, y_train, X_test, y_test, y_train_ohe

def define_nn_mlp_model(X_train, y_train_ohe, activation, neurons, optimizer):
    ''' defines multi-layer-perceptron neural network '''
    # available activation functions at:
    # https://keras.io/activations/
    # https://en.wikipedia.org/wiki/Activation_function
    # options: 'linear', 'sigmoid', 'tanh', 'relu', 'softplus', 'softsign'
    # there are other ways to initialize the weights besides 'uniform', too

    model = Sequential() # sequence of layers
    num_neurons_in_layer = neurons # number of neurons in a layer
    num_inputs = X_train.shape[1] # number ofneurons features (784)
    num_classes = y_train_ohe.shape[1]  # number of classes, 0-9
    model.add(Dense(units=num_neurons_in_layer,
                    input_dim=num_inputs,
                    kernel_initializer='uniform',
                    activation=activation)) # only 12 neurons in this layer!
    model.add(Dense(units=num_classes,
                    input_dim=num_neurons_in_layer,
                    kernel_initializer='uniform',
                    activation=activation)) # only 12 neurons - keep softmax at last layer
    #sgd = SGD(lr=lr, decay=1e-7, momentum=0) # using stochastic gradient descent (keep)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=["accuracy"] ) # (keep)
    return model

def print_output(model, y_train, y_test, rng_seed):
    '''prints model accuracy results'''
    y_train_pred = model.predict_classes(X_train, verbose=0)
    y_test_pred = model.predict_classes(X_test, verbose=0)
    # print('\nRandom number generator seed: {}'.format(rng_seed))
    # print('\nFirst 30 labels:      {}'.format(y_train[:30]))
    # print('First 30 predictions: {}'.format(y_train_pred[:30]))
    train_acc = np.sum(y_train == y_train_pred, axis=0) / X_train.shape[0]
    # print('\nTraining accuracy: %.2f%%' % (train_acc * 100))
    test_acc = np.sum(y_test == y_test_pred, axis=0) / X_test.shape[0]
    # print('Test accuracy: %.2f%%' % (test_acc * 100))
    # if test_acc < 0.95:
    #     print('\nMan, your test accuracy is bad!')
    #     print("Can't you get it up to 95%?")
    # else:
    #     print("\nYou've made some improvements, I see...")
    return train_acc, test_acc


if __name__ == '__main__':
    np.random.seed(1) # set random number generator seed
    json_dict = json.loads(json_input)
    X_train, y_train, X_test, y_test, y_train_ohe = 
    
    #activation_arr = ['linear', 'sigmoid', 'tanh', 'relu', 'softplus', 'softsign']
    #epochs = np.arange(1, 10, 1)
    #lr_arr = np.logspace(-4,1, 6)
    # opt_arr = [ SGD(lr=.001, decay=1e-7, momentum=0), 
    #             RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0),
    #             Adagrad(lr=0.001, epsilon=1e-08, decay=0.0),
    #             Adadelta(lr=1, rho=0.95, epsilon=1e-08, decay=0.0)]
    acc_arr = []
    # for lr in lr_arr:
    model = define_nn_mlp_model(X_train, y_train_ohe, activation='softplus', neurons=784, optimizer=Adadelta(lr=0.1, rho=0.95, epsilon=1e-08, decay=0.0))
    model.fit(X_train, y_train_ohe, epochs=6, batch_size=None, verbose=1,
              validation_split=0.1) # cross val to estimate test error
    print(print_output(model, y_train, y_test, rng_seed))
    # train_acc, test_acc = print_output(model, y_train, y_test, rng_seed)
    # acc_arr.append([train_acc, test_acc])
    # for a, b in zip(lr_arr, acc_arr):
    #     print('{} - train: {} - test: {}'.format(a, b[0], b[1]))

"""
From tests:
- Best activation function is softplus
- Optimal epochs is 7
- Optimal momentum is 0
"""